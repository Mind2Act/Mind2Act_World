"""F2 source state machine; no simulator, expert, or source package dependency.

Ported from pinned Session and F2 Task. Native reset has zero physical samples;
the separate source clock retains its recorded origin. Stability caches per tick.
"""
import math
from copy import deepcopy
from .geometry import Body, EvidenceError, in_region, supported, uses_shared_station
from .provenance import validate_source_config
import json
from pathlib import Path
INSTRUCTIONS = {"F2": json.loads((Path(__file__).with_name("config") / "instructions.json").read_text())["entries"][0]["en"]}

class Session:
    def __init__(self, config, *, source_start_tick=0):
        validate_source_config(config)
        if type(source_start_tick) is not int or source_start_tick < 0:
            raise ValueError('F2 source start tick must be a nonnegative integer')
        self.source_start_tick = source_start_tick
        self.physics_samples = 0
        self.config = deepcopy(config)
        from mind2act_eval.runner.observation import task_deadline_with_observation
        self.trial_tick_budget = task_deadline_with_observation('F2', config)
        self.tick = -1
        self.bodies = {}
        self.status = 'running'
        self.reason = None
        self.events = []
        self._stable = {}
        self._locations_tick = -2
        self._locations = {}
        self._active_locations = {}
        self._cue = None
        self._cue_until = -1
        self._feedback = None
        self._feedback_until = -1
        self._initial_centers = None
        self.start()
        # A recorded projection may begin at source tick 600. This is a logical
        # source-clock offset, never a claim of 600 new simulation samples.
        self.tick = source_start_tick - 1

    def start(self):
        raise NotImplementedError

    def notice(self, data):
        self._cue = deepcopy(data)
        self._cue_until = max(0, self.tick) + self.config['cue_ticks']
        self.event('notice', content=data, until=self._cue_until)

    @property
    def action_enabled(self):
        return self.status == 'running' and self.tick >= self._cue_until

    def public(self):
        result = dict(task=self.config['task'], instruction=INSTRUCTIONS[self.config['task']],
                      action_enabled=self.action_enabled, done=self.status != 'running')
        result['panel'] = deepcopy(self._cue) if self.tick < self._cue_until else None
        result['feedback'] = deepcopy(self._feedback) if self.tick < self._feedback_until else None
        return result

    def event(self, kind, **fields):
        self.events.append(dict(tick=max(0, self.tick), kind=kind, **deepcopy(fields)))

    def fail(self, reason):
        if self.status == 'running':
            self.status, self.reason = 'failure', reason
            self.event('failure', reason=reason)

    def succeed(self):
        if self.status == 'running':
            self.status = 'success'
            self.event('success')

    def sample(self, frame):
        if self.status != 'running':
            raise EvidenceError('Evidence after terminal state')
        try:
            if type(frame['tick']) is not int or frame['tick'] != self.tick+1:
                raise EvidenceError('Need every consecutive 120 Hz physics tick, starting at 0')
            if set(frame['bodies']) != set(self.config['objects']):
                raise EvidenceError('Missing/extra body measurements')
            bodies = {name: Body.parse(row) for name, row in frame['bodies'].items()}
            if uses_shared_station(self.config) and any(b.supports is None for b in bodies.values()):
                raise EvidenceError('Aligned tasks require measured contact-point support evidence')
            self.tick = frame['tick']
            if self._initial_centers is not None:
                self.physics_samples += 1
            self.bodies = bodies
            if self._initial_centers is None:
                self._initial_centers = {k: b.center for k, b in bodies.items()}
                for name, body in bodies.items():
                    want = self.config['objects'][name]['position']
                    if any(abs(body.center[i]-want[i]) > .005 for i in range(3)):
                        raise EvidenceError('Initial object layout differs from bound case')
            # Notice frames and physical actions stay observable. Contact during
            # the notice does not terminate the trial or reset measured bodies.
            if self.status == 'running':
                self.advance(frame)
        except (KeyError, TypeError, ValueError) as exc:
            self.status, self.reason = 'invalid', str(exc)
            raise EvidenceError(str(exc)) from exc
        return self.status

    def stable_locations(self):
        if self._locations_tick == self.tick:
            return dict(self._locations)
        result = {}
        for name, body in self.bodies.items():
            matches = []
            for region_id, region in self.config['regions'].items():
                eligible = in_region(body, region) and supported(name, region_id, self.bodies, self.config)
                if eligible:
                    matches.append(region_id)
            if len(matches) > 1:
                raise EvidenceError('Ambiguous overlapping receiver regions')
            if matches:
                result[name] = matches[0]
        self._locations_tick = self.tick
        self._locations = dict(result)
        return result

    def placement_events(self, locations):
        new = {name: region for name, region in locations.items() if self._active_locations.get(name) != region}
        self._active_locations = dict(locations)
        return new

    def advance(self, frame):
        raise NotImplementedError

    def result(self):
        return dict(task=self.config['task'], difficulty=self.config['difficulty'],
                    case_sha256=self.config['case_sha256'], version=self.config['version'],
                    status=self.status, success=self.status == 'success',
                    score=100 if self.status == 'success' else 0 if self.status == 'failure' else None,
                    reason=self.reason, source_tick=self.tick, source_start_tick=self.source_start_tick,
                    physics_samples=self.physics_samples, events=deepcopy(self.events),
                    formal_eligible=False)


class Task(Session):
    def start(self):
        self.attempts = []
        self.armed = False
        self.press_ticks = 0
        self.pending_release = False
        self.pending_trial = None
        self.notice(dict(type='rules', block_count=len(self.config['objects']),
                         attempt_budget=self.config['attempt_budget'], feedback='exact_position_count'))

    def advance(self, frame):
        locations = self.stable_locations()
        button = frame.get('button')
        if not isinstance(button, dict) or type(button.get('robot_contact')) is not bool:
            raise EvidenceError('Missing physical confirm-button contact')
        depth = button.get('depth')
        if type(depth) not in (int, float) or not math.isfinite(depth):
            raise EvidenceError('Missing physical confirm-button travel')
        spec = self.config['button']
        if depth < -.002 or depth > spec['travel']+.003:
            raise EvidenceError('Button travel outside mechanical limits')
        if self.physics_samples == 0:
            self.armed = depth <= spec['release_depth']
            self.press_ticks = 0
            return
        if not self.action_enabled:
            self.press_ticks = 0
            if depth >= spec['press_depth']:
                self.armed = False
            elif depth <= spec['release_depth']:
                self.armed = True
            return
        if self.pending_release:
            if depth<=spec['release_depth']:
                self.pending_release=False
                # Rearm on this actual release frame; an extra idle frame is not
                # part of the protocol. Stable placement remains measured each tick.
                self.armed=True
                self.press_ticks=0
                slots=list(self.config['regions'])
                if len(locations)==len(slots) and set(locations.values())==set(slots):
                    permutation=[next(k for k,v in locations.items() if v==slot) for slot in slots]
                    correct=sum(a==b for a,b in zip(permutation,self.config['target']))
                    self.attempts.append(permutation)
                    self.event('confirmed_trial',permutation=permutation,correct=correct,release_tick=self.tick)
                    self._feedback=dict(correct_positions=correct,total=len(slots))
                    self._feedback_until=self.tick+self.config['feedback_ticks']
                    if correct==len(slots):self.succeed()
                    elif len(self.attempts)>=self.config['attempt_budget']:self.fail('attempt_budget_exhausted')
                else:
                    self.event('incomplete_confirm_released')
                self.pending_trial=None
            return
        if depth <= spec['release_depth']:
            self.armed = True
            self.press_ticks = 0
        elif depth >= spec['press_depth'] and self.action_enabled:
            self.press_ticks += 1
        else:
            self.press_ticks = 0
        if not self.action_enabled or not self.armed or self.press_ticks < 1:
            return
        self.armed = False
        slots = list(self.config['regions'])
        # Incomplete layouts do not submit an experiment or consume budget.
        if len(locations) != len(slots) or set(locations.values()) != set(slots):
            return
        permutation = [next(k for k, v in locations.items() if v == slot) for slot in slots]
        self.pending_trial=permutation
        self.pending_release=True
        self.event('confirm_pressed',press_tick=self.tick)

    def result(self):
        result = super().result()
        result['diagnostics'] = dict(attempts=len(self.attempts), repeated_trials=len(self.attempts)-len({tuple(x) for x in self.attempts}))
        return result
