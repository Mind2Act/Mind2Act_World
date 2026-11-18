"""Scoring-only Session subset ported from locked source; no old runtime."""
from copy import deepcopy
from ..binding import validate_config as validate_case
from .body import Body, EvidenceError
def uses_shared_station(config): return True
INSTRUCTIONS = {'S2': 'Remember the briefly highlighted forbidden cells. Push the slider to GOAL while keeping its whole footprint on the board and out of forbidden cells. Never lift it. Wait until the map is hidden.'}
class Session:

    def __init__(self, config):
        validate_case(config)
        self.config = deepcopy(config)
        from mind2act_eval.runner.observation import task_deadline_with_observation
        self.trial_tick_budget = task_deadline_with_observation('S2', config)
        self.tick = -1
        self.bodies = {}
        self.status = 'running'
        self.reason = None
        self.violations = set()
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

    def notice(self, data):
        self._cue = deepcopy(data)
        self._cue_until = max(0, self.tick) + self.config['cue_ticks']
        self.event('notice', content=data, until=self._cue_until)

    @property
    def action_enabled(self):
        return self.status == 'running' and self.tick >= self._cue_until

    def public(self):
        result = dict(task=self.config['task'], instruction=INSTRUCTIONS[self.config['task']], action_enabled=self.action_enabled, done=self.status != 'running')
        result['panel'] = deepcopy(self._cue) if self.tick < self._cue_until else None
        result['feedback'] = deepcopy(self._feedback) if self.tick < self._feedback_until else None
        return result

    def event(self, kind, **fields):
        self.events.append(dict(tick=max(0, self.tick), kind=kind, **deepcopy(fields)))

    def fail(self, reason):
        if reason not in self.violations:
            self.violations.add(reason)
            self.reason = reason
            self.event('violation', reason=reason)

    def succeed(self):
        if self.status == 'running':
            self.status = 'success'
            self.event('success')

    def sample(self, frame):
        if self.status != 'running':
            raise EvidenceError('Evidence after terminal state')
        try:
            if type(frame['tick']) is not int or frame['tick'] != self.tick + 1:
                raise EvidenceError('Need every consecutive 120 Hz physics tick, starting at 0')
            if set(frame['bodies']) != set(self.config['objects']):
                raise EvidenceError('Missing/extra body measurements')
            bodies = {name: Body.parse(row) for name, row in frame['bodies'].items()}
            if uses_shared_station(self.config) and any((b.supports is None for b in bodies.values())):
                raise EvidenceError('Aligned tasks require measured contact-point support evidence')
            self.tick = frame['tick']
            self.bodies = bodies
            if self._initial_centers is None:
                self._initial_centers = {k: b.center for k, b in bodies.items()}
                for name, body in bodies.items():
                    want = self.config['objects'][name]['position']
                    if any((abs(body.center[i] - want[i]) > 0.005 for i in range(3))):
                        raise EvidenceError('Initial object layout differs from bound case')
            # Observation remains live during the cue. Contacts there are
            # physical state, not a terminal task violation.
            if self.status == 'running':
                self.advance(frame)
        except (KeyError, TypeError, ValueError) as exc:
            self.status, self.reason = ('invalid', str(exc))
            raise EvidenceError(str(exc)) from exc
        return self.status

    def result(self):
        return dict(task=self.config['task'], difficulty=self.config['difficulty'], case_sha256=self.config['case_sha256'], version=self.config['version'], status=self.status, success=self.status == 'success', score=100 if self.status == 'success' else 0 if self.status == 'failure' else None, reason=self.reason, physics_ticks=self.tick + 1, events=deepcopy(self.events), formal_eligible=False)
