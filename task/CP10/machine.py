"""Source arena clock plus independent measured receipt fields, CPU-only draft.

No robot/controller, renderer, physics step or body mutation belongs here.
The source clock chooses mechanism targets; scoring reads only the receipts.
"""
from copy import deepcopy
import math

from mind2act_eval.contracts import to_json
from mind2act_eval.tasks.CP10 import CONTRACT
from mind2act_eval.tasks.CP10.evidence import validate_schedule
from mind2act_eval.tasks.CP10.protocol import PROTOCOL_ID, RETENTION, digest, validate
from mind2act_eval.tasks.CP10.retention import BallBounds, contained_net
from .mechanism.arena_cycle import ShotCycle, NetReceipt, sample_heading
from .mechanism.arena_preparation import launch_pose


class ArenaMachine:
    def __init__(self, config, protocol, rule_version):
        from mind2act_eval.runner.observation import task_deadline_with_observation
        self.observation_deadline = task_deadline_with_observation('CP10', config)
        validate(protocol, rule_version)
        self.config = to_json(config)
        self.colors = tuple(self.config['arena']['shot_colors'])
        validate_schedule(self.config['difficulty'], self.colors)
        a = self.config['arena']
        if a['net_settle_ticks'] != 60 or a['per_shot_timeout_ticks'] != 2400:
            raise ValueError('CP10 selected collection/timeout contract differs')
        self.tick = -1
        self.status, self.reason, self.terminal_tick = 'running', None, None
        self.failure = None
        self.final_nets = {}
        self.cycle = None
        self.expected_emitted, self.shots = [], []
        self.paddle_contact, self.net_ticks, self.net_name = False, 0, None
        self.fired_tick = None
        self.hardware = dict(phase='resetting', shot_number=1, practice=True,
                             score_eligible=False, target_yaw_deg=sample_heading(self.config, 0),
                             target_pitch_deg=0., countdown=None, release_now=False,
                             ball_color=None)
        if 'slide_limits_m' in a['launcher']:
            from .mechanism.mobile_launcher import sample_pose
            self.hardware['target_slide_x_m'] = sample_pose(self.config, 0)[0]

    def _finish(self, reason, failure=False):
        self.status = 'success' if reason == 'completed' else 'failure'
        self.reason, self.terminal_tick = reason, self.tick
        if failure:
            self.failure = dict(tick=self.tick, reason=reason)

    def sample(self, frame):
        tick = frame['tick']
        if type(tick) is not int or tick != self.tick + 1:
            raise ValueError('CP10 needs consecutive samples from reset tick zero')
        if set(frame['bodies']) != set(self.config['objects']):
            raise ValueError('CP10 body inventory differs from selected source')
        if frame['emitted_balls'] != sorted(self.expected_emitted):
            raise ValueError('CP10 measured emissions differ from native release clock')
        bodies = {}
        for name, row in frame['bodies'].items():
            contacts = row['contacts']
            if not isinstance(contacts, (list, tuple)) or any(type(p) is not str for p in contacts):
                raise ValueError('CP10 contacts must be measured collider paths')
            lo, hi = tuple(row['lower']), tuple(row['upper'])
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (*lo, *hi)):
                raise ValueError('CP10 bounds must be finite measured numbers')
            if any(x >= y for x, y in zip(lo, hi)):
                raise ValueError('CP10 degenerate measured body bounds')
            center = tuple((x + y) / 2 for x, y in zip(lo, hi))
            support = row['support_nets']
            if set(support) != {'near','far'} or any(type(v) is not bool for v in support.values()):
                raise ValueError('CP10 actual net support evidence missing')
            bodies[name] = BallBounds(center, lo, hi, True, tuple(n for n,v in support.items() if v))
            if self.tick == -1 and any(abs(center[i] - self.config['objects'][name]['position'][i]) > .005
                                       for i in range(3)):
                raise ValueError('CP10 reset object layout differs from source')
        self.tick = tick
        self.final_nets = {name: contained_net(body,self.config) for name,body in bodies.items() if name.startswith('ball')}
        formal_phase = self.cycle is not None and self.cycle.plan['shot_index'] >= 2
        for name,row in frame['bodies'].items():
            if not name.startswith('ball') or int(name[4:]) < 2 or not formal_phase:
                continue
            if type(row.get('physical_robot_contact')) is not bool:
                raise ValueError('CP10 real robot contact evidence missing')
            if row['physical_robot_contact'] and self.failure is None:
                self.failure = dict(tick=tick,reason='robot_touched_ball',ball=name)
        if self.status != 'running':
            return  # Keep current final nets after sequence completion through the action boundary.
        if self.cycle is None:
            self.cycle = ShotCycle(self.config, self.colors)
        index = self.cycle.plan['shot_index']
        name = 'ball' + str(index)
        body, receipt, net = bodies[name], None, None
        if self.cycle.fired and self.cycle.phase != 'collected':
            if any(p.startswith('/World/Task/Objects/paddle') for p in frame['bodies'][name]['physical_contacts']):
                self.paddle_contact = True
            net = contained_net(body, self.config)
            self.net_ticks = (self.net_ticks + 1 if net is not None and net == self.net_name
                              else 1 if net is not None else 0)
            self.net_name = net
            if net is not None:
                receipt = NetReceipt(index, net, True, self.paddle_contact)
            if self.fired_tick is not None and tick - self.fired_tick > 2400:
                receipt = NetReceipt(index, None, True, self.paddle_contact)
        origin, direction = launch_pose(self.config['arena'], self.cycle.plan['yaw_deg'],
                                        self.cycle.plan.get('slide_x_m', 0.))
        clear = sum((body.center[j] - origin[j]) * direction[j] for j in range(2)) > .065
        prior_count = len(self.cycle.results)
        self.hardware = self.cycle.sample(tick, **frame['launcher'],
                                         ball_cleared_muzzle=clear, collection=receipt)
        if self.hardware['release_now']:
            self.expected_emitted.append(name)
            self.fired_tick = tick
        if len(self.cycle.results) > prior_count:
            self.shots.append(dict(shot_index=index, color=self.colors[index], net=net,
                paddle_contact=self.paddle_contact, released=True, launch_tick=self.fired_tick,
                collection_tick=tick, retention_ticks=self.net_ticks))
            self.paddle_contact, self.net_ticks, self.net_name, self.fired_tick = False, 0, None, None
        if self.hardware['phase'] == 'complete':
            correct = all(r['correct'] for r in self.cycle.results if r['score_eligible'])
            self._finish('completed' if correct else 'formal_shot_error')


    def snapshot(self, termination='policy_ended'):
        if self.tick < 0:
            raise ValueError('CP10 initial sample missing')
        # Every shot being collected is a physical episode endpoint even when
        # a formal shot was wrong; the scorer independently finds that error.
        reason = 'completed' if self.reason == 'formal_shot_error' else self.reason or termination
        return dict(contract=CONTRACT, source='measured_physical_state',
                    colors=list(self.colors), shots=deepcopy(self.shots),
                    terminal_reason=reason, failure=deepcopy(self.failure), last_tick=self.tick,
                    final_nets=deepcopy(self.final_nets), scoring_version='mind2act-final-scoring-20261105',
                    collection_protocol_version=PROTOCOL_ID,
                    collection_protocol_sha256=digest(RETENTION))
