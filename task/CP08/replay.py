"""Normalize raw poses and owned PhysX contacts for chronological CP08 grading."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .tcp import measured_tcp


def vector(value, length):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError('invalid CP08 measured vector')
    return result


class Replay:
    def __init__(self, config, initial_frame=None, *, source_state_tick=72):
        self.config = config
        self.source_state_tick = source_state_tick
        self.initial_frame = initial_frame
        self.specs = {b['id']: b for b in config['blocks'] + config['distractors']}
        self.rows, self.contacts = [], {}

    def sample(self, raw, measured_q=None):
        tick = raw['tick']
        if type(tick) is not int or tick != len(self.rows) + 1:
            raise ValueError('missing CP08 physics tick')
        if set(raw['blocks']) != set(self.specs):
            raise ValueError('missing CP08 block evidence')
        from .physical import CONTACT_SCHEMA, owned_support
        complete_support = raw.get('contact_measurement_schema') == CONTACT_SCHEMA
        pairs, seen, support_rows = [], set(), []
        contact_impulses = {}
        for event in raw['contacts']:
            pair = event['pair']
            if len(pair) != 2 or any(not isinstance(p, str) or not p.startswith('/World/') for p in pair):
                raise ValueError('invalid CP08 collider pair')
            pair = tuple(sorted(pair))
            if pair in seen:
                raise ValueError('duplicate CP08 collider pair')
            seen.add(pair)
            impulse = vector(event['impulse_ns'], 3)
            self.contacts[pair] = max(self.contacts.get(pair, 0.), float(np.linalg.norm(impulse)))
            contact_impulses[pair] = float(np.linalg.norm(impulse))
            if 'contact_count' not in event or 'owned_support_points' not in event:
                complete_support = False
            elif type(event['contact_count']) is not int or event['contact_count'] < 0 or not isinstance(event['owned_support_points'],list):
                raise ValueError('invalid CP08 physical contact manifold')
            else:
                if len(event['owned_support_points']) != event['contact_count']:
                    raise ValueError('CP08 incomplete physical contact points')
                geometry=[dict(normal=p['normal_on_pair0'],separation_m=p['separation_m']) for p in event['owned_support_points']]
                owned=owned_support(pair,geometry)
                relevant=any('/World/CP11/Blocks/' in p for p in pair) and all('/World/CP11/Blocks/' in p or '/Worktable' in p for p in pair)
                if relevant and not owned:
                    raise ValueError('CP08 active support contact lacks owned normals')
                if event['contact_count'] > 0:
                    for point in owned:
                        point['physical'] = bool(np.linalg.norm(impulse)>0 or point['separation_m']<=0)
                    support_rows.extend(owned)
                    pairs.append(pair)
            if not complete_support and pair not in pairs:
                pairs.append(pair)  # legacy diagnostics only; new detector rejects missing ownership


        def touching(a, b):
            # Exact source semantics: presence of both substrings in this step's
            # contact report, without introducing an impulse/support threshold.
            return any(any(a in path for path in pair) and any(b in path for path in pair) for pair in pairs)

        blocks = {}
        for name, spec in self.specs.items():
            body = raw['blocks'][name]
            position = vector(body['position'], 3)
            quat = vector(body['orientation_wxyz'], 4)
            if abs(np.linalg.norm(quat) - 1) > 1e-4:
                raise ValueError('CP08 quaternion must be normalized')
            rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            velocity = vector(body['linear_velocity_m_s'], 3)
            root = '/World/CP11/Blocks/' + name + '/'
            bilateral = {side: touching(root, f'/Robot{side}/Asset/link7/') and touching(root, f'/Robot{side}/Asset/link8/') for side in ('Left', 'Right')}
            blocks[name] = dict(position=position.tolist(), quaternion=quat.tolist(),
                tilt_deg=float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
                yaw_error_deg=0. if spec['shape'] == 'cylinder' else float(abs(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))),
                linear_speed=float(np.linalg.norm(velocity)), bilateral=any(bilateral.values()), bilateral_by_arm=bilateral,
                robot_contact=touching(root, '/Robot'),
                table_support=any(p['collider'].startswith(root) and '/Worktable' in p['other'] and p['upward_normal']>0 and p['physical'] for p in support_rows),
                supports=[other for other in self.specs if other != name and any(
                    p['collider'].startswith(root) and p['other'].startswith('/World/CP11/Blocks/'+other+'/') and p['upward_normal']>0 and p['physical'] for p in support_rows)])
        tcps = {side: vector(raw['tcps'][side], 3).tolist() for side in ('Left', 'Right')}
        if set(raw['flange_poses']) != {'Left','Right'}:
            raise ValueError('CP08 measured flange coverage missing')
        for side,frame in raw['flange_poses'].items():
            actual=measured_tcp(frame['position'],frame['orientation_wxyz'])
            if not np.allclose(tcps[side],actual,rtol=0,atol=1e-12):
                raise ValueError('CP08 TCP differs from measured source flange frame')
        joints = {side: vector(raw['joints'][side], 6).tolist() for side in ('Left', 'Right')}
        expected_cue = tick + self.source_state_tick <= round(self.config['cue_seconds'] * 120)
        if type(raw['cue_visible']) is not bool or raw['cue_visible'] != expected_cue:
            raise ValueError('CP08 screen exposure differs from projected source tick')
        if measured_q is None:
            fingers=raw.get('measured_finger_positions_m')
            if fingers is None:raise ValueError('CP08 physical scorer requires measured8DOF evidence; arm6DOF or commands cannot substitute')
            measured_q={side:list(raw['joints'][side])+list(fingers[side]) for side in ('Left','Right')}
        jaws={side:vector(measured_q[side],8)[6:].tolist() for side in ('Left','Right')}
        self.rows.append(dict(tick=tick, blocks=blocks,support_measurements_complete=complete_support,contact_impulses=contact_impulses, cue_visible=raw['cue_visible'], tcp_by_arm=tcps, joints_by_arm=joints,measured_jaws=jaws))
        return self.rows[-1]

    def result(self, events=None):
        # Controller events are diagnostic only and never consumed.
        from .physical import evaluate, CONTACT_SCHEMA
        if self.initial_frame is None or self.initial_frame.get('contact_measurement_schema') != CONTACT_SCHEMA:
            raise ValueError('CP08 reset lacks owned physical contact measurements')
        return evaluate(self.rows,self.config,self.initial_frame)
