"""DR4 native tray/ball measurements and bounded episode-wide event facts."""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np

from mind2act_eval.contracts import NativeEvidence, to_json
from .mechanism.balance import displacement, local_positions, NUMERIC_TOLERANCE_M
from .mechanism.containment import containment

TRAY = '/World/PR7/Tray'
HANDLE = TRAY + '/HandleBeam'
GOAL = '/World/PR7/GoalPlatform'


def under(path, root):
    return path == root or path.startswith(root + '/')


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('DR4 missing/nonfinite measured vector')
    return result


def rotation(value):
    w, x, y, z = vector(value, 4)
    if abs(w*w+x*x+y*y+z*z-1.) > 2e-5:
        raise ValueError('DR4 nonunit measured quaternion')
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def validate_frame(frame, count, tick):
    if (set(frame) != {'tick', 'time_s', 'tray_position', 'tray_rotation_wxyz', 'tray_linear_velocity',
                      'tray_angular_velocity', 'balls', 'tcps', 'contacts'}
            or type(frame['tick']) is not int or frame['tick'] != tick):
        raise ValueError('DR4 native frame/tick mismatch')
    if type(frame['time_s']) not in (int, float) or not np.isfinite(frame['time_s']) or frame['time_s'] < 0:
        raise ValueError('DR4 missing measured simulation time')
    for key in ('tray_position', 'tray_linear_velocity', 'tray_angular_velocity'):
        vector(frame[key], 3)
    rotation(frame['tray_rotation_wxyz'])
    if set(frame['balls']) != {'Ball'+str(i) for i in range(count)}:
        raise ValueError('DR4 ball identities/count differ from frozen instance')
    for body in frame['balls'].values():
        if set(body) != {'position', 'linear_velocity'}:
            raise ValueError('DR4 missing native ball pose/velocity')
        vector(body['position'], 3)
        vector(body['linear_velocity'], 3)
    if set(frame['tcps']) != {'Left', 'Right'}:
        raise ValueError('DR4 requires both measured arm TCPs')
    for value in frame['tcps'].values():
        vector(value, 3)
    seen = set()
    for row in frame['contacts']:
        if set(row) not in ({'pair', 'impulse_ns'}, {'pair', 'impulse_ns', 'support_normal_z'}):
            raise ValueError('DR4 invalid native contact')
        pair = tuple(row['pair'])
        if (len(pair) != 2 or pair != tuple(sorted(pair)) or pair in seen
                or not all(isinstance(p, str) and p.startswith('/World/') for p in pair)):
            raise ValueError('DR4 invalid/duplicate collider pair')
        seen.add(pair)
        vector(row['impulse_ns'], 3)
        if 'support_normal_z' in row and (type(row['support_normal_z']) not in (int, float)
                or not np.isfinite(row['support_normal_z'])):
            raise ValueError('DR4 nonfinite owned support normal')


class Recorder:
    def __init__(self, task_id, instance_id, config, geometry, initial, *, trace_path=None):
        self.task_id, self.instance_id = task_id, instance_id
        self.config, self.geometry, self.initial = to_json(config), to_json(geometry), to_json(initial)
        self.count = config['ball_count']
        levels = {'easy': 30., 'medium': 25., 'hard': 20.}
        if (type(self.count) is not int or self.count < 1 or config.get('level') not in levels
                or config['radius_mm'] != levels[config['level']]
                or (config['level'] in ('easy', 'hard') and self.count != {'easy':2, 'hard':4}[config['level']])):
            raise ValueError('DR4 missing original ball count/radius')
        validate_frame(initial, self.count, 0)
        self.goal = vector(geometry['goal'], 3)
        self.polygon = np.asarray(geometry['polygon'], dtype=float)
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2 or len(self.polygon) < 3 or not np.isfinite(self.polygon).all():
            raise ValueError('DR4 missing original tray polygon')
        if (np.linalg.norm(np.roll(self.polygon,-1,axis=0)-self.polygon, axis=1) == 0).any():
            raise ValueError('DR4 degenerate tray polygon')
        if not all(type(geometry[k]) in (int, float) and np.isfinite(geometry[k]) for k in ('floor_z', 'rim_z')):
            raise ValueError('DR4 missing original tray heights')
        self.anchors = local_positions(initial['tray_position'], rotation(initial['tray_rotation_wxyz']),
            [initial['balls']['Ball'+str(i)]['position'] for i in range(self.count)])
        self.maximum, self.spilled, self.bad = np.zeros(self.count), set(), set()
        self.final_displacement = np.zeros(self.count)
        self.grasp_tick, self.frames = None, 0
        self.tray_fallen = False
        self.previous_time = initial['time_s']
        self.recent = deque(maxlen=120)
        self.trace_path = Path(trace_path).resolve() if trace_path is not None else None
        self.stream, self.digest, self.byte_count = None, hashlib.sha256(), 0
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.trace_path.open('xb')

    def record(self, frame):
        validate_frame(frame, self.count, self.frames+1)
        if abs(frame['time_s']-self.previous_time-1/120) > 1e-7:
            raise ValueError('DR4 measured physics clock gap')
        p, r = vector(frame['tray_position'], 3), rotation(frame['tray_rotation_wxyz'])
        self.tray_fallen = bool(p[2] < self.initial['tray_position'][2] - .40)
        v, omega = vector(frame['tray_linear_velocity'], 3), vector(frame['tray_angular_velocity'], 3)
        balls = np.asarray([frame['balls']['Ball'+str(i)]['position'] for i in range(self.count)])
        velocities = np.asarray([frame['balls']['Ball'+str(i)]['linear_velocity'] for i in range(self.count)])
        local = local_positions(p, r, balls)
        self.final_displacement = displacement(local, self.anchors)
        self.maximum = np.maximum(self.maximum, self.final_displacement)
        for i, point in enumerate(local):
            if not containment(point, self.polygon, .014, self.geometry['floor_z'], self.geometry['rim_z'])[1]:
                self.spilled.add(i)
        supported = False
        for contact in frame['contacts']:
            impulse, pair = vector(contact['impulse_ns'], 3), contact['pair']
            tray = any(under(path, TRAY) for path in pair)
            obstacle = any(path.startswith('/World/PR7/Column') for path in pair)
            payload = tray or any(path.startswith('/World/Robot') or
                                 path.startswith('/World/PR7/Ball') for path in pair)
            if obstacle and payload:
                self.bad.add(tuple(pair))
            if tray and any(under(path, GOAL) for path in pair):
                if 'support_normal_z' not in contact:
                    raise ValueError('DR4 missing owned platform support normal')
                supported |= contact['support_normal_z'] > 0.
        position = bool(np.linalg.norm(p[:2]-self.goal[:2]) <= .020 and abs(p[2]-self.goal[2]) < .003)
        self.recent.append(dict(time=frame['time_s'], position=position, supported=bool(supported)))
        self.frames += 1
        self.previous_time = frame['time_s']
        if self.stream is not None:
            raw = (json.dumps(frame, separators=(',', ':'), allow_nan=False)+'\n').encode()
            self.stream.write(raw)
            self.digest.update(raw)
            self.byte_count += len(raw)

    def facts(self):
        # R[DR4] explicitly publishes +0.5mm calibration for the painted circles.
        radius = self.config['radius_mm'] / 1000 + NUMERIC_TOLERANCE_M
        escaped = [int(i) for i in np.flatnonzero(self.maximum > np.nextafter(radius, np.inf))]
        eligible = not (escaped or self.bad or self.spilled)
        current_inside = bool(np.all(self.final_displacement <= np.nextafter(radius, np.inf)))
        delivered = bool(not self.tray_fallen and self.recent
                         and self.recent[-1]['position'] and self.recent[-1]['supported'])
        success = bool(eligible and current_inside and delivered)
        return dict(success=success, prerequisites=bool(eligible),
            all_balls_retained=not escaped, each_ball_within_initial_radius=not escaped,
            all_balls_currently_in_circles=current_inside,
            final_ball_displacement_m=self.final_displacement.tolist(),
            no_obstacle_contacts=not self.bad, current_delivered=delivered,
            no_balls_spilled=not self.spilled, spilled_balls=sorted(self.spilled),
            terminal_platform_support=bool(self.recent and self.recent[-1]['supported']),
            maximum_ball_displacement_m=self.maximum.tolist(), radius_violations_latched=escaped,
            obstacle_contact_pairs=[list(pair) for pair in sorted(self.bad)], instance_ball_count=self.count)

    def evidence(self, tick):
        if tick != self.frames or self.stream is None:
            raise ValueError('DR4 missing native audit stream/tick binding')
        self.stream.flush()
        return NativeEvidence(self.task_id, self.instance_id, tick, dict(config=self.config, geometry=self.geometry,
            initial=self.initial, raw_trace=dict(path=str(self.trace_path), sha256=self.digest.hexdigest(), bytes=self.byte_count, frames=self.frames)))

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def reconstruct(evidence, *, strict=False):
    if not isinstance(evidence, NativeEvidence) or evidence.task_id != 'DR4' or type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError('DR4 missing native evidence')
    values = to_json(evidence.values)
    if set(values) != {'config', 'geometry', 'initial', 'raw_trace'}:
        raise ValueError('DR4 evidence schema mismatch')
    trace = values['raw_trace']
    if (set(trace) != {'path', 'sha256', 'bytes', 'frames'} or type(trace['frames']) is not int
            or trace['frames'] != evidence.tick or type(trace['bytes']) is not int or trace['bytes'] < 0
            or not isinstance(trace['sha256'], str) or len(trace['sha256']) != 64):
        raise ValueError('DR4 raw trace receipt mismatch')
    path = Path(trace['path'])
    if not path.is_absolute() or path.stat().st_size != trace['bytes']:
        raise ValueError('DR4 raw trace length mismatch')
    rec = Recorder(evidence.task_id, evidence.instance_id, values['config'], values['geometry'], values['initial'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            rec.record(json.loads(line))
    if rec.frames != evidence.tick or (strict and digest.hexdigest() != trace['sha256']):
        raise ValueError('DR4 raw trace hash/coverage mismatch')
    return rec.facts()
