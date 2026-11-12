"""F2 native manifolds and bounded, hash-sealed physical measurement streams."""
import hashlib
import json
import math
from pathlib import Path

from .geometry import vector, Body

CAP = '/World/Task/Confirm/Asset/E_button_01_13'
BUTTON_ROOT = '/World/Task/Confirm/Asset'
OBJECT_ROOT = '/World/Task/Objects/'


class Rows:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('xb')
        self.digest = hashlib.sha256()
        self.count, self.size = 0, 0

    def append(self, row):
        if row['physics_tick'] != self.count + 1:
            raise ValueError('F2 native measurement gap')
        raw = (json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.count += 1
        self.size += len(raw)

    def receipt(self):
        self.stream.flush()
        return dict(schema='f2.native-measurements.v1', path=str(self.path), sha256=self.digest.hexdigest(),
                    rows=self.count, bytes=self.size)

    def close(self):
        self.stream.close()


def read_rows(receipt):
    if receipt['schema'] != 'f2.native-measurements.v1':
        raise ValueError('F2 native measurement schema mismatch')
    digest, size, count = hashlib.sha256(), 0, 0
    with Path(receipt['path']).open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            size += len(raw)
            count += 1
            yield json.loads(raw)
    if (digest.hexdigest(), size, count) != (receipt['sha256'], receipt['bytes'], receipt['rows']):
        raise ValueError('F2 native measurement SHA/size/count mismatch')


class Manifolds:
    def __init__(self):
        self.events, self.pairs, self.details = [], {}, {}
        self.subscription = None

    def subscribe(self):
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType

        def report(headers, data):
            for h in headers:
                ordered = tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
                pair = tuple(sorted(ordered))
                if not any('/World/Task/' in p for p in pair):
                    continue
                active = h.type != ContactEventType.CONTACT_LOST
                rows = data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
                sign = 1 if pair[0] == ordered[0] else -1
                points = [dict(position=list(map(float, p.position)), normal=[sign*float(v) for v in p.normal],
                               separation=float(p.separation)) for p in rows]
                self.pairs[pair] = active
                if active and points:
                    self.details[pair] = points
                elif not active:
                    self.details.pop(pair, None)
                self.events.append(dict(pair=list(pair), active=active, contact_count=len(points),
                    min_separation_m=min((p['separation'] for p in points), default=1.), points=points))
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(report)

    def contacts(self, path):
        found = set()
        for pair, active in self.pairs.items():
            if active:
                for own, other in (pair, pair[::-1]):
                    if own == path or own.startswith(path+'/'):
                        found.add('/'.join(other.split('/')[:5]) if other.startswith(OBJECT_ROOT) else other)
        return sorted(found)

    def supports(self, path):
        result = []
        for pair, points in self.details.items():
            if not self.pairs.get(pair):
                continue
            for own, other, sign in ((pair[0], pair[1], 1), (pair[1], pair[0], -1)):
                if own != path and not own.startswith(path+'/'):
                    continue
                other = '/'.join(other.split('/')[:5]) if other.startswith(OBJECT_ROOT) else other
                for point in points:
                    normal = [sign*v for v in point['normal']]
                    if normal[2] >= .5:
                        result.append(dict(point, path=other, normal=normal))
        return result

    def close(self):
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None


def cap_contact(events):
    return any(e['active'] and e['contact_count'] > 0 and e['min_separation_m'] <= .0001
               and any(own.startswith(CAP+'/') and other.startswith('/World/Robot')
                       for own, other in (e['pair'], e['pair'][::-1])) for e in events)


def measure(tick, source_start, c, bodies, button, cap, contacts, world):
    import numpy as np
    from scipy.spatial.transform import Rotation
    raw = dict(tick=source_start+tick, bodies={}, contact_events=list(contacts.events))
    for name, body in bodies.items():
        p, q = body.get_world_pose()
        rotation = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
        half = np.abs(rotation) @ (np.asarray(c['objects'][name]['size']) / 2)
        raw['bodies'][name] = dict(lower=(p-half).tolist(), upper=(p+half).tolist(),
            position=p.tolist(), orientation_wxyz=q.tolist(), linear=body.get_linear_velocity().tolist(),
            angular=body.get_angular_velocity().tolist(), contacts=contacts.contacts(OBJECT_ROOT+name),
            support_contacts=contacts.supports(OBJECT_ROOT+name))
    q = button.get_joint_positions().tolist()
    p, orientation = cap.get_world_pose()
    raw['button'] = dict(depth=float(c['button']['rest_joint_position_m']-q[0]),
        pose_depth_m=float(c['button']['position'][2]-p[2]), position=p.tolist(),
        orientation_wxyz=orientation.tolist(), joint_position_m=q,
        joint_velocity_m_s=button.get_joint_velocities().tolist(), robot_contact=cap_contact(contacts.events))
    return dict(physics_tick=tick, source_tick=source_start+tick,
                simulation_time_s=float(world.current_time), f2_raw=raw)


def validate_frame(c, row, *, native_tick, source_start):
    raw = row['f2_raw']
    if (type(row['physics_tick']) is not int or row['physics_tick'] != native_tick
            or raw['tick'] != source_start+native_tick or row['source_tick'] != raw['tick']):
        raise ValueError('F2 native/source clock mismatch')
    if not math.isfinite(row['simulation_time_s']) or set(raw['bodies']) != set(c['objects']):
        raise ValueError('F2 missing world time or body measurements')
    for body in raw['bodies'].values():
        Body.parse(body)
        vector(body['position'], 3, 'native body position')
        vector(body['orientation_wxyz'], 4, 'native body orientation')
    for event in raw['contact_events']:
        if (type(event['active']) is not bool or type(event['contact_count']) is not int
                or event['contact_count'] < 0 or event['contact_count'] != len(event['points'])
                or len(event['pair']) != 2 or any(type(path) is not str for path in event['pair'])):
            raise ValueError('F2 invalid native contact batch')
        if event['min_separation_m'] != min((point['separation'] for point in event['points']), default=1.):
            raise ValueError('F2 native separation differs from contact manifold')
        for point in event['points']:
            vector(point['position'], 3, 'contact point')
            vector(point['normal'], 3, 'contact normal')
            if not math.isfinite(point['separation']):
                raise ValueError('F2 non-finite contact separation')
    b = raw['button']
    vector([b['depth'], b['pose_depth_m']], 2, 'confirm depths')
    q = vector(b['joint_position_m'], 1, 'confirm joint')
    vector(b['joint_velocity_m_s'], 1, 'confirm joint velocity')
    vector(b['position'], 3, 'confirm cap position')
    vector(b['orientation_wxyz'], 4, 'confirm cap orientation')
    if abs(b['depth'] - (c['button']['rest_joint_position_m']-q[0])) > 1e-12:
        raise ValueError('F2 button depth differs from native joint measurement')
    if type(b['robot_contact']) is not bool or b['robot_contact'] != cap_contact(raw['contact_events']):
        raise ValueError('F2 button contact differs from current native manifold')
