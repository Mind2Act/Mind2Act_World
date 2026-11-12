"""Append-only raw native evidence; bounded online receipt and one final replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from mind2act_eval.contracts import NativeEvidence, to_json
from .mechanism.dominos import Dominos
from .mechanism.evidence import vector

ROOT = '/World/Task/Objects/'


def under(path, root):
    return path == root or path.startswith(root+'/')


def canonical(path):
    return '/'.join(path.split('/')[:5]) if path.startswith(ROOT) else path


def rule_frame(raw, config, tick):
    if (set(raw) != {'tick','time_s','bodies','contacts'} or type(raw['tick']) is not int
            or raw['tick'] != tick or set(raw['bodies']) != set(config['objects'])):
        raise ValueError('PR6 requires every native physics tick and every frozen body')
    vector([raw['time_s']], 1, 'time_s')
    if raw['time_s'] < 0 or not isinstance(raw['contacts'], list):
        raise ValueError('PR6 invalid native clock/contact data')
    events, details, seen = [], {}, set()
    for row in raw['contacts']:
        if set(row) != {'pair','active','points'} or type(row['active']) is not bool:
            raise ValueError('PR6 invalid native contact schema')
        pair = tuple(row['pair'])
        if (len(pair) != 2 or pair != tuple(sorted(pair)) or pair in seen
                or any(not isinstance(p,str) or not p.startswith('/World/') for p in pair)):
            raise ValueError('PR6 invalid/duplicate native collider pair')
        seen.add(pair)
        if not isinstance(row['points'], list) or (not row['active'] and row['points']):
            raise ValueError('PR6 lost contact cannot have live support points')
        for point in row['points']:
            if set(point) != {'position','normal','separation'}:
                raise ValueError('PR6 missing raw contact manifold')
            vector(point['position'],3,'contact position')
            normal = vector(point['normal'],3,'contact normal')
            if not .99 <= np.linalg.norm(normal) <= 1.01:
                raise ValueError('PR6 nonunit contact normal')
            vector([point['separation']],1,'contact separation')
        details[pair] = row
        events.append(dict(pair=list(pair), active=row['active'], contact_count=len(row['points']),
            min_separation_m=min((p['separation'] for p in row['points']), default=1.)))
    bodies = {}
    for name, row in raw['bodies'].items():
        if set(row) != {'position','orientation_wxyz','linear','angular'}:
            raise ValueError('PR6 accepts measured poses/velocities only')
        p = np.asarray(vector(row['position'],3,'position'))
        q = np.asarray(vector(row['orientation_wxyz'],4,'orientation'))
        if abs(np.linalg.norm(q)-1) > .001:
            raise ValueError('PR6 nonunit measured quaternion')
        vector(row['linear'],3,'linear'); vector(row['angular'],3,'angular')
        half = abs(Rotation.from_quat(q[[1,2,3,0]]).as_matrix()) @ (np.asarray(config['objects'][name]['size'])/2)
        contacts, supports = set(), []
        for pair, contact in details.items():
            if not contact['active'] or not contact['points']:
                continue
            for own, other, sign in ((pair[0],pair[1],1),(pair[1],pair[0],-1)):
                if under(own, ROOT+name):
                    contacts.add(canonical(other))
                    for point in contact['points']:
                        normal = [sign*v for v in point['normal']]
                        if normal[2] >= .5:
                            supports.append(dict(point, path=canonical(other), normal=normal))
        bodies[name] = dict(row, lower=(p-half).tolist(), upper=(p+half).tolist(),
                            contacts=sorted(contacts), support_contacts=supports)
    return dict(tick=tick, bodies=bodies, contact_events=events)


class Recorder:
    def __init__(self, instance_id, config, *, trace_path=None):
        self.instance_id, self.config = instance_id, to_json(config)
        self.rules = Dominos(self.config)
        self.frames, self.bytes = 0, 0
        self.digest = hashlib.sha256()
        self.previous_time = None
        self.path = Path(trace_path).resolve() if trace_path is not None else None
        self.stream = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open('xb')

    def record(self, raw):
        frame = rule_frame(raw, self.config, self.frames)
        if self.previous_time is not None and abs(raw['time_s']-self.previous_time-1/120) > 1e-6:
            raise ValueError('PR6 native time must advance exactly one 120 Hz tick')
        if self.frames == 0:
            for name, body in frame['bodies'].items():
                if np.max(np.abs(np.asarray(body['position'])-self.config['objects'][name]['position'])) > .005:
                    raise ValueError('PR6 reset layout differs from selected source')
        self.rules.sample(frame)
        payload = json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
        if self.stream is not None:
            self.stream.write(payload)
        self.digest.update(payload)
        self.bytes += len(payload)
        self.frames += 1
        self.previous_time = raw['time_s']

    def evidence(self, tick):
        if self.stream is None or tick != self.frames-1:
            raise ValueError('PR6 missing raw audit/tick binding')
        self.stream.flush()
        return NativeEvidence('PR6',self.instance_id,tick,dict(config=self.config,raw_trace=dict(
            path=str(self.path),sha256=self.digest.hexdigest(),bytes=self.bytes,frames=self.frames)))

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def reconstruct(evidence):
    if not isinstance(evidence,NativeEvidence) or evidence.task_id != 'PR6' or type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError('PR6 missing native evidence')
    values = to_json(evidence.values)
    if set(values) != {'config','raw_trace'}:
        raise ValueError('PR6 evidence schema mismatch')
    from .instances import validate_case
    validate_case(values['config'],values['config']['difficulty'])
    receipt = values['raw_trace']
    if (set(receipt) != {'path','sha256','bytes','frames'} or type(receipt['frames']) is not int
            or receipt['frames'] != evidence.tick+1 or type(receipt['bytes']) is not int or receipt['bytes'] < 0):
        raise ValueError('PR6 native audit receipt mismatch')
    path = Path(receipt['path'])
    if not path.is_absolute() or path.stat().st_size != receipt['bytes']:
        raise ValueError('PR6 raw trace length mismatch')
    rec, digest = Recorder(evidence.instance_id,values['config']), hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            rec.record(json.loads(line))
    if rec.frames != receipt['frames'] or digest.hexdigest() != receipt['sha256']:
        raise ValueError('PR6 raw trace SHA/coverage mismatch')
    return rec.rules.facts()
