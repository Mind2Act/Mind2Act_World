"""Reconstruct stamp marks from measured poses and per-tick PhysX contact impulses.

Source algorithms: PR5/rules/imprint.py and rules/scoring.py at b6432b4.
No controller event, phase, milestone, reference trajectory or commanded pose is read.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mind2act_eval.contracts import NativeEvidence
from .mechanism.imprint import Frame
from .mechanism.streaming import StreamingImprints
from .mechanism.grading import score as judge, stamp_home

STAMP = '/World/Task/StampA'


def under(path, root):
    return path == root or path.startswith(root + '/')


def vector(value, size):
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError('nonfinite or malformed PR5 physical measurement')
    return v


def rotation(q):
    w, x, y, z = vector(q, 4)
    if abs(w*w + x*x + y*y + z*z - 1.) > 2e-5:
        raise ValueError('PR5 stamp orientation must be a unit quaternion')
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


class Recorder:
    def __init__(self, task_id, instance_id, source_case, support_z=.768, *, trace_path=None):
        self.task_id, self.instance_id = task_id, instance_id
        self.case = deepcopy(source_case)
        self.support_z = float(support_z)
        self.tracker = StreamingImprints()
        self.frame_count = 0
        self.final_frame = None
        self.trace_path = Path(trace_path).resolve() if trace_path is not None else None
        self._stream = None
        self._trace_sha256 = hashlib.sha256()
        self._trace_bytes = 0
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.trace_path.open('xb')

    def record(self, tick, position, orientation_wxyz, tcps, contacts):
        if type(tick) is not int or tick != self.frame_count + 1:
            raise ValueError('PR5 requires each consecutive post-physics tick')
        p, q = vector(position, 3), vector(orientation_wxyz, 4)
        r = rotation(q)
        if set(tcps) != {'Left', 'Right'}:
            raise ValueError('PR5 requires measured TCP poses for both arms')
        tcps = {side: vector(value, 3).tolist() for side, value in tcps.items()}
        impulse, seen, rows = np.zeros(3), set(), []
        for event in contacts:
            pair = tuple(event['pair'])
            if (len(pair) != 2 or pair != tuple(sorted(pair)) or pair in seen
                    or not all(isinstance(path, str) and path.startswith('/World/') for path in pair)):
                raise ValueError('invalid or duplicate PR5 collider pair')
            seen.add(pair)
            value = vector(event['impulse_ns'], 3)
            row = dict(pair=list(pair), impulse_ns=value.tolist())
            if 'support_normal_z' in event:
                normal = float(event['support_normal_z'])
                if not math.isfinite(normal):
                    raise ValueError('nonfinite PR5 support normal')
                row['support_normal_z'] = normal
            rows.append(row)
            if any(under(path, STAMP + '/Face') for path in pair) and any(under(path, '/World/Task/CardA') for path in pair):
                impulse += value
        stamp = self.case['stamps'][0]
        emitted = self.tracker.update(Frame(tick/120, abs(float(impulse[2]))*120,
            float(p[0]-stamp['card_xy'][0]), float(p[1]-stamp['card_xy'][1]),
            math.degrees(math.atan2(r[1, 0], r[0, 0])),
            math.degrees(math.acos(float(np.clip(r[2, 2], -1., 1.))))))
        self.final_frame = dict(tick=tick, position=p.tolist(), orientation_wxyz=q.tolist(), tcps=tcps, contacts=rows)
        self.frame_count += 1
        if self._stream is not None:
            payload = (json.dumps(self.final_frame, separators=(',', ':'), allow_nan=False) + '\n').encode()
            self._stream.write(payload)
            self._trace_sha256.update(payload)
            self._trace_bytes += len(payload)
        return emitted

    def facts(self):
        if not self.frame_count:
            raise ValueError('PR5 evidence contains no post-physics sample')
        count, marks = self.tracker.preview()
        final, stamp = self.final_frame, self.case['stamps'][0]
        position = np.asarray(final['position'])
        # The 2026-11-04 ruling measures return only at the terminal sample.
        # A singleton preserves the pinned source pose thresholds without a dwell gate.
        home, details = stamp_home(position, rotation(final['orientation_wxyz']), [position],
                                   stamp['holder_xy'], self.support_z + .004)
        details.pop('still')
        holder_contacts = [row for row in final['contacts']
            if any(under(path, STAMP) for path in row['pair'])
            and any(under(path, '/World/Task/HolderA') for path in row['pair'])]
        if any('support_normal_z' not in row for row in holder_contacts):
            raise ValueError('missing owned PR5 holder support normal')
        supported = any(row['support_normal_z'] > 0. for row in holder_contacts)
        home = home and supported
        details['supported_by_holder'] = supported
        graded = judge(marks, dict(stamp, box_offset=[.020, 0.]), .050, home, True)
        graded.pop('arm_clear')
        return dict(graded, imprint_count=count, imprints=[vars(m) for m in marks], final=details)

    def evidence(self, tick):
        if tick != self.frame_count:
            raise ValueError('PR5 evidence tick mismatch')
        if self._stream is None:
            raise ValueError('PR5 export requires an environment-owned raw audit stream')
        self._stream.flush()
        return NativeEvidence(self.task_id, self.instance_id, tick,
            dict(source_case=self.case, support_z_m=self.support_z,
                 raw_trace=dict(path=str(self.trace_path), sha256=self._trace_sha256.hexdigest(),
                                bytes=self._trace_bytes, frames=self.frame_count)))

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def reconstruct(evidence):
    if not isinstance(evidence, NativeEvidence) or evidence.task_id != 'PR5' or type(evidence.tick) is not int or evidence.tick < 1:
        raise ValueError('missing PR5 native evidence')
    values = evidence.values
    if set(values) != {'source_case', 'support_z_m', 'raw_trace'}:
        raise ValueError('PR5 evidence schema/coverage mismatch')
    from mind2act_eval.contracts import to_json
    from .instances import validate_case
    validate_case(values['source_case'], values['source_case']['level'])
    if values['support_z_m'] != .768:
        raise ValueError('PR5 evidence support height differs from selected source')
    trace = values['raw_trace']
    if (set(trace) != {'path', 'sha256', 'bytes', 'frames'} or type(trace['frames']) is not int
            or trace['frames'] != evidence.tick or type(trace['bytes']) is not int or trace['bytes'] <= 0
            or not isinstance(trace['sha256'], str) or len(trace['sha256']) != 64):
        raise ValueError('PR5 raw trace receipt mismatch')
    path = Path(trace['path'])
    if not path.is_absolute() or path.stat().st_size != trace['bytes']:
        raise ValueError('PR5 raw trace length mismatch')
    rec = Recorder(evidence.task_id, evidence.instance_id, to_json(values['source_case']), values['support_z_m'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            frame = json.loads(line)
            if set(frame) != {'tick', 'position', 'orientation_wxyz', 'tcps', 'contacts'}:
                raise ValueError('PR5 scorer accepts only native measurements')
            rec.record(**frame)
    if rec.frame_count != evidence.tick or digest.hexdigest() != trace['sha256']:
        raise ValueError('PR5 raw trace hash/coverage mismatch')
    return rec.facts()
