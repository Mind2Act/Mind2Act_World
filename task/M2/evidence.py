"""Private, streamed physical measurements; replay uses the approved mechanism/scorer."""
import hashlib
import json
from math import isfinite
from pathlib import Path

from mind2act_eval.contracts import NativeEvidence, to_json
from .binding import validate
from .mechanism import MemoryMechanism, angle_error_deg
from .scorer import SCHEMA


def checked_frame(frame, count):
    if set(frame) != {'button_depths_m', 'finger_contacts', 'card_quaternions_wxyz', 'card_hinge_angles_deg'}:
        raise ValueError('M2 raw physical fields mismatch')
    if any(len(frame[key]) != count for key in frame):
        raise ValueError('M2 physical card/button evidence missing')
    if any(type(x) not in (int, float) or not isfinite(x) for x in frame['button_depths_m']):
        raise ValueError('M2 physical button depth invalid')
    if any(type(x) is not bool for x in frame['finger_contacts']):
        raise ValueError('M2 physical finger contact invalid')
    if any(type(x) not in (int, float) or not isfinite(x) for x in frame['card_hinge_angles_deg']):
        raise ValueError('M2 actual hinge angles unavailable')
    for q in frame['card_quaternions_wxyz']:
        angle_error_deg(q, q)
    return frame


def advance(game, tick, frame):
    if type(tick) is not int or tick != game.tick + 1:
        raise ValueError("M2 raw physical tick gap")
    checked_frame(frame, game.count)
    angles = frame['card_hinge_angles_deg']
    errors = [abs((x-2.+180.) % 360.-180.) for x in angles]
    game.step(tick, frame['button_depths_m'], frame['finger_contacts'],
              [value < 2. for value in errors],
              sum(abs(value) < 90. and i not in game.matched for i, value in enumerate(angles)))
    # Events are transient mechanism diagnostics, not a second unbounded trace.
    game.events.clear()


def snapshot(spec, binding, game, frame, tick):
    if type(tick) is not int or tick != game.tick:
        raise ValueError('M2 native evidence tick mismatch')
    checked_frame(frame, game.count)
    errors = [abs((x-2.+180.) % 360.-180.) for x in frame['card_hinge_angles_deg']]
    return NativeEvidence('M2', spec.instance_id, tick, dict(schema=SCHEMA,
        difficulty=spec.difficulty, card_count=game.count, pairs=game.pair_receipts(),
        card_angle_errors_deg=errors, angle_measurement='hinge', all_buttons_released=all(d < .0003 for d in frame['button_depths_m']),
        max_unmatched_faceup=game.max_unmatched_faceup, normal_termination=True,
        binding_mode=binding['mode']))


class RawTrace:
    def __init__(self, path, spec, binding, initial):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('xb')
        self.digest, self.size, self.tick = hashlib.sha256(), 0, 0
        self._write(dict(schema='m2.raw-physical.v1', identity=binding['identity'],
                         binding=binding, physics_hz=spec.clock.physics_hz, initial=initial))

    def _write(self, value):
        raw = (json.dumps(to_json(value), separators=(',', ':'), allow_nan=False)+'\n').encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.size += len(raw)

    def append(self, tick, frame):
        if type(tick) is not int or tick != self.tick + 1:
            raise ValueError('M2 raw physical tick gap')
        self._write(dict(physics_tick=tick, physical=frame))
        self.tick = tick

    def receipt(self):
        if not self.stream.closed:
            self.stream.flush()
        return dict(schema='m2.raw-physical.v1', path=str(self.path), sha256=self.digest.hexdigest(),
                    bytes=self.size, physics_rows=self.tick)

    def close(self):
        self.stream.close()


def replay_raw(spec, receipt, *, strict=False):
    """Recompute the terminal NativeEvidence from each raw measurement, in O(T)."""
    if receipt['schema'] != 'm2.raw-physical.v1':
        raise ValueError('M2 raw schema mismatch')
    digest, size = hashlib.sha256(), 0
    with Path(receipt['path']).open('rb') as stream:
        def read(raw):
            nonlocal size
            digest.update(raw)
            size += len(raw)
            return json.loads(raw)
        header = read(next(stream))
        if header['schema'] != receipt['schema'] or header['physics_hz'] != spec.clock.physics_hz:
            raise ValueError('M2 raw source clock/schema mismatch')
        binding = validate(spec, header['binding'])
        if header['identity'] != binding['identity']:
            raise ValueError('M2 raw identity mismatch')
        game = MemoryMechanism(binding['card_identities'])
        frame = checked_frame(header['initial'], game.count)
        for raw in stream:
            row = read(raw)
            frame = row['physical']
            advance(game, row['physics_tick'], frame)
    if game.tick != receipt['physics_rows']:
        raise ValueError('M2 raw row count mismatch')
    if strict and (digest.hexdigest(), size) != (receipt['sha256'], receipt['bytes']):
        raise ValueError('M2 raw SHA/size mismatch')
    return snapshot(spec, binding, game, frame, game.tick)
