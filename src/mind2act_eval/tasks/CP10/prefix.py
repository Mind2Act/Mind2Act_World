"""Complete real-tick physical journal, including measurements after sequence completion."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mind2act_eval.contracts import to_json
from .machine import ArenaMachine

TRACE_CONTRACT = 'cp10-native-prefix-v1'


def encode(value):
    return (json.dumps(to_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def snapshot_sha(value):
    return hashlib.sha256(encode(value)).hexdigest()


class PhysicalPrefix:
    def __init__(self, machine, output_dir):
        self.machine = machine
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'cp10-physical-prefix.jsonl'
        self.tail_path = self.directory / 'cp10-postterminal-replay.jsonl'
        self._stream = self.path.open('xb')
        self._tail = None
        self._hash = hashlib.sha256()
        self.byte_count = self.count = 0
        self.tick = -1
        self.frozen = None
        self.receipt = None
        self.tail_count = 0

    def observe(self, frame):
        if type(frame['tick']) is not int or frame['tick'] != self.tick + 1:
            raise ValueError('CP10 physical/tail journal requires consecutive real ticks')
        payload = encode(frame)
        self.machine.sample(frame)
        self._stream.write(payload); self._stream.flush()
        self._hash.update(payload); self.byte_count += len(payload); self.count += 1
        self.tick = frame['tick']

    def _receipt(self):
        return dict(path=str(self.path.resolve()), sha256=self._hash.hexdigest(),
                    bytes=self.byte_count, count=self.count, first_tick=0,
                    last_tick=self.machine.tick)

    def values(self):
        if self.tick < 0:
            raise ValueError('CP10 reset physical prefix missing')
        snapshot = self.machine.snapshot()
        terminal = self.machine.terminal_tick
        return dict(trace_contract=TRACE_CONTRACT, cp10=snapshot,
            source_config=deepcopy(self.machine.config),
            scored_until_tick=snapshot['last_tick'],
            native_terminal_tick=terminal, native_terminal_reason=self.machine.reason,
            frozen_native_snapshot=(dict(cp10=deepcopy(snapshot), sha256=snapshot_sha(snapshot))
                                    if self.frozen is not None else None),
            physical_prefix_receipt=deepcopy(self.receipt) if self.receipt is not None else self._receipt(),
            postterminal_replay=dict(tail_after_native_terminal=bool(self.tail_count),
                count=self.tail_count, first_tick=terminal + 1 if self.tail_count else None,
                last_tick=self.tick if self.tail_count else None))

    def close(self):
        for stream in (self._stream, self._tail):
            if stream is not None:
                stream.close()


def verify_prefix(values, outer_tick):
    """Read exactly the hashed scoring prefix; never consume the tail journal."""
    values = to_json(values)
    if values['trace_contract'] != TRACE_CONTRACT:
        raise ValueError('CP10 physical prefix contract differs')
    end = values['scored_until_tick']
    if type(end) is not int or not 0 <= end <= outer_tick:
        raise ValueError('CP10 scored cutoff is outside the real episode clock')
    receipt = values['physical_prefix_receipt']
    if (type(receipt['count']) is not int or receipt['count'] != end + 1
            or type(receipt['first_tick']) is not int or receipt['first_tick'] != 0
            or type(receipt['last_tick']) is not int or receipt['last_tick'] != end
            or type(receipt['bytes']) is not int or receipt['bytes'] <= 0):
        raise ValueError('CP10 physical prefix receipt clock/count differs')
    if values['native_terminal_tick'] is not None and type(values['native_terminal_tick']) is not int:
        raise ValueError('CP10 native terminal tick must be an integer')
    machine = ArenaMachine(values['source_config'], values['collection_protocol'], values['rule_version'])
    byte_count, count, hasher = 0, 0, hashlib.sha256()
    try:
        with Path(receipt['path']).open('rb') as stream:
            for line in stream:
                if count > end or not line.endswith(b'\n'):
                    raise ValueError('CP10 physical prefix has an appended or incomplete row')
                frame = json.loads(line)
                if frame['tick'] != count:
                    raise ValueError('CP10 physical prefix row clock differs')
                machine.sample(frame)
                count += 1
                byte_count += len(line)
                hasher.update(line)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('CP10 physical prefix unavailable or malformed') from error
    if (count != receipt['count'] or byte_count != receipt['bytes']
            or hasher.hexdigest() != receipt['sha256']):
        raise ValueError('CP10 physical prefix SHA/bytes/count differs')
    snapshot = values['cp10']
    if (snapshot != machine.snapshot(snapshot['terminal_reason']) or snapshot['last_tick'] != end
            or values['native_terminal_tick'] != machine.terminal_tick
            or values['native_terminal_reason'] != machine.reason):
        raise ValueError('CP10 scored snapshot disagrees with the physical prefix')
    if values['frozen_native_snapshot'] is not None or end != outer_tick:
        raise ValueError('CP10 full real trace must reach the action boundary')
    tail = values['postterminal_replay']
    tail_count = outer_tick - end
    if (type(tail['count']) is not int or tail['count'] != tail_count
            or tail['tail_after_native_terminal'] is not bool(tail_count)
            or any(v is not None and type(v) is not int for v in (tail['first_tick'], tail['last_tick']))
            or tail['first_tick'] != (end + 1 if tail_count else None)
            or tail['last_tick'] != (outer_tick if tail_count else None)):
        raise ValueError('CP10 postterminal coverage differs from the real clock')
    return snapshot
