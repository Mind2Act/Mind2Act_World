"""Per-user Kit initialization slots on the current machine."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

GATE_SCHEMA = 'mind2act_eval.kit_start_gate.v1'


def read_gate_state(raw):
    if not raw:
        return None
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('schema') != GATE_SCHEMA
            or value.get('state') not in {'active', 'cooling', 'idle'}):
        raise RuntimeError('invalid simulator startup gate state')
    return value['state']


@dataclass(frozen=True)
class StartupPaths:
    root: Path
    kit: Path


def startup_paths():
    root = local_startup_root()
    return StartupPaths(root, root / 'kit.lock')


def startup_configuration():
    """Explicit per-node initialization capacity; conservative defaults remain."""
    slots = int(os.environ.get('MIND2ACT_KIT_STARTUP_CONCURRENCY', '1'))
    interval = float(os.environ.get('MIND2ACT_KIT_START_INTERVAL_S', '60'))
    if slots < 1 or not math.isfinite(interval) or interval < 0:
        raise ValueError('Kit startup concurrency must be positive and interval finite/nonnegative')
    return slots, interval


def local_startup_root():
    """Use the same node-local directory even with per-episode TMPDIR values."""
    return (Path(os.sep) / 'tmp' / ('mind2act-startup-' + str(os.getuid()))).resolve()
