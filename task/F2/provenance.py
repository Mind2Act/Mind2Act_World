"""Bind original task conditions and explicit current/source input versions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = "f2-spec-20261030-v1"
SOURCE_VERSION = "f2-four-case-native-v2"
SOURCE_ID = "f2-35050e5-7f67c3e-uniform"
BUDGET = {"easy":5,"medium":6,"hard":7}
COUNTS = {"easy":3,"medium":4,"hard":5}


def canonical(c):
    return hashlib.sha256(json.dumps({k:v for k,v in c.items() if k != "case_sha256"},
        sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def validate_source_config(c):
    if c["task"] != "F2" or c["version"] != VERSION:
        raise ValueError("F2 effective rule/config binding mismatch")
    n = COUNTS[c["difficulty"]]
    if len(c["objects"]) != n or len(c["regions"]) != n or set(c["target"]) != set(c["objects"]) or len(c["target"]) != n:
        raise ValueError("F2 tier/layout binding mismatch")
    if c["physics_hz"] != 120 or c["stable_ticks"] != 60 or c["button"]["require_confirm_release"] is not True:
        raise ValueError("F2 requires actual released 0.5-second placements and confirm release")
    version = c["input_protocol_version"]
    if version not in (VERSION,SOURCE_VERSION):
        raise ValueError("F2 input protocol must be explicit")
    if c["attempt_budget"] != BUDGET[c["difficulty"]]:
        raise ValueError("F2 valid submission budget must be Easy5/Medium6/Hard7")


def effective_config(source, *, input_version):
    if source["version"] != SOURCE_VERSION or source["task"] != "F2":
        raise ValueError("F2 original source case checksum/version mismatch")
    c = deepcopy(source)
    c.update(version=VERSION, input_protocol_version=input_version,
             source_case_sha256=source["case_sha256"], source_attempt_budget=source["attempt_budget"])
    c["attempt_budget"] = BUDGET[source["difficulty"]]
    c["button"]["require_confirm_release"] = True
    c["case_sha256"] = canonical(c)
    validate_source_config(c)
    return c


def read_origin(receipt, difficulty, *, source_bytes=None):
    if receipt is None:
        return None
    raw = Path(receipt['path']).read_bytes() if source_bytes is None else source_bytes
    rows = json.loads(raw)['cells']
    matches = [row for row in rows if row['public_task']=='F2'
               and row['instance_id']==receipt['instance_id'] and row['cell']=='F2_'+difficulty]
    if len(matches) != 1:
        raise ValueError('F2 requires exactly the selected source index instance')
    return matches[0]


def validate_start(start, origin):
    if type(start) is not int or start < 0:
        raise ValueError('F2 source clock offset must be a nonnegative integer')
    if origin is None:
        if start != 0:
            raise ValueError('F2 nonzero source offset requires the exact projection receipt')
        return
    first = origin['timeline'][0]
    clock = origin['source_clock']
    stride = clock['physics_hz']//clock['control_hz']
    if (clock['physics_hz'] != 120 or clock['rgb_hz'] != 30
            or first['state_physics_step'] != start or first['action_physics_step'] != start+stride
            or first['state_control_index'] != 0 or first['action_control_index'] != stride):
        raise ValueError('F2 source offset/cadence contradicts the recorded projection')
    if origin['rows'] <= 0:
        raise ValueError('F2 incomplete source projection identity')
