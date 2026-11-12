"""Static task/config binding; no legacy generator or source-loader imports."""
from copy import deepcopy
import hashlib
import json

VERSION = "s2-spec-20261030-v1"
SOURCE_VERSION = "s2-four-case-native-v4"
SOURCE_ID = "s2-fbe12eb-5e523fc"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(c):
    if c["task"] != "S2" or c["version"] != VERSION:
        raise ValueError("S2 rule identity mismatch")
    if c["case_sha256"] != digest({k: v for k, v in c.items() if k != "case_sha256"}):
        raise ValueError("S2 effective case checksum mismatch")
    protocol = c["input_protocol_version"]
    if protocol not in (VERSION, SOURCE_VERSION):
        raise ValueError("S2 input protocol must be explicitly versioned")
    cue = 240 if protocol == VERSION else c["source_cue_ticks"]
    if c["physics_hz"] != 120 or c["cue_ticks"] != cue or c["stable_ticks"] != 60:
        raise ValueError("S2 cue/stability differs from the bound input version")
    n, blocked = {"easy": (4, 3), "medium": (5, 7), "hard": (6, 12)}[c["difficulty"]]
    if c["board"]["size"] != n or len(c["board"]["blocked"]) != blocked or set(c["objects"]) != {"slider"}:
        raise ValueError("S2 tier/board binding mismatch")
    d = c["direction_control"]
    if (set(d["buttons"]) != {"up", "down", "left", "right"}
            or d["press_depth_m"] != .003 or d["release_depth_m"] != .001 or d["press_ticks"] != 3
            or d["one_cell_per_press"] is not True or d["require_release"] is not True):
        raise ValueError("S2 source button protocol changed")


def effective_config(source, *, input_version=VERSION):
    if source["version"] != SOURCE_VERSION or source["task"] != "S2":
        raise ValueError("S2 needs the selected native source config")
    if source["case_sha256"] != digest({k: v for k, v in source.items() if k != "case_sha256"}):
        raise ValueError("S2 original case checksum mismatch")
    c = deepcopy(source)
    c["source_case_sha256"] = source["case_sha256"]
    c["source_cue_ticks"] = source["cue_ticks"]
    c["version"] = VERSION
    c["input_protocol_version"] = input_version
    c["cue_ticks"] = source["cue_ticks"] if input_version == SOURCE_VERSION else 240
    c["board_presentation"]["initial_map_ticks"] = c["cue_ticks"]
    c["case_sha256"] = digest({k: v for k, v in c.items() if k != "case_sha256"})
    validate_config(c)
    return c


def cell_center(board, cell):
    return [board["origin"][i] + (cell[i] + .5) * board["cell_size"] for i in range(2)]
