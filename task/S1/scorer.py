"""S1: count final supported centers in the four original physical slots."""
import re
from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION

N_BLOCKS = 4
MIN_STABLE_S = 1.0
MAX_RANGE_M = .003
_FIELDS = {"early_placement", "rotation_program_finished", "blocks", "timeline", "difficulty"}


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError("S1 %s must be boolean" % name)
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        raise ValueError("S1 %s must be a non-negative number" % name)
    return float(value)


def _counts(row, finished):
    for key in ("inside", "support", "released", "original_entity_slot_match", "placed_after_program_finished"):
        _bool(row[key], key)
    stable = _number(row["stable_seconds"], "stable_seconds")
    spread = _number(row["position_range_m"], "position_range_m")
    return bool(row['inside'] and row['support'] and row['original_entity_slot_match'])



def score(evidence: NativeEvidence) -> Score:
    values = evidence.values
    extra = set(values) - _FIELDS
    if extra:
        raise ValueError("S1 unknown evidence field: %s" % sorted(extra))
    blocks = list(values["blocks"])
    ids = [row.get("object_id") for row in blocks]
    if len(blocks) != N_BLOCKS or not all(type(i) is str and i for i in ids) or len(set(ids)) != N_BLOCKS:
        raise ValueError("S1 requires four unique task block identities")
    early = _bool(values["early_placement"], "early_placement")
    finished = _bool(values["rotation_program_finished"], "rotation_program_finished")
    level = values.get('difficulty')
    if level is None:
        found = set(re.split(r'[_/\-]+', evidence.instance_id)) & {'easy','medium','hard'}
        if len(found) != 1:raise ValueError('S1 evidence requires difficulty')
        level = found.pop()
    cap = {'easy':30.,'medium':22.,'hard':15.}[level]
    credited = sum(_counts(row, finished) for row in blocks)
    success = credited == N_BLOCKS
    if success:
        value = 100
    else:
        value = {3: cap/2, 2: cap/4}.get(credited, 0)
    return Score(float(value), {"scoring_version": SCORING_VERSION, "credited_units": credited, "total_units": N_BLOCKS, "success": success,
                                "early_placement": early, "timeline": dict(values.get("timeline", {}))})
