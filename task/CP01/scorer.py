"""Core-only partial credit with reviewed CP difficulty caps; setup gates full success."""
import json
from importlib import resources

from mind2act_eval.scoring import SCORING_VERSION, cp_score_from_completion
from mind2act_eval.contracts import NativeEvidence, Score

_RULE = json.loads(resources.files(__package__).joinpath("config/scoring.json").read_text())


def _count(values, key):
    n = values[key]
    if type(n) is not int or n < 0:
        raise ValueError(f"CP01 evidence {key} must be a non-negative integer")
    return n


def score(evidence: NativeEvidence) -> Score:
    v = evidence.values
    if v.get("scoring_version") != SCORING_VERSION:
        raise ValueError("CP01 terminal placement evidence version required; old aggregate cannot be rescored")
    n = _RULE["required_steps"][v["difficulty"]]
    if _count(v, "required_steps") != n:
        raise ValueError("CP01 required-step count differs from the difficulty")
    satisfied, wrong = _count(v, "steps_satisfied"), _count(v, "wrong_or_extra_items")
    if satisfied > n:
        raise ValueError("CP01 satisfied steps exceed the requirement")
    success = bool(v["task_success"])
    if success and (satisfied != n or wrong):
        raise ValueError("CP01 success requires every step and no wrong or extra item")
    core_n = _RULE['core_units'][v['difficulty']]
    core = _count(v, 'credited_foods')
    if core > core_n:
        raise ValueError('CP01 core count exceeds the requirement')
    core_wrong = wrong
    k = max(0, core - core_wrong)
    return cp_score_from_completion(success, k, core_n, difficulty=v['difficulty'], components={
        "core_correct": core, "core_wrong_or_extra_items": core_wrong, "steps_satisfied": satisfied, "wrong_or_extra_items": wrong, "rule_version": SCORING_VERSION, "scoring_version": SCORING_VERSION})
