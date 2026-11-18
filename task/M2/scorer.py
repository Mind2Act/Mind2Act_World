"""M2: actual mechanical selections and strict terminal hinge error below two degrees."""
from math import isfinite

from mind2act_eval.contracts import Score
from mind2act_eval.scoring import SCORING_VERSION

PAIRS = {"easy": 2, "medium": 4, "hard": 8}
SCHEMA = "m2.physical.v1"


def score(evidence):
    if evidence.task_id != "M2" or evidence.values["schema"] != SCHEMA:
        raise ValueError("M2 evidence identity/schema mismatch")
    v = evidence.values
    total = PAIRS[v["difficulty"]]
    if v["card_count"] != 2 * total:
        raise ValueError("M2 native card count mismatch")
    if v.get("angle_measurement") != "hinge":
        raise ValueError("M2 actual hinge-angle evidence unavailable")
    errors = v["card_angle_errors_deg"]
    if len(errors) != 2 * total or any(type(x) not in (int, float) or not isfinite(x) or x < 0 for x in errors):
        raise ValueError("M2 physical card poses unavailable")
    for key in ("all_buttons_released", "normal_termination"):
        if type(v[key]) is not bool:
            raise ValueError("M2 physical release/termination evidence missing")
    maximum = v["max_unmatched_faceup"]
    if type(maximum) is not int or not 0 <= maximum <= 2 * total:
        raise ValueError("M2 physical unmatched-card count invalid")
    used, completed = set(), 0
    for pair in v["pairs"]:
        cards = tuple(pair["cards"])
        if (len(cards) != 2 or len(set(cards)) != 2 or set(cards).intersection(used)
                or any(type(i) is not int or not 0 <= i < 2 * total for i in cards)):
            raise ValueError("M2 contradictory physical pair evidence")
        for key in ("valid_press_release", "same_picture", "physical_faceup"):
            if type(pair[key]) is not bool:
                raise ValueError("M2 pair evidence requires physical booleans")
        used.update(cards)
        completed += int(all(pair[key] for key in ("valid_press_release", "same_picture", "physical_faceup")))
    violation = maximum > 2
    success = (not violation and completed == total and max(errors) < 2
               )
    cap = {'easy':30.,'medium':22.,'hard':15.}[v['difficulty']]
    partial = 0. if violation else cap * completed / total
    return Score(100. if success else partial, {
        "partial_ceiling": cap, "scoring_version": SCORING_VERSION, "task_success": bool(success), "completed_pairs": completed, "required_pairs": total,
        "rule": "m2-spec-20261030-v1", "max_unmatched_faceup": maximum,
        "all_buttons_released": v["all_buttons_released"], "max_card_angle_error_deg": max(errors),
        "binding_mode": v["binding_mode"]})
