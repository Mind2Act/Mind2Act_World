"""E2: exact mechanically measured input, with the last key returned."""
from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION
from .instances import RUBRIC
from .mechanism.clock import input_start
from .mechanism.keys import Keys
from .evidence import read_measurements


def score(evidence: NativeEvidence) -> Score:
    if evidence.task_id != "E2":
        raise ValueError("E2 scorer received another task")
    values = evidence.values
    target = tuple(values["target_digits"])
    required = {"easy": (2,), "medium": (3,), "hard": (3, 4)}[values["difficulty"]]
    if len(target) not in required or any(type(d) is not int or not 1 <= d <= 9 for d in target):
        raise ValueError("E2 target does not match difficulty")
    if dict(values["timing"]) != {"learn_seconds": 2., "shuffle_seconds": .5, "reasoning_seconds": .25}:
        raise ValueError("E2 evidence changed the training observation timeline")
    gate = input_start(values["timing"])
    keys, completion_tick = Keys(), None
    for row in read_measurements(values["measurements"]):
        keys.sample(row["tick"], row["travel_m"], row["fingertip_contacts"], enabled=row["tick"] >= gate)
        if completion_tick is None and keys.complete(target):
            completion_tick = row["tick"]
    if keys.last_tick != evidence.tick:
        raise ValueError("E2 measurements do not cover the evidence clock")
    success = bool(keys.complete(target))
    cap = {'easy':30.,'medium':22.,'hard':15.}[values['difficulty']]
    correct = sum(a == b for a,b in zip(keys.digits[:len(target)], target))
    return Score(100.0 if success else cap * correct / len(target),
                 {"scoring_version": SCORING_VERSION, "task_success": success, "rubric": RUBRIC,
                  "entered_digits": keys.digits, "last_key_released": keys.released,
                  "serial_input_valid": not keys.serial_violation,
                  "credited_units": correct, "total_units": len(target), "partial_ceiling": cap, "required_presses": len(target), "completion_tick": completion_tick})
