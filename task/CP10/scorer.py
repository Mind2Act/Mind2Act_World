"""CP10 section 6: fixed white-ball denominator and first-wrong-shot prefix."""
from mind2act_eval.scoring import cp_score_from_completion
from mind2act_eval.contracts import NativeEvidence
from . import CONTRACT
from .evidence import LAYOUT, validate_schedule
from .protocol import digest, validate

NORMAL_REASONS = {"completed", "policy_ended", "give_up", "physics_budget_exhausted",
                  "ball_not_collected", "robot_touched_ball", "unreleased_ball_moved"}


def _score_receipts(evidence: NativeEvidence):
    if evidence.task_id != "CP10":
        raise ValueError("CP10 evidence task mismatch")
    try:
        values = evidence.values
        difficulty, e = values["difficulty"], values["cp10"]
        validate(values["collection_protocol"], values["rule_version"])
        if (e["collection_protocol_version"] != values["rule_version"]
                or e["collection_protocol_sha256"] != digest(values["collection_protocol"])):
            raise ValueError("CP10 evidence/scorer collection protocol mismatch")
        if e["contract"] != CONTRACT or e["source"] != "measured_physical_state":
            raise ValueError("CP10 physical evidence contract missing")
        colors = tuple(e["colors"])
        validate_schedule(difficulty, colors)
        if type(values["normal_termination"]) is not bool:
            raise ValueError("CP10 normal termination measurement missing")
        if (type(evidence.tick) is not int or evidence.tick < 0
                or e["last_tick"] != evidence.tick or type(e["last_tick"]) is not int):
            raise ValueError("CP10 physical trace does not reach the score tick")
        if e["terminal_reason"] not in NORMAL_REASONS:
            raise ValueError("CP10 infrastructure failure or missing terminal evidence")
        rows = tuple(e["shots"])
        if len(rows) > len(colors):
            raise ValueError("CP10 excess shot receipts")
        if e["terminal_reason"] == "completed" and len(rows) != len(colors):
            raise ValueError("CP10 completion without every required shot receipt")
        k, stopped, last_collection = 0, False, -1
        for index, row in enumerate(rows):
            if (type(row["shot_index"]) is not int or row["shot_index"] != index
                    or row["color"] != colors[index]
                    or row["net"] not in (None, "near", "far")
                    or type(row["paddle_contact"]) is not bool or row["released"] is not True
                    or type(row["retention_ticks"]) is not int or row["retention_ticks"] < 0
                    or type(row["launch_tick"]) is not int
                    or type(row["collection_tick"]) is not int
                    or not last_collection < row["launch_tick"] <= row["collection_tick"] <= evidence.tick
                    or row["retention_ticks"] > row["collection_tick"] - row["launch_tick"] + 1):
                raise ValueError("CP10 invalid or incomplete measured shot receipt")
            last_collection = row["collection_tick"]
            correct = ((row["color"] == "white" and row["net"] == "far" and row["paddle_contact"])
                       or (row["color"] == "red" and row["net"] == "near" and not row["paddle_contact"]))
            if index < 2:  # Neither practice outcome contributes a step/error.
                continue
            correct = correct and e['final_nets'].get('ball'+str(index)) == row['net']
            if not correct:
                stopped = True
            if not stopped and row["color"] == "white":
                k += 1
        failure = e["failure"]
        if failure is not None and (failure['reason'] != 'robot_touched_ball' or
                type(failure['tick']) is not int or not 0<=failure['tick']<=evidence.tick):
            raise ValueError('CP10 invalid physical robot contact evidence')
        if failure is not None:
            k=0; stopped=True
        success = bool(len(rows) == len(colors) and not stopped and failure is None
                       and e["terminal_reason"] == "completed")
        n = LAYOUT[difficulty][1]
        return cp_score_from_completion(success, k, n, difficulty=difficulty,
            components={"scoring_version": "mind2act-final-scoring-20261105",
                        "first_error_free_white_prefix": k,
                        "physical_evidence_contract": CONTRACT})
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("CP10 incomplete physical evidence") from exc


def score(evidence: NativeEvidence):
    from .scoring_envelope import scoring_evidence
    try:
        return _score_receipts(scoring_evidence(evidence))
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("CP10 incomplete physical prefix envelope") from exc
