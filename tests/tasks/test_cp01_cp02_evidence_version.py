"""Old aggregate measurements cannot become final-scoring evidence by relabeling."""
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.tasks.CP01.scorer import score as cp01_score
from mind2act_eval.tasks.CP02.scorer import score as cp02_score


@pytest.mark.parametrize("task,scorer,n,flags", [
    ("CP01", cp01_score, 3, {"task_success": True}),
    ("CP02", cp02_score, 4, {"physical_task_success": True, "task_process_pass": True}),
])
@pytest.mark.parametrize("version", [None, "legacy-scoring"])
def test_legacy_aggregate_is_rejected_before_new_score_is_emitted(task, scorer, n, flags, version):
    values = dict(difficulty="easy", required_steps=n, steps_satisfied=n,
                  wrong_or_extra_items=0, **flags)
    if version is not None:
        values["scoring_version"] = version
    with pytest.raises(ValueError, match="evidence version required"):
        scorer(NativeEvidence(task, "legacy-aggregate-only", 100, values))


@pytest.mark.parametrize("task,scorer,n,flags", [
    ("CP01", cp01_score, 3, {"task_success": True}),
    ("CP02", cp02_score, 4, {"physical_task_success": True, "task_process_pass": True}),
])
def test_rebuilt_current_version_evidence_reaches_the_scorer(task, scorer, n, flags):
    values = dict(difficulty="easy", required_steps=n, steps_satisfied=n,
                  wrong_or_extra_items=0, scoring_version=SCORING_VERSION, **flags)
    if task == "CP01":
        values["credited_foods"] = n
    else:
        values.update(credited_macarons=n, core_wrong_or_extra_items=0)
    result = scorer(NativeEvidence(task, "current-reconstruction", 100, values))
    assert result.value == 100
    assert result.components["scoring_version"] == SCORING_VERSION
