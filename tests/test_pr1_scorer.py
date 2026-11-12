import math

import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks import PR1, TaskRegistry
from mind2act_eval.tasks._precision.stability import window_ticks
from mind2act_eval.tasks.PR1.scorer import score


def evidence(values, hz=120, target=10., consecutive=True, touching=None):
    touching = tuple(touching if touching is not None else [False] * len(values))
    return NativeEvidence('PR1', 'i', len(values), dict(difficulty='easy', tick_hz=hz, target_value=target, first_tick=1,
                                                        values=tuple(values), touching=touching, consecutive=consecutive))


def test_task_definition_registers_without_pxr_or_isaac():
    assert TaskRegistry([PR1.DEFINITION]).task_ids == ('PR1',)


def test_released_half_second_hold_scores_100():
    assert score(evidence([10.1] * 61)).value == 100.      # 60 intervals / 120 Hz = 0.5 s
    assert score(evidence([20.] * 5 + [10.1] * 61)).value == 100.
    assert window_ticks(120, .5) == 61


def test_held_device_is_allowed_at_t0():
    # in tolerance for 100 ticks while touched, released only for the last 30 ticks
    assert score(evidence([10.] * 130, touching=[True] * 100 + [False] * 30)).value == 100.
    assert score(evidence([10.] * 161, touching=[True] * 100 + [False] * 61)).value == 100.


def test_regrasp_history_does_not_change_t0_score():
    touching = [False] * 40 + [True] * 5 + [False] * 30
    assert score(evidence([10.] * 75, touching=touching)).value == 100.


def test_short_hold_off_target_or_drifting_away_scores_0():
    assert score(evidence([10.1] * 60)).value == 100.        # 0.4917 s
    assert score(evidence([20.] * 100)).value == 0.
    assert score(evidence([10.] * 200 + [10.5])).value == pytest.approx(28.125)   # window must run to the last sample
    assert score(evidence([10.2] * 61)).value == 100.      # boundary is inclusive
    assert score(evidence([10.21] * 61)).value == pytest.approx(29.2125)


def test_final_reading_is_independent_of_sample_rate():
    assert score(evidence([10.] * 31, hz=60)).value == 100.   # 30 intervals at 60 Hz
    assert score(evidence([10.] * 31, hz=120)).value == 100.


@pytest.mark.parametrize('bad', [
    evidence([]),
    NativeEvidence("PR1", "i", 62, dict(evidence([10.] * 61).values))])
def test_invalid_evidence_raises_instead_of_scoring(bad):
    with pytest.raises(ValueError):
        score(bad)


def test_scorer_rejects_other_task_and_extra_fields():
    other = NativeEvidence('PR2', 'i', 1, dict(tick_hz=120, target_value=1., first_tick=1, values=(1.,),
                                               touching=(False,), consecutive=True))
    with pytest.raises(ValueError):
        score(other)
    extra = NativeEvidence('PR1', 'i', 1, dict(tick_hz=120, target_value=1., first_tick=1, values=(1.,),
                                               touching=(False,), consecutive=True, expert_phase='release'))
    with pytest.raises(ValueError):
        score(extra)


def test_nonfinite_measurement_cannot_become_evidence():
    with pytest.raises(ValueError):
        evidence([math.nan] * 61)


def test_final_reading_does_not_require_release_or_a_hold_window():
    assert score(evidence([10.], touching=[True])).value == 100.
    assert score(evidence([10., 12.])).value == pytest.approx(22.5)
