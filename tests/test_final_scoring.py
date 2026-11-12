"""Public final-scoring arithmetic and physical-boundary contracts."""
import pytest
from mind2act_eval.scoring import (SCORING_VERSION, PressEdge, container_contains,
                                    lcs_length, partial_score)


def test_container_requires_real_support_and_actual_center_bounds():
    assert container_contains(supported=True, center=(0., 1.), lower=(-1., -1.), upper=(1., 1.))
    assert not container_contains(supported=False, center=(0., 0.), lower=(-1., -1.), upper=(1., 1.))
    assert not container_contains(supported=True, center=(1.000001, 0.), lower=(-1., -1.), upper=(1., 1.))


def test_mechanism_edges_require_release_and_allow_any_actuator():
    key = PressEdge(.002)
    assert [key.update(d) for d in [0., .002, .003, .002, .0019, .002]] == [False, True, False, False, False, True]


def test_lcs_allows_later_correct_events_but_preserves_order():
    assert lcs_length(['A', 'X', 'B', 'D'], ['A', 'B', 'C', 'D']) == 3
    assert lcs_length(['B', 'A'], ['A', 'B']) == 1
    assert lcs_length([], ['A']) == 0


def test_partial_score_is_exact_and_cannot_exceed_ceiling():
    assert partial_score(1, 6) == 20 / 6
    assert partial_score(4, 4) == 20
    with pytest.raises(ValueError):
        partial_score(5, 4)
    assert SCORING_VERSION == 'mind2act-final-scoring-20261105'
