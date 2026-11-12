"""An observed exchange prefix is a valid failure; measured divergence remains invalid."""
import pytest

from mind2act_eval.tasks.M1.scorer import score
from test_m1_scorer import Film


@pytest.mark.parametrize('cutoff', [1, 600, 'first_arrival', 'last_exchange_tick'])
def test_stop_during_exchange_scores_the_observed_prefix(cutoff):
    film = Film([]).build()
    if cutoff == 'first_arrival':
        cutoff = film.arrivals[0]['step']
    elif cutoff == 'last_exchange_tick':
        cutoff = film.start - 1
    film.rows = film.rows[:cutoff]
    film.start = None
    result = score(film.evidence())
    assert result.value == 0
    assert not result.components['success']
    assert not result.components['premise']
    assert result.components['route'] == ()
    assert result.components['completion'] == 0


def test_observed_arrival_divergence_remains_invalid():
    film = Film([]).build()
    first = film.arrivals[0]['step']
    film.rows = film.rows[:first]
    film.rows[-1]['target_center'] = [99, 99, 99]
    film.start = None
    with pytest.raises(ValueError, match='environment exchange path diverged'):
        score(film.evidence())


def test_empty_exchange_trace_remains_invalid():
    film = Film([]).build()
    film.rows = []
    film.start = None
    with pytest.raises(ValueError, match='requires every physics tick'):
        score(film.evidence())


def test_exchange_prefix_does_not_require_unused_robot_contact():
    film = Film([]).build()
    film.rows = film.rows[:600]
    film.start = None
    del film.rows[-1]['target_robot_contact']
    assert score(film.evidence()).value == 0
