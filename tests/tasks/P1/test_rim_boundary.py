"""Cup geometry must not reserve an unpublished eight-millimeter headroom."""
from types import SimpleNamespace
import numpy as np
import pytest
from mind2act_eval.tasks.P1.mechanism.cup_geometry import contained_by_open_cup


class EnclosedSides:
    def intersects_any(self, origins, directions):
        return np.ones(len(origins), dtype=bool)


@pytest.mark.parametrize('top', [.895, .900])
def test_food_inside_actual_rim_needs_no_extra_headroom(top):
    hull = np.array([[-.01, -.01, .800], [.01, .01, top]])
    result = contained_by_open_cup(SimpleNamespace(ray=EnclosedSides()), hull,
                                  [0., 0., 0., 1., 0., 0., 0.], .900, .790)
    assert result['height_pass'] and result['inside']


def test_food_above_actual_rim_still_fails_containment():
    hull = np.array([[-.01, -.01, .800], [.01, .01, .9001]])
    result = contained_by_open_cup(SimpleNamespace(ray=EnclosedSides()), hull,
                                  [0., 0., 0., 1., 0., 0., 0.], .900, .790)
    assert not result['inside']
