import numpy as np
import pytest

from mind2act_eval.tasks.PR4.mechanism.fold_shapes import _finish
from mind2act_eval.tasks.PR4.mechanism.geometry import surface_layer_order


@pytest.mark.parametrize('fraction,expected', [(.978, False), (.979, False), (.980, True), (1., True)])
def test_original_target_region_fraction_boundary_is_not_relaxed(fraction, expected):
    n = 1000
    points = np.zeros((n, 3))
    points[:, 0] = np.linspace(-.5, .5, n)
    points[:, 1] = np.linspace(0., .5, n)
    points[:, 2] = .8
    inside = np.arange(n) < round(fraction*n)
    result = _finish(points, [np.zeros(n)], [np.zeros(10)],
        [dict(top_layer_fraction=1., projected_front_overlap_fraction=1.)], [1., 1.],
        inside, .1, .1, .8, 'synthetic boundary fixture')
    assert result['metrics']['in_target_fraction'] == pytest.approx(fraction)
    assert result['checks']['inside_target'] is expected


def test_layer_order_uses_actual_triangles_and_cannot_bridge_a_missing_surface():
    triangles = np.array([[[0., 0., .8], [1., 0., .8], [0., 1., .8]]])
    queries = np.array([[.2, .2, .803], [.9, .9, .803]])
    result = surface_layer_order(queries, triangles)
    assert result['top_layer_fraction'] == 1.
    assert result['projected_front_overlap_fraction'] == .5
    assert result['overlapping_front_samples'] == 1


def test_a_cloth_point_below_the_fixed_surface_does_not_count_as_top_layer():
    triangles = np.array([[[0., 0., .8], [1., 0., .8], [0., 1., .8]]])
    assert surface_layer_order([[.2, .2, .799]], triangles)['top_layer_fraction'] == 0.


@pytest.mark.parametrize('shift,expected', [(.0179999, True), (.018, True), (.0180001, False)])
def test_final_target_area_uses_the_declared_ten_mm_expansion(shift, expected):
    from mind2act_eval.tasks.PR4.mechanism.terminal_shape import terminal_shape
    points = np.array([[-.2+shift,0.,.8], [.2+shift,0.,.8],
                       [.2+shift,.2,.8], [-.2+shift,.2,.8]])
    result = terminal_shape(points, [[0,1,2],[0,2,3]], 'easy', [0.,0.], .4)
    assert result['shape_success'] is expected
    assert result['metrics']['in_expanded_target_area_fraction'] == pytest.approx(
        1-(shift-.01)/.4)
