"""Exact SAT equivalence and allocation guards, without executing a dense old SAT."""
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mind2act_eval.tasks.P1.mechanism import cup_geometry as geometry


def old_intersects(cup, food, pose):
    pose = np.asarray(pose)
    rotation = Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()
    points = food.vertices @ rotation.T + pose[:3]
    if np.any(points.max(0) < cup.vertices.min(0)) or np.any(points.min(0) > cup.vertices.max(0)):
        return False
    axes = geometry._axes(np.vstack((cup.normals, food.normals @ rotation.T,
        np.cross(cup.edges[:, None, :], (food.edges @ rotation.T)[None, :, :]).reshape(-1, 3))))
    a, b = cup.vertices @ axes.T, points @ axes.T
    return not bool(np.any(a.max(0) < b.min(0) - 1e-10) or np.any(b.max(0) < a.min(0) - 1e-10))


def box():
    return np.array(list(product([-1., 1.], repeat=3)))


def test_dense_runtime_shapes_never_materialize_cartesian_cross(monkeypatch):
    # Authoritative P_hard_54fa7dda921d91d8 processed edge/vertex counts.
    # A guard rejects the old call before NumPy can allocate 163,316,520 axes.
    cup = geometry.OccupancyVolume(box())
    food = geometry.OccupancyVolume(box())
    cup.vertices = np.tile(cup.vertices, (279, 1))[:2230]
    food.vertices = np.tile(food.vertices, (1022, 1))[:8176]
    cup.edges = np.tile(cup.edges, (2220, 1))[:6660]
    food.edges = np.tile(food.edges, (8174, 1))[:24522]
    calls = []
    def bounded_cross(a, b, *args, **kwargs):
        size = np.prod(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]))
        assert size <= 1024, 'unsafe Cartesian cross allocation'
        calls.append(size)
        # Force a separator in the first batch with overlapping AABBs.
        return np.tile([1., 1., 0.], (int(size), 1))
    cup.normals = food.normals = np.empty((0, 3))
    cup.vertices[:] = [0., 1., 0.]
    cup.vertices[0] = [1., 0., 0.]
    food.vertices[:] = [.75, .75, 0.]
    monkeypatch.setattr(np, 'cross', bounded_cross)
    assert not cup.intersects(food, [0, 0, 0, 1, 0, 0, 0])
    assert calls


@pytest.mark.parametrize('seed', range(5))
def test_exact_original_convex_results(seed):
    rng = np.random.default_rng(seed)
    cup = geometry.OccupancyVolume(rng.normal(size=(20, 3)))
    food = geometry.OccupancyVolume(rng.normal(size=(15, 3)) * .5)
    assert cup.intersects(food, [0, 0, 0, 1, 0, 0, 0]) == old_intersects(cup, food, [0, 0, 0, 1, 0, 0, 0])
    for _ in range(12):
        q = Rotation.random(random_state=rng).as_quat()
        pose = [*rng.uniform(-3, 3, 3), q[3], *q[:3]]
        assert cup.intersects(food, pose) == old_intersects(cup, food, pose)


@pytest.mark.parametrize('delta', [0., -1e-10, 1e-10, -1e-9, 1e-9])
def test_contact_threshold(delta):
    cup = geometry.OccupancyVolume(box())
    food = geometry.OccupancyVolume(box())
    pose = [2 + delta, 0, 0, 1, 0, 0, 0]
    assert cup.intersects(food, pose) == old_intersects(cup, food, pose)


def test_global_first_representatives_and_spill_cleanup(monkeypatch, tmp_path):
    # Equal rounded keys can have different unrounded unit vectors. Both chunk
    # boundaries and disk spill must retain the original first representative.
    vectors = np.array([[1., 2e-11, 0.], [0., 1., 0.],
                        [1., -2e-11, 0.], [0., 0., 1.],
                        [-1., 2e-11, -0.], [0., -1., 0.]])
    monkeypatch.setattr(geometry, '_SAT_AXIS_BATCH', 2)
    monkeypatch.setattr(geometry, '_SAT_MEMORY_KEYS', 1)
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    batches = geometry._sat_axes(vectors, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
    actual = np.concatenate(list(batches))
    expected = geometry._axes(vectors)
    order = lambda a: a[np.lexsort(np.round(a, 10).T[::-1])]
    np.testing.assert_array_equal(order(actual), order(expected))
    assert not list(tmp_path.iterdir())
    batches = geometry._sat_axes(vectors, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
    next(batches)
    assert list(tmp_path.iterdir())
    batches.close()
    assert not list(tmp_path.iterdir())


def test_projection_allocation_is_bounded(monkeypatch):
    class GuardedVertices(np.ndarray):
        def __matmul__(self, axes):
            assert self.shape[0] <= 1024
            assert axes.shape[1] <= 1024
            return super().__matmul__(axes)
    rng = np.random.default_rng(92)
    vertices = rng.normal(size=(8176, 3))
    axes = geometry._axes(rng.normal(size=(1024, 3)))
    expected = vertices @ axes.T
    lo, hi = geometry._projection_bounds(vertices.view(GuardedVertices), axes)
    np.testing.assert_array_equal(lo, expected.min(0))
    np.testing.assert_array_equal(hi, expected.max(0))


def test_dense_source_geometry_when_supplied(monkeypatch):
    # Do not bundle private environment geometry in source or synthesize a
    # replacement. This opt-in check uses only the explicitly supplied fixture.
    import hashlib
    import os
    path = os.environ.get('P1_SAT_GEOMETRY')
    if not path:
        pytest.skip('set P1_SAT_GEOMETRY to the authoritative hard geometry.npz')
    with open(path, 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'bc187ae6d5119305f8f9e0ca29c88e043f9b2d9776a43714383b170d0b036483'
    with np.load(path, allow_pickle=False) as data:
        cup = geometry.OccupancyVolume(data['cup_cream_vertices'])
        food = geometry.OccupancyVolume(data['food_waffle_1'])
    assert (len(cup.vertices), len(cup.edges)) == (2230, 6660)
    assert (len(food.vertices), len(food.edges)) == (8176, 24522)
    translation = cup.vertices.max(0) - food.vertices.min(0) - 1e-6
    points = food.vertices + translation
    assert np.all(points.min(0) <= cup.vertices.max(0))
    assert np.all(points.max(0) >= cup.vertices.min(0))
    # Establish a face separator before invoking SAT: never accidentally run
    # 163M edge pairs on a dense fully overlapping test shape.
    axes = geometry._axes(np.vstack((cup.normals, food.normals)))[:1024]
    cup_lo, cup_hi = geometry._projection_bounds(cup.vertices, axes)
    food_lo, food_hi = geometry._projection_bounds(points, axes)
    assert np.any(cup_hi < food_lo - 1e-10) or np.any(food_hi < cup_lo - 1e-10)
    assert not cup.intersects(food, [*translation, 1., 0., 0., 0.])
    def no_axis_traversal(*args):
        raise AssertionError('dense positive witness must avoid the full SAT traversal')
    monkeypatch.setattr(geometry, '_sat_vectors', no_axis_traversal)
    center = (cup.vertices.min(0) + cup.vertices.max(0) - food.vertices.min(0) - food.vertices.max(0)) / 2
    for dz in [0., .025]:
        assert cup.intersects(food, [*(center + [0., 0., dz]), 1., 0., 0., 0.])
    # Five millimeters above the cup's highest source vertex, the food mean
    # cannot be in the cup. A real food vertex still certifies partial entry.
    partial = center.copy()
    partial[2] = cup.vertices.max(0)[2] + .005 - food.vertices.mean(0)[2]
    points = food.vertices + partial
    assert points.mean(0)[2] > cup.vertices.max(0)[2] + .004
    assert cup.intersects(food, [*partial, 1., 0., 0., 0.])
    contact = cup.vertices[np.argmax(cup.vertices[:, 2])] - food.vertices[np.argmin(food.vertices[:, 2])]
    assert cup.intersects(food, [*contact, 1., 0., 0., 0.])
    # Seeded CPU diagnostic poses on the unchanged source geometry, not poses
    # recovered from the stalled native actor. Both miss all prior certificates.
    for pose in [
        [-.18368282602982614, .0981740839253093, .9165301024555721,
         -.10983074061357634, -.7955577842280109, -.578692239465346, .14191656828247207],
        [-.20183178978061533, -.062414730128739736, .9400669390982974,
         -.6256930031741726, .6110116375085266, -.4633146184337332, .14322223624017988],
    ]:
        points = geometry.hull_points(food.vertices, pose)
        assert np.any(points.mean(0) > cup.vertices.max(0))
        with monkeypatch.context() as patch:
            patch.setattr(geometry, '_general_overlap_certificate', lambda *args: False)
            assert not cup._certifies_overlap(points)
        assert geometry._general_overlap_certificate(cup.vertices, points)
        assert cup.intersects(food, pose)


def test_transformed_contact_with_overlapping_aabbs():
    rotation = Rotation.from_euler('z', .43).as_matrix()
    cup = geometry.OccupancyVolume(box() @ rotation.T)
    food = geometry.OccupancyVolume(box() @ rotation.T)
    for delta in [-2e-10, -1e-10, 0., 1e-10, 2e-10]:
        pose = [*(rotation @ np.array([2 + delta, 0., 0.])), 1., 0., 0., 0.]
        assert cup.intersects(food, pose) == old_intersects(cup, food, pose)


def test_streamed_axes_match_original_representatives(monkeypatch):
    rng = np.random.default_rng(482)
    cup = geometry.OccupancyVolume(rng.normal(size=(12, 3)))
    food = geometry.OccupancyVolume(rng.normal(size=(10, 3)))
    rotation = Rotation.from_euler('xyz', [.3, -.7, .2]).as_matrix()
    normals, edges = food.normals @ rotation.T, food.edges @ rotation.T
    expected = geometry._axes(np.vstack((cup.normals, normals,
        np.cross(cup.edges[:, None, :], edges[None, :, :]).reshape(-1, 3))))
    monkeypatch.setattr(geometry, '_SAT_AXIS_BATCH', 7)
    monkeypatch.setattr(geometry, '_SAT_MEMORY_KEYS', 5)
    actual = np.concatenate(list(geometry._sat_axes(cup.normals, normals, cup.edges, edges)))
    order = np.lexsort(np.round(actual, 10).T[::-1])
    np.testing.assert_array_equal(actual[order], expected)


@pytest.mark.parametrize('point,expected', [([.25, .25, .25], True), ([0., 0., 0.], True), ([.5, .5, .5], False)])
def test_rational_certificate_signs(point, expected):
    tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert geometry._centroid_in_tetrahedron(np.tile(point, (5, 1)), tetra) == expected
    assert not geometry._centroid_in_tetrahedron(np.tile(point, (5, 1)), np.zeros((4, 3)))


def test_bounded_fallback_full_overlap_matches_original(monkeypatch):
    monkeypatch.setattr(geometry.OccupancyVolume, '_certifies_overlap', lambda *args: False)
    monkeypatch.setattr(geometry, '_SAT_MEMORY_KEYS', 5)
    rng = np.random.default_rng(23)
    cup = geometry.OccupancyVolume(rng.normal(size=(18, 3)))
    food = geometry.OccupancyVolume(rng.normal(size=(13, 3)))
    for angle in [0., .43]:
        q = Rotation.from_euler('xyz', [angle, -.2, .1]).as_quat()
        for x in [0., .5, 4.]:
            pose = [x, 0., 0., q[3], *q[:3]]
            assert cup.intersects(food, pose) == old_intersects(cup, food, pose)


def test_certificate_arithmetic_bound_only_controls_fast_path(monkeypatch):
    cup = geometry.OccupancyVolume(box() + [1000., 0., 0.])
    food = geometry.OccupancyVolume(box())
    assert not cup._certifies_overlap(food.vertices + [1000., 0., 0.])
    pose = [1000., 0., 0., 1., 0., 0., 0.]
    assert cup.intersects(food, pose) == old_intersects(cup, food, pose)


def test_certificate_nonfinite_and_inverse_failure_use_fallback(monkeypatch):
    cup = geometry.OccupancyVolume(box())
    food = geometry.OccupancyVolume(box() * .5)
    assert not cup._certifies_overlap(food.vertices + [float('nan'), 0., 0.])
    calls = []
    def failed_inverse(*args):
        calls.append(True)
        raise np.linalg.LinAlgError('diagnostic ill-conditioned inverse')
    monkeypatch.setattr(np.linalg, 'inv', failed_inverse)
    pose = [0., 0., 0., 1., 0., 0., 0.]
    assert cup.intersects(food, pose) == old_intersects(cup, food, pose)
    assert calls == [True]


@pytest.mark.parametrize('delta', [-2e-10, -2e-12, -2e-13, 0., 2e-13, 2e-12, 2e-10])
def test_close_source_pair_preserves_original_tolerance(delta, monkeypatch):
    rotation = Rotation.from_euler('z', .43).as_matrix()
    cup = geometry.OccupancyVolume(box() @ rotation.T)
    food = geometry.OccupancyVolume(box() @ rotation.T)
    pose = [*(rotation @ np.array([2 + delta, 0., 0.])), 1., 0., 0., 0.]
    expected = old_intersects(cup, food, pose)
    if 0 <= delta <= 2e-13:
        assert expected
        def no_traversal(*args):
            raise AssertionError('close source pair should certify every old tolerance comparison')
        monkeypatch.setattr(geometry, '_sat_vectors', no_traversal)
    assert cup.intersects(food, pose) == expected


def test_crossing_hulls_without_tested_point_witness(monkeypatch):
    cup = geometry.OccupancyVolume(box())
    food = geometry.OccupancyVolume(box() * [3.5, .2, .2])
    pose = [1.5, 0., 0., 1., 0., 0., 0.]
    points = food.vertices + pose[:3]
    assert points.mean(0)[0] > cup.vertices.max(0)[0]
    assert np.all(np.any(np.abs(points) > 1., axis=1))
    assert old_intersects(cup, food, pose)
    def no_traversal(*args):
        raise AssertionError('general convex intersection must avoid full SAT')
    monkeypatch.setattr(geometry, '_sat_vectors', no_traversal)
    assert cup.intersects(food, pose)


@pytest.mark.parametrize('offset,expected', [(0., True), (2e-13, False), (-2e-13, False)])
def test_exact_shared_combination_planar(offset, expected):
    # Neither polytope has a vertex in the other. Equality in the missing
    # dimension must be exact, even far below the LP feasibility tolerance.
    cup = np.array([[0., 0., 0.], [2., 0., 0.]])
    food = np.array([[1., -1., offset], [1., 1., offset]])
    assert geometry._exact_shared_combination(cup, food) == expected
    assert geometry._exact_shared_combination(cup[::-1], food[::-1]) == expected


def test_exact_shared_combination_rejects_negative_and_degenerate_weights():
    cup = np.array([[0., 0., 0.], [2., 0., 0.]])
    assert not geometry._exact_shared_combination(cup, np.array([[3., 0., 0.]]))
    assert not geometry._exact_shared_combination(np.zeros((2, 3)), np.zeros((2, 3)))
    assert not geometry._exact_shared_combination(box(), box())
    assert not geometry._exact_shared_combination(cup, np.empty((0, 3)))


@pytest.mark.parametrize('delta', [-2e-10, -2e-13, 0., 2e-13, 2e-10])
def test_general_certificate_near_contact_matches_old(delta):
    rotation = Rotation.from_euler('xyz', [.31, -.26, .43]).as_matrix()
    cup = geometry.OccupancyVolume(box() @ rotation.T)
    food = geometry.OccupancyVolume((box() * [3.5, .2, .2]) @ rotation.T)
    pose = [*(rotation @ [1.5, 1.2 + delta, 0.]), 1., 0., 0., 0.]
    points = food.vertices + pose[:3]
    expected = old_intersects(cup, food, pose)
    if geometry._general_overlap_certificate(cup.vertices, points):
        assert expected
    assert cup.intersects(food, pose) == expected


@pytest.mark.parametrize('failure', ['limit', 'nan', 'too_many', 'inexact', 'error'])
def test_general_nomination_uncertainty_uses_original_fallback(monkeypatch, failure):
    from scipy import optimize
    from types import SimpleNamespace
    def nominate(*args, **kwargs):
        if 'A_ub' in kwargs:
            return SimpleNamespace(success=False, x=np.zeros(5))
        assert kwargs['A_eq'].shape == (5, len(args[0]))
        assert kwargs['options'] == {'maxiter': 256, 'time_limit': .25, 'threads': 1}
        if failure == 'error':
            raise ValueError('numerical nomination failure')
        count = len(args[0])
        x = np.zeros(count)
        if failure == 'nan':
            x[0] = np.nan
        elif failure == 'too_many':
            x[:] = 1. / count
        else:
            x[0] = x[-1] = 1.
        return SimpleNamespace(success=failure != 'limit', x=x)
    monkeypatch.setattr(optimize, 'linprog', nominate)
    cup = geometry.OccupancyVolume(box())
    food = geometry.OccupancyVolume(box() * [3.5, .2, .2])
    pose = [1.5, 0., 0., 1., 0., 0., 0.]
    assert not geometry._general_overlap_certificate(cup.vertices, food.vertices + pose[:3])
    assert cup.intersects(food, pose) == old_intersects(cup, food, pose)


def test_edge_only_separator_avoids_full_traversal(monkeypatch):
    cup = geometry.OccupancyVolume(box() * [1.7, .3, .2])
    food = geometry.OccupancyVolume(box() * [1.4, .2, .3])
    pose = [1.031476115701939, .6974803547965085, -.48987468758036545,
            -.1099010146529896, -.8817947866175825, -.4195204723539709, -.1853706949552621]
    rotation = Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()
    points = food.vertices @ rotation.T + pose[:3]
    assert np.all(points.min(0) <= cup.vertices.max(0))
    assert np.all(points.max(0) >= cup.vertices.min(0))
    axes = geometry._axes(np.vstack((cup.normals, food.normals @ rotation.T)))
    al, ah = geometry._projection_bounds(cup.vertices, axes)
    bl, bh = geometry._projection_bounds(points, axes)
    assert not np.any(ah < bl - 1e-10) and not np.any(bh < al - 1e-10)
    assert not old_intersects(cup, food, pose)
    def no_traversal(*args):
        raise AssertionError('edge-only separator still enters the complete traversal')
    monkeypatch.setattr(geometry, '_sat_vectors', no_traversal)
    assert not cup.intersects(food, pose)


def test_near_contact_shared_combination_avoids_full_traversal(monkeypatch):
    # Skew source edges have no exact common point, but exact convex
    # combinations less than the original comparison tolerance apart.
    cup = geometry.OccupancyVolume(box())
    food = geometry.OccupancyVolume(box())
    cup.vertices = np.array([[-1., 0., -1.], [1., 0., 1.]])
    food.vertices = np.array([[0., -1., 2e-12], [0., 1., 2e-12]])
    cup.normals = food.normals = geometry._axes([[1., 0., -1.]])
    cup.edges = geometry._axes([[1., 0., 1.]])
    food.edges = np.array([[0., 1., 0.]])
    pose = [0., 0., 0., 1., 0., 0., 0.]
    assert old_intersects(cup, food, pose)
    def no_traversal(*args):
        raise AssertionError('near-contact support still enters the complete traversal')
    monkeypatch.setattr(geometry, '_sat_vectors', no_traversal)
    assert cup.intersects(food, pose)


@pytest.mark.parametrize('delta,expected', [(0., True), (2e-11, True), (5e-11, False), (2e-10, False)])
def test_close_combinations_exact_distance_bound(delta, expected):
    a = np.array([[-1., 0., -1.], [1., 0., 1.]])
    b = np.array([[0., -1., delta], [0., 1., delta]])
    assert geometry._exact_close_combinations(a, b, [.5, .5], [2., 2.]) == expected
    assert not geometry._exact_close_combinations(a, b, [-.5, 1.5], [.5, .5])
    assert not geometry._exact_close_combinations(a, b, [0., 0.], [.5, .5])
    assert not geometry._exact_close_combinations(a, b, [np.nan, .5], [.5, .5])
    assert not geometry._exact_close_combinations(a + 101, b, [.5, .5], [.5, .5])
    assert not geometry._exact_close_combinations(a[:0], b, [], [.5, .5])


def test_close_degenerate_support_is_tolerance_proof():
    # Exact common-point elimination intentionally rejects dependent columns;
    # exact convex combinations can still certify the original tolerant result.
    a = np.array([[0., 0., 0.], [0., 0., 0.]])
    b = np.array([[0., 0., 2e-12], [0., 0., 2e-12]])
    assert not geometry._exact_shared_combination(a, b)
    assert geometry._exact_close_combinations(a, b, [.5, .5], [.5, .5])


def test_global_representative_can_reverse_nominated_axis_result(monkeypatch):
    normals = np.array([[1., 4e-11, 0.], [1., -4e-11, 0.]])
    empty = np.empty((0, 3))
    a = np.array([[0., 100., 0.]])
    b = np.array([[8e-9, -100., 0.]])
    first = geometry._axes(normals)
    nominated = geometry._axes(normals[1:])
    assert len(first) == 1
    al, ah = geometry._projection_bounds(a, first)
    bl, bh = geometry._projection_bounds(b, first)
    assert not np.any(ah < bl - 1e-10) and not np.any(bh < al - 1e-10)
    al, ah = geometry._projection_bounds(a, nominated)
    bl, bh = geometry._projection_bounds(b, nominated)
    assert np.any(ah < bl - 1e-10)
    monkeypatch.setattr(geometry, '_source_axis_candidates', lambda *args: nominated)
    assert not geometry._certifies_source_separation(a, b, normals, empty, empty, empty)
    # A larger gap proves every possible first representative of the same key.
    b[0, 0] = 1e-5
    assert geometry._certifies_source_separation(a, b, normals, empty, empty, empty)


def test_nominated_axes_are_original_source_keys():
    cup = geometry.OccupancyVolume(box() * [1.7, .3, .2])
    food = geometry.OccupancyVolume(box() * [1.4, .2, .3])
    rotation = Rotation.from_euler('xyz', [.3, -.4, .7]).as_matrix()
    normals, edges = food.normals @ rotation.T, food.edges @ rotation.T
    points = food.vertices @ rotation.T + [.7, 2., 2.]
    candidates = geometry._source_axis_candidates(cup.vertices, points, cup.normals, normals, cup.edges, edges)
    # Compare full unrounded representatives from independently generated raw
    # original vectors, not merely approximate direction similarity.
    raw = np.vstack((cup.normals, normals,
        np.cross(cup.edges[:, None, :], edges[None, :, :]).reshape(-1, 3)))
    source = np.vstack([geometry._axes(v[None]) for v in raw])
    assert 0 < len(candidates) <= 272
    for candidate in candidates:
        assert np.any(np.all(source == candidate, axis=1))


@pytest.mark.parametrize('failure', ['limit', 'nan', 'zero', 'error'])
def test_separation_nomination_uncertainty_uses_full_fallback(monkeypatch, failure):
    from scipy import optimize
    from types import SimpleNamespace
    def nominate(*args, **kwargs):
        assert kwargs['A_ub'].shape[1] == 5
        assert kwargs['options'] == {'maxiter': 256, 'time_limit': .25, 'threads': 1}
        if failure == 'error':
            raise ValueError('diagnostic nomination failure')
        return SimpleNamespace(success=failure != 'limit', x=np.full(5, np.nan) if failure == 'nan' else np.zeros(5))
    monkeypatch.setattr(optimize, 'linprog', nominate)
    monkeypatch.setattr(geometry.OccupancyVolume, '_certifies_overlap', lambda *args: False)
    cup, food = geometry.OccupancyVolume(box()), geometry.OccupancyVolume(box())
    pose = [0., 0., 0., 1., 0., 0., 0.]
    calls = []
    original = geometry._sat_vectors
    def traversed(*args):
        calls.append(True)
        yield from original(*args)
    monkeypatch.setattr(geometry, '_sat_vectors', traversed)
    assert cup.intersects(food, pose) == old_intersects(cup, food, pose)
    assert calls == [True]


@pytest.mark.parametrize('delta', [-2e-11, 2e-11])
def test_close_source_pair_without_lp_or_full_traversal(delta, monkeypatch):
    rotation = Rotation.from_euler('xyz', [.31, -.26, .43]).as_matrix()
    cup = geometry.OccupancyVolume(box() @ rotation.T)
    food = geometry.OccupancyVolume(box() @ rotation.T)
    pose = [*(rotation @ [2 + delta, 0., 0.]), 1., 0., 0., 0.]
    assert old_intersects(cup, food, pose)
    monkeypatch.setattr(geometry, '_general_overlap_certificate', lambda *args: False)
    def no_full(*args):
        raise AssertionError('exact close source pair must avoid LP and full traversal')
    monkeypatch.setattr(geometry, '_sat_vectors', no_full)
    assert cup.intersects(food, pose)
