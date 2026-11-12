"""CPU broad-phase equivalence; actual private meshes are checked on a GPU."""
import numpy as np
import pytest

from mind2act_eval.tasks.P1.mechanism.batched_rays import BatchedCupRays, ProjectedBoundsIndex, _QueryTree


def brute(bounds, queries):
    return [set(np.flatnonzero(((q[:3] <= bounds[:, 3:]) &
                              (q[3:] >= bounds[:, :3])).all(axis=1))) for q in queries]


def unpack(index, queries, axis):
    ids, offsets = index.query(queries, axis)
    return [set(ids[offsets[i]:offsets[i + 1]]) for i in range(len(queries))]


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('resolution', [1, 7, 32])
def test_batch_candidates_equal_inclusive_aabb_reference(axis, resolution):
    rng = np.random.default_rng(15483)
    lower = rng.uniform(-1, 1, (100, 3))
    upper = lower + rng.uniform(0, .4, (100, 3))
    boxes = np.hstack((lower, upper))
    bounds = [lower.min(axis=0), upper.max(axis=0)]
    index = ProjectedBoundsIndex(boxes, bounds, resolution)
    starts = rng.uniform(-2, 2, (180, 3))
    queries = np.hstack((starts, starts + rng.uniform(0, 2, (180, 3))))
    # Inclusive shared face, exact cell boundary, nextafter on either side,
    # zero-volume box, outside grid, and one query spanning every grid cell.
    face = lower[0].copy()
    extras = [np.r_[face, face], np.r_[upper[0], upper[0]],
              np.r_[np.nextafter(upper[0], np.inf), np.nextafter(upper[0], np.inf)],
              np.r_[np.nextafter(lower[0], -np.inf), np.nextafter(lower[0], -np.inf)],
              np.array([-5., -5., -5., 5., 5., 5.])]
    queries = np.vstack((queries, extras))
    assert unpack(index, queries, axis) == brute(boxes, queries)


@pytest.mark.parametrize('axis', [0, 1])
def test_degenerate_projection_and_duplicate_cell_membership(axis):
    boxes = np.array([[0., 0., 0., 1., 1., 0.], [.25, .25, 0., .75, .75, 0.]])
    index = ProjectedBoundsIndex(boxes, [[0., 0., 0.], [1., 1., 0.]])
    queries = np.array([[0., 0., 0., 1., 1., 0.], [1., 1., 1., 2., 2., 2.]])
    ids, offsets = index.query(queries, axis)
    assert list(ids) == [0, 1]
    assert list(offsets) == [0, 2, 2]
    assert unpack(index, queries, axis) == brute(boxes, queries)


def test_local_proxy_preserves_mismatched_future_query_behavior():
    class Tree:
        bounds = [0., 0., 0., 1., 1., 1.]
        calls = 0

        def intersection(self, bounds):
            self.calls += 1
            return [9]

    original = Tree()
    queries = np.array([[0., 0., 0., .1, .1, .1]])
    proxy = _QueryTree(original, queries, np.array([2, 3]), np.array([0, 2]))
    assert list(proxy.intersection(queries[0])) == [2, 3]
    assert list(proxy.intersection([0., 0., 0., 2., 2., 2.])) == [9]
    assert original.calls == proxy.fallback_calls == 1


def test_empty_query_batch_and_alternative_backend_passthrough():
    boxes = np.array([[0., 0., 0., 1., 1., 1.]])
    ids, offsets = ProjectedBoundsIndex(boxes, [[0., 0., 0.], [1., 1., 1.]]).query(np.empty((0, 6)), 0)
    assert ids.size == 0 and offsets.tolist() == [0]
    from types import SimpleNamespace
    expected = np.array([True, False])
    query = object.__new__(BatchedCupRays)
    query.mesh = SimpleNamespace(ray=SimpleNamespace(intersects_any=lambda origins, directions: expected))
    query.compatible = False
    query.calls = query.ray_points = query.fallback_calls = 0
    assert query.intersects_any(np.zeros((2, 3)), np.tile([1, 0, 0], (2, 1))) is expected
    assert (query.calls, query.ray_points, query.fallback_calls) == (1, 2, 2)


def test_trimesh_ray_hits_equal_reference_when_optional_runtime_available():
    trimesh = pytest.importorskip('trimesh')
    pytest.importorskip('rtree')
    from mind2act_eval.tasks.P1.mechanism.batched_rays import BatchedCupRays
    mesh = trimesh.creation.box()
    # This assertion exercises the triangle broad phase; an installed Embree
    # backend intentionally takes the separately tested passthrough path.
    from trimesh.ray.ray_triangle import RayMeshIntersector
    mesh.ray = RayMeshIntersector(mesh)
    query = BatchedCupRays(mesh)
    rng = np.random.default_rng(3572)
    origins = rng.uniform(-.8, .8, (150, 3))
    for axis in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
        directions = np.tile(axis, (len(origins), 1))
        np.testing.assert_array_equal(query.intersects_any(origins, directions),
                                      mesh.ray.intersects_any(origins, directions))
    assert query.calls == 4 and query.ray_points == 600 and query.fallback_calls == 0
