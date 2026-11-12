"""Conservative batch broad phase for P1's static cup triangle meshes.

Only candidate lookup changes. Trimesh still computes ray bounds and every
triangle intersection, including its original tolerances and forward test.
The installed rtree 1.3 has no intersection_v, so use a projected uniform grid.
"""
import numpy as np


class ProjectedBoundsIndex:
    """Index triangle AABBs; each query returns exactly the overlapping AABBs."""

    def __init__(self, triangle_bounds, bounds, resolution=32):
        self.triangles = np.asarray(triangle_bounds, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        self.resolution = resolution
        self.grids = {}
        for axis in (0, 1):
            perpendicular = [d for d in range(3) if d != axis]
            lo = self.bounds[0, perpendicular]
            span = self.bounds[1, perpendicular] - lo
            scale = resolution / np.where(span > 0, span, 1.)
            lower = self._cells(self.triangles[:, :3][:, perpendicular], lo, scale)
            upper = self._cells(self.triangles[:, 3:][:, perpendicular], lo, scale)
            cells = {}
            for x in range(resolution):
                for y in range(resolution):
                    mask = ((lower <= (x, y)) & (upper >= (x, y))).all(axis=1)
                    ids = np.flatnonzero(mask)
                    if len(ids):
                        cells[x, y] = ids
            self.grids[axis] = perpendicular, lo, scale, cells

    def _cells(self, values, lo, scale):
        return np.clip(np.floor((values - lo) * scale), 0, self.resolution - 1).astype(np.int64)

    def query(self, query_bounds, axis):
        """Return flattened triangle IDs plus per-query offsets, without Rtree calls."""
        query = np.asarray(query_bounds, dtype=np.float64)
        if not len(query):
            return np.empty(0, dtype=np.int64), np.array([0], dtype=np.int64)
        perpendicular, lo, scale, cells = self.grids[axis]
        lower = self._cells(query[:, :3][:, perpendicular], lo, scale)
        upper = self._cells(query[:, 3:][:, perpendicular], lo, scale)
        pairs = []
        count = len(self.triangles)
        first, last = lower.min(axis=0), upper.max(axis=0)
        for (x, y), triangles in cells.items():
            if x < first[0] or x > last[0] or y < first[1] or y > last[1]:
                continue
            rays = np.flatnonzero(((lower <= (x, y)) & (upper >= (x, y))).all(axis=1))
            if not len(rays):
                continue
            q = query[rays]
            t = self.triangles[triangles]
            overlap = ((q[:, None, :3] <= t[None, :, 3:]) &
                       (q[:, None, 3:] >= t[None, :, :3])).all(axis=2)
            r, c = np.nonzero(overlap)
            pairs.append(rays[r] * count + triangles[c])
        encoded = np.unique(np.concatenate(pairs)) if pairs else np.empty(0, dtype=np.int64)
        ray_ids, triangle_ids = encoded // count, encoded % count
        offsets = np.r_[0, np.cumsum(np.bincount(ray_ids, minlength=len(query)))]
        return triangle_ids, offsets


class _QueryTree:
    """Per-call proxy consumed only by the unchanged trimesh narrow phase."""

    def __init__(self, tree, queries, triangle_ids, offsets):
        self.original = tree
        self.bounds = tree.bounds
        self.queries, self.ids, self.offsets = queries, triangle_ids, offsets
        self.cursor = 0
        self.fallback_calls = 0

    def intersection(self, bounds):
        i = self.cursor
        self.cursor += 1
        if i < len(self.queries) and np.array_equal(bounds, self.queries[i]):
            return self.ids[self.offsets[i]:self.offsets[i + 1]]
        # Preserve correctness if a future trimesh version changes query order.
        self.fallback_calls += 1
        return self.original.intersection(bounds)


class BatchedCupRays:
    """Retain all rays/vertices; batch only the immutable triangle broad phase."""

    def __init__(self, mesh):
        from trimesh.ray.ray_triangle import RayMeshIntersector, ray_bounds, ray_triangle_id
        self.mesh = mesh
        self.ray_bounds, self.ray_triangle_id = ray_bounds, ray_triangle_id
        self.compatible = isinstance(mesh.ray, RayMeshIntersector)
        self.calls = self.ray_points = self.fallback_calls = 0
        if not self.compatible:
            return  # An alternative installed backend keeps its original behavior.
        self.tree = mesh.triangles_tree
        triangles = np.asarray(mesh.triangles)
        self.index = ProjectedBoundsIndex(np.hstack((triangles.min(axis=1), triangles.max(axis=1))), self.tree.bounds)

    def intersects_any(self, origins, directions):
        self.calls += 1
        self.ray_points += len(origins)
        axis = int(np.abs(directions[0]).argmax()) if len(directions) else 2
        if (not self.compatible or axis not in (0, 1) or
                not np.array_equal(directions, np.tile(directions[0], (len(directions), 1))) or
                not np.isfinite(origins).all()):
            self.fallback_calls += len(origins)
            return self.mesh.ray.intersects_any(origins, directions)
        queries = self.ray_bounds(origins, directions, self.tree.bounds)
        ids, offsets = self.index.query(queries, axis)
        proxy = _QueryTree(self.tree, queries, ids, offsets)
        _, ray_ids, _ = self.ray_triangle_id(
            triangles=self.mesh.triangles, ray_origins=origins, ray_directions=directions,
            triangles_normal=self.mesh.face_normals, tree=proxy)
        result = np.zeros(len(origins), dtype=bool)
        result[np.unique(ray_ids)] = True
        self.fallback_calls += proxy.fallback_calls
        return result
