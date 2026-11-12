"""Original open-cup containment (P-148 baseline cases_unified/P/geometry.py; same predicate).

`contained_by_open_cup` tests measured food hull vertices against the cup's triangle sidewalls with four
horizontal rays per vertex. `outside_cup_bounds` is an exact early exit used to avoid ray casting:
the cup mesh lies inside its world bounds, so a hull vertex outside them has an escaping horizontal
ray and the full test can never report "inside". It does not change any result.
"""
import numpy as np
from scipy.spatial.transform import Rotation

RAY_AXES = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])


def _axes(vectors):
    """Unique unoriented unit axes, omitting zero cross products."""
    vectors = np.asarray(vectors).reshape(-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    if not len(vectors):
        return vectors
    leading = np.argmax(np.abs(vectors) > 1e-10, axis=1)
    vectors *= np.where(vectors[np.arange(len(vectors)), leading] < 0, -1., 1.)[:, None]
    _, indices = np.unique(np.round(vectors, 10), axis=0, return_index=True)
    return vectors[indices]


# Bound both Cartesian edge generation and vertex/axis projection temporaries.
_SAT_AXIS_BATCH = 1024
_SAT_VERTEX_BATCH = 1024
_SAT_MEMORY_KEYS = 16384


def _sat_vectors(cup_normals, food_normals, cup_edges, food_edges):
    for normals in (cup_normals, food_normals):
        for start in range(0, len(normals), _SAT_AXIS_BATCH):
            yield normals[start:start + _SAT_AXIS_BATCH]
    # Preserve the old Cartesian row order without materializing the product.
    for edge in cup_edges:
        for start in range(0, len(food_edges), _SAT_AXIS_BATCH):
            yield np.cross(edge, food_edges[start:start + _SAT_AXIS_BATCH])


def _sat_axes(cup_normals, food_normals, cup_edges, food_edges):
    """Stream the same first normalized representative for each rounded key.

    Independent chunk deduplication would retain later, slightly different
    representatives too. Keep a bounded key cache, spilling to a temporary
    SQLite index when necessary; the index has a fixed 2 MiB page cache.
    """
    seen, database, directory = set(), None, None
    try:
        for vectors in _sat_vectors(cup_normals, food_normals, cup_edges, food_edges):
            axes = _axes(vectors)
            keys = np.round(axes, 10)
            keys[keys == 0] = 0.  # np.unique treats signed zero as equal.
            keep = []
            for index, key in enumerate(keys):
                key = key.tobytes()
                if database is None and len(seen) >= _SAT_MEMORY_KEYS:
                    import sqlite3
                    import tempfile
                    directory = tempfile.TemporaryDirectory(prefix="p1-sat-")
                    database = sqlite3.connect(directory.name + "/axes.sqlite")
                    database.execute("PRAGMA journal_mode=OFF")
                    database.execute("PRAGMA synchronous=OFF")
                    database.execute("PRAGMA cache_size=-2048")
                    database.execute("CREATE TABLE axes (key BLOB PRIMARY KEY) WITHOUT ROWID")
                    database.executemany("INSERT INTO axes VALUES (?)", ((k,) for k in seen))
                    seen.clear()
                if database is None:
                    if key in seen:
                        continue
                    seen.add(key)
                elif not database.execute("INSERT OR IGNORE INTO axes VALUES (?)", (key,)).rowcount:
                    continue
                keep.append(index)
            if keep:
                yield axes[keep]
    finally:
        if database is not None:
            database.close()
        if directory is not None:
            directory.cleanup()


def _projection_bounds(vertices, axes):
    lo, hi = np.full(len(axes), np.inf), np.full(len(axes), -np.inf)
    for start in range(0, len(vertices), _SAT_VERTEX_BATCH):
        projection = vertices[start:start + _SAT_VERTEX_BATCH] @ axes.T
        lo = np.minimum(lo, projection.min(axis=0))
        hi = np.maximum(hi, projection.max(axis=0))
    return lo, hi


def _witness_tetrahedra(vertices):
    # These source vertices supply positive certificates only. They never
    # replace the hull or any axis in the fallback SAT predicate.
    from itertools import combinations, product
    directions = np.array(list(product([-1., 1.], repeat=3)))
    indices = np.unique(np.argmax(vertices @ directions.T, axis=0))
    if len(indices) < 4:
        return np.empty((0, 4, 3)), np.empty((0, 3, 3))
    tetrahedra = vertices[list(combinations(indices, 4))]
    edges = tetrahedra[:, 1:] - tetrahedra[:, :1]
    # Conditioning is only a candidate filter; exact determinants decide truth.
    valid = abs(np.linalg.det(edges)) > 1e-8 * np.prod(np.linalg.norm(edges, axis=2), axis=1)
    try:
        inverses = np.linalg.inv(edges[valid])
    except np.linalg.LinAlgError:
        return np.empty((0, 4, 3)), np.empty((0, 3, 3))
    finite = np.isfinite(inverses).all(axis=(1, 2))
    return tetrahedra[valid][finite], inverses[finite]


def _determinant(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def _centroid_in_tetrahedron(points, tetrahedron):
    """Certify the exact rational mean of supplied food vertices in a source tetra."""
    from fractions import Fraction
    point = [sum((Fraction(float(v)) for v in points[:, j]), Fraction()) / len(points) for j in range(3)]
    tetra = [[Fraction(float(v)) for v in row] for row in tetrahedron]
    matrix = [[row[j] - tetra[0][j] for row in tetra[1:]] for j in range(3)]
    delta = [point[j] - tetra[0][j] for j in range(3)]
    denominator = _determinant(matrix)
    numerators = [_determinant([[delta[r] if c == i else matrix[r][c] for c in range(3)]
                               for r in range(3)]) for i in range(3)]
    numerators.append(denominator - sum(numerators))
    return bool(denominator and (all(v >= 0 for v in numerators) if denominator > 0
                                 else all(v <= 0 for v in numerators)))


def _exact_shared_combination(cup, food):
    """Prove a shared convex combination of at most five source vertices.

    The five equations are A*a - B*b = 0, sum(a) = sum(b) = 1.
    Solve and check all equations over exact binary-float rationals; nomination
    tolerances, residuals and weights never establish overlap.
    """
    from fractions import Fraction
    count = len(cup) + len(food)
    if not 1 <= count <= 5:
        return False
    columns = [[Fraction(float(v)) for v in p] + [Fraction(1), Fraction(0)] for p in cup]
    columns += [[-Fraction(float(v)) for v in p] + [Fraction(0), Fraction(1)] for p in food]
    matrix = [[column[row] for column in columns] + [Fraction(row >= 3)] for row in range(5)]
    for column in range(count):
        pivot = next((row for row in range(column, 5) if matrix[row][column]), None)
        if pivot is None:
            return False
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(5):
            if row != column:
                factor = matrix[row][column]
                matrix[row] = [x - factor * y for x, y in zip(matrix[row], matrix[column])]
    return (all(matrix[row][-1] >= 0 for row in range(count))
            and all(matrix[row][-1] == 0 for row in range(count, 5)))


def _exact_close_combinations(cup, food, cup_weights, food_weights):
    """Certify near contact from two exact source convex combinations.

    A common point is unnecessary for the original tolerant predicate. With
    finite coordinates <=100, L1 distance <=4e-11 bounds every normalized-axis
    gap by 8e-11; the existing conservative dot/subtraction error is <1e-12.
    Solver weights nominate rational combinations, never a residual tolerance.
    """
    from fractions import Fraction
    if (not len(cup) or not len(food) or not np.isfinite(cup).all() or not np.isfinite(food).all()
            or np.max(np.abs(cup)) > 100 or np.max(np.abs(food)) > 100):
        return False
    combinations = []
    for points, weights in ((cup, cup_weights), (food, food_weights)):
        if (len(points) != len(weights) or not np.isfinite(weights).all()
                or np.any(np.asarray(weights) < 0)):
            return False
        weights = [Fraction(float(value)) for value in weights]
        total = sum(weights)
        if not total:
            return False
        combinations.append([sum(w * Fraction(float(p[j])) for w, p in zip(weights, points)) / total
                             for j in range(3)])
    return sum(abs(a - b) for a, b in zip(*combinations)) <= Fraction(4, 10**11)


def _general_overlap_certificate(cup, food):
    """Bounded LP nomination, followed by a source-only exact positive proof.

    Five rows and O(V) storage replace no geometry or fallback axes. A basic
    feasible solution needs at most five columns. Failure, iteration/time limits,
    degenerate support, and near-contact numerical feasibility all fall back.
    """
    import warnings
    from scipy.optimize import OptimizeWarning, linprog
    count = len(cup) + len(food)
    matrix = np.zeros((5, count))
    matrix[:3, :len(cup)] = cup.T
    matrix[:3, len(cup):] = -food.T
    matrix[3, :len(cup)] = 1.
    matrix[4, len(cup):] = 1.
    try:
        # SciPy forwards HiGHS' thread option verbatim but warns because it is
        # outside SciPy's own option list. Bound native worker/stack allocation
        # too; otherwise a many-core host can create hundreds of solver threads.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads.*',
                                    category=OptimizeWarning)
            result = linprog(np.zeros(count), A_eq=matrix, b_eq=[0., 0., 0., 1., 1.],
                             bounds=(0., None), method='highs-ds',
                             options={'maxiter': 256, 'time_limit': .25, 'threads': 1})
    except ValueError:
        return False
    if not result.success or not np.isfinite(result.x).all():
        return False
    support = np.flatnonzero(result.x > 0.)
    if not 1 <= len(support) <= 5:
        return False
    a, b = support[support < len(cup)], support[support >= len(cup)] - len(cup)
    return (_exact_shared_combination(cup[a], food[b])
            or _exact_close_combinations(cup[a], food[b], result.x[a], result.x[len(cup) + b]))


def _source_axis_candidates(cup, food, cup_normals, food_normals, cup_edges, food_edges):
    """Nominate bounded source axes near an LP direction; never use it as an axis."""
    import warnings
    from scipy.optimize import OptimizeWarning, linprog
    # Maximize the gap between two support planes with an L-infinity bounded
    # normal. This direction is only a feature nomination, not a SAT proof.
    matrix = np.zeros((len(cup) + len(food), 5))
    matrix[:len(cup), :3], matrix[:len(cup), 3] = cup, -1.
    matrix[len(cup):, :3] = -food
    matrix[len(cup):, 3:] = 1.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads.*',
                                    category=OptimizeWarning)
            result = linprog([0., 0., 0., 0., -1.], A_ub=matrix, b_ub=np.zeros(len(matrix)),
                             bounds=[(-1., 1.)] * 3 + [(None, None)] * 2, method='highs-ds',
                             options={'maxiter': 256, 'time_limit': .25, 'threads': 1})
    except ValueError:
        return np.empty((0, 3))
    if not result.success or not np.isfinite(result.x).all():
        return np.empty((0, 3))
    direction = result.x[:3]
    length = np.linalg.norm(direction)
    if not length:
        return np.empty((0, 3))
    direction = direction / length
    # Selection only nominates existing face axes / existing edge pairs. The
    # complete source hull, skip rules and fallback are unchanged.
    normals = np.vstack((cup_normals, food_normals))
    normals = normals[np.argsort(-np.abs(normals @ direction), kind='stable')[:16]]
    a = cup_edges[np.argsort(np.abs(cup_edges @ direction), kind='stable')[:16]]
    b = food_edges[np.argsort(np.abs(food_edges @ direction), kind='stable')[:16]]
    crosses = np.cross(a[:, None, :], b[None, :, :]).reshape(-1, 3)
    return _axes(np.vstack((normals, crosses)))


def _certifies_source_separation(cup, food, cup_normals, food_normals, cup_edges, food_edges):
    """Prove separation on an original key, including its first representative.

    A nominated source axis can differ from the global first unrounded axis.
    Two normalized axes with the same round10 key differ by <2e-10 per
    component (including roundoff in NumPy's scaling/rounding). Bound the gap
    change for EVERY vertex pair, plus <2e-12 for both old/new dot products and
    tolerance subtractions. Only a strict gap beyond that bound certifies false.
    Uncertain rounding cells and solver failures keep the complete original SAT.
    """
    if (not np.isfinite(cup).all() or not np.isfinite(food).all()
            or np.max(np.abs(cup)) > 100 or np.max(np.abs(food)) > 100):
        return False
    axes = _source_axis_candidates(cup, food, cup_normals, food_normals, cup_edges, food_edges)
    if not len(axes):
        return False
    margin = 2e-10 * np.sum(np.max(np.abs(cup), axis=0) + np.max(np.abs(food), axis=0)) + 2e-12
    cup_lo, cup_hi = _projection_bounds(cup, axes)
    food_lo, food_hi = _projection_bounds(food, axes)
    return bool(np.any(cup_hi < food_lo - 1e-10 - margin)
                or np.any(food_hi < cup_lo - 1e-10 - margin))


class OccupancyVolume:
    """Convex food/cup envelope for partial occupancy, never full containment.

    A cup's convex envelope includes its open interior. The separating-axis
    test includes face normals and edge cross products, so overlapping AABBs
    alone cannot create an occupant. Geometry is cached once per fixed shape.
    Full food containment still uses the original cup sidewall rays below.
    """
    def __init__(self, vertices):
        from scipy.spatial import ConvexHull
        points = np.asarray(vertices, dtype=float)
        hull = ConvexHull(points)
        self.vertices = points[hull.vertices]
        self._tetrahedra = None
        self._vertex_tree = None
        self.normals = _axes(hull.equations[:, :3])
        faces = points[hull.simplices]
        self.edges = _axes(np.concatenate((faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 1],
                                           faces[:, 0] - faces[:, 2])))

    def _certifies_overlap(self, points):
        # A certified shared convex combination overlaps on EVERY source axis.
        # With coordinates <=100 and normalized-axis components <2, IEEE-754
        # three-term dot errors (conservative gamma_5 * sum(abs(products))) and the existing
        # tolerance subtraction total <1e-12, below the unchanged 1e-10 gap.
        # Outside this conservative arithmetic bound, use the original SAT.
        if (not np.isfinite(self.vertices).all() or not np.isfinite(points).all()
                or np.max(np.abs(self.vertices)) > 100 or np.max(np.abs(points)) > 100):
            return False
        # A sufficiently close pair also proves the ORIGINAL tolerance test
        # true on every axis: 2*L1_distance <=8e-11, plus <1e-12
        # arithmetic error, remains below the unchanged 1e-10 comparison.
        # The tree only nominates a pair; exact rational distance certifies it.
        from scipy.spatial import cKDTree
        if self._vertex_tree is None:
            self._vertex_tree = cKDTree(self.vertices)
        distances, indices = self._vertex_tree.query(points)
        nearest_pair = int(np.argmin(distances))
        if distances[nearest_pair] <= 4e-11:
            from fractions import Fraction
            a, b = self.vertices[indices[nearest_pair]], points[nearest_pair]
            distance = sum(abs(Fraction(float(x)) - Fraction(float(y))) for x, y in zip(a, b))
            if distance <= Fraction(4, 10**11):
                return True
        if self._tetrahedra is None:
            self._tetrahedra = _witness_tetrahedra(self.vertices)
        tetrahedra, inverses = self._tetrahedra
        # The mean of all food vertices and any single actual food vertex are
        # both exact convex combinations. A vertex can certify partial entry
        # while the mean remains outside the cup. This is a positive witness,
        # never a replacement or approximation of the shape being scored.
        nearest = np.argmin(np.linalg.norm(points - self.vertices.mean(axis=0), axis=1))
        for witness in (points, points[nearest:nearest + 1]):
            weights = np.einsum('ti,tij->tj', witness.mean(axis=0) - tetrahedra[:, 0], inverses)
            candidates = np.flatnonzero((weights > 1e-6).all(axis=1) & (weights.sum(axis=1) < 1 - 1e-6))
            # Floating arithmetic only nominates a candidate. Exact signs must
            # certify membership; missed candidates merely take the full SAT.
            if len(candidates) and _centroid_in_tetrahedron(witness, tetrahedra[candidates[0]]):
                return True
        return _general_overlap_certificate(self.vertices, points)

    def intersects(self, food, pose_wxyz):
        pose = np.asarray(pose_wxyz)
        rotation = Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()
        points = food.vertices @ rotation.T + pose[:3]
        if np.any(points.max(axis=0) < self.vertices.min(axis=0)) or np.any(points.min(axis=0) > self.vertices.max(axis=0)):
            return False
        if self._certifies_overlap(points):
            return True
        normals, edges = food.normals @ rotation.T, food.edges @ rotation.T
        if _certifies_source_separation(self.vertices, points, self.normals, normals, self.edges, edges):
            return False
        batches = _sat_axes(self.normals, normals, self.edges, edges)
        try:
            for axes in batches:
                cup_lo, cup_hi = _projection_bounds(self.vertices, axes)
                food_lo, food_hi = _projection_bounds(points, axes)
                if np.any(cup_hi < food_lo - 1e-10) or np.any(food_hi < cup_lo - 1e-10):
                    return False
            return True
        finally:
            batches.close()


def hull_points(local_hull_vertices, pose_wxyz):
    pose = np.asarray(pose_wxyz)
    rotation = Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()
    return np.asarray(local_hull_vertices) @ rotation.T + pose[:3]


def outside_cup_bounds(points, cup_world_bounds):
    lo, hi = np.asarray(cup_world_bounds[0], float), np.asarray(cup_world_bounds[1], float)
    return bool(np.any(points[:, :2] < lo[:2]) or np.any(points[:, :2] > hi[:2]))


def contained_by_open_cup(cup_mesh, local_hull_vertices, pose_wxyz, rim_z, support_z, *, ray_query=None):
    points = hull_points(local_hull_vertices, pose_wxyz)
    lo, hi = points.min(axis=0), points.max(axis=0)
    height = bool(lo[2] > support_z - .002 and hi[2] <= rim_z)
    fractions = []
    if height:
        origins = points + np.array([0, 0, .001])  # 1 mm up avoids ambiguous floor-contact rays
        for axis in RAY_AXES:
            query = cup_mesh.ray if ray_query is None else ray_query
            hits = query.intersects_any(origins, np.tile(axis, (len(points), 1)))
            fractions.append(float(hits.mean()))
    return {"inside": bool(height and len(fractions) == 4 and min(fractions) == 1.),
            "actual_mesh_bounds": [lo.tolist(), hi.tolist()], "sidewall_enclosure_fractions": fractions,
            "hull_vertex_count": len(points), "height_pass": height}


def inside_cup(cup_mesh, cup_world_bounds, local_hull_vertices, pose_wxyz, support_z, *, ray_query=None):
    """Same truth value as contained_by_open_cup(...)["inside"], with the exact bounds early exit."""
    points = hull_points(local_hull_vertices, pose_wxyz)
    if outside_cup_bounds(points, cup_world_bounds):
        return False
    return contained_by_open_cup(cup_mesh, local_hull_vertices, pose_wxyz, cup_world_bounds[1][2], support_z,
                                 ray_query=ray_query)["inside"]
