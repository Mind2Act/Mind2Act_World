"""Original PR4 hexagon shape checks; no initial mesh authoring or state setters."""
import numpy as np


def outline(center, width, folded=False):
    r = width / 2
    h = width * np.sqrt(3) / 4
    relative = ([[-r, 0], [r, 0], [r / 2, h], [-r / 2, h]] if folded else
                [[-r / 2, -h], [r / 2, -h], [r, 0],
                 [r / 2, h], [-r / 2, h], [-r, 0]])
    return np.asarray(relative) + center


def shape(points, initial, triangles, center, width, top, n):
    from .fold_shapes import _finish
    from .geometry import surface_layer_order
    p = np.asarray(points)
    mesh = p.reshape(n, n, 3)
    mid = n // 2
    front, rear = mesh[:mid], mesh[:mid:-1]
    errors = np.linalg.norm(front[..., :2] - rear[..., :2], axis=2)
    boundary_errors = np.r_[errors[0], errors[:, 0], errors[:, -1]]
    material_rows = np.arange(n * n) // n
    triangles = np.asarray(triangles)
    back_triangles = triangles[(material_rows[triangles] > mid + 2).all(1)]
    layer = surface_layer_order(front[:-3, 2:-2], p[back_triangles])
    faces = p[triangles]
    normals_z = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])[:, 2]
    orientation = [float(np.mean(normals_z[(material_rows[triangles] < mid - 2).all(1)] < 0)),
                   float(np.mean(normals_z[(material_rows[triangles] > mid + 2).all(1)] > 0))]
    x, y = (p[:, :2] - center).T
    h = width * np.sqrt(3) / 4
    inside = ((y >= -.010) & (y <= h + .010) &
              (abs(x) + np.maximum(y, 0) / np.sqrt(3) <= width / 2 + .010))
    result = _finish(p, [errors], [boundary_errors], [layer], orientation,
                     inside, width * .85, h * .75, top,
                     'Single regular-hexagon symmetry fold; fixed trapezoid target, '
                     'immutable reflected material pairs and both slanted outlines')
    maximum = float(boundary_errors.max())
    result['metrics']['max_outline_pair_xy_error_m'] = maximum
    result['checks']['all_outline_pairs_aligned'] = maximum < .010
    result['shape_success'] = all(result['checks'].values())
    result['fixed_target_vertices_xy'] = outline(center, width, folded=True).tolist()
    return result
