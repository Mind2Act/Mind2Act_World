"""Actual projected cloth silhouette in the fixed, authored table target frame.

Use the union of measured mesh triangles, never material-pair correspondence,
convex-hull replacement, or a best-fit translation/rotation. Boundary errors
use 4096 equal arc-length samples on each boundary (including holes); this is
numerical quadrature, not an added geometric acceptance tolerance.
"""
import numpy as np
from shapely import polygons, points as shapely_points, distance, line_interpolate_point
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import unary_union


def target_outline(level, center, width):
    if level == 'easy':
        relative = np.array([[-width/2, 0.], [width/2, 0.], [width/2, width/2], [-width/2, width/2]])
    elif level == 'medium':
        r = width / np.sqrt(2)
        relative = np.array([[-r, 0.], [r, 0.], [0., r]])
    elif level == 'hard':
        h = width * np.sqrt(3) / 4
        relative = np.array([[-width/2, 0.], [width/2, 0.], [width/4, h], [-width/4, h]])
    else:
        raise ValueError('unknown PR4 target shape')
    return relative + np.asarray(center)


def boundary_samples(boundary):
    lines = list(boundary.geoms) if hasattr(boundary, 'geoms') else [boundary]
    lengths = np.array([line.length for line in lines])
    cumulative = np.r_[0., np.cumsum(lengths)]
    positions = np.arange(4096) * cumulative[-1] / 4096
    samples = []
    for index, line in enumerate(lines):
        selected = positions[(positions >= cumulative[index]) & (positions < cumulative[index+1])]
        if len(selected):
            samples.extend(line_interpolate_point(line, selected - cumulative[index]))
    return np.asarray(samples), np.concatenate([np.asarray(line.coords) for line in lines])


def terminal_shape(points, triangles, level, center, width):
    p, triangles = np.asarray(points, dtype=float), np.asarray(triangles, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('missing finite measured cloth points')
    target_vertices = target_outline(level, center, width)
    target = Polygon(target_vertices)
    faces = polygons(p[triangles, :2])
    faces = [face for face in faces if face.area > 0.]
    actual = unary_union(faces)
    if actual.is_empty or actual.area <= 0.:
        return dict(shape_success=False, checks=dict(nonzero_projected_area=False),
                    metrics=dict(projected_area_m2=0.), fixed_target_vertices_xy=target_vertices.tolist())
    # GEOS overlay may leave zero-area roundoff loops at shared vertices.
    # Discard only loops below machine-epsilon * area, not physical tolerances.
    components = list(actual.geoms) if hasattr(actual, 'geoms') else [actual]
    contours = []
    for component in components:
        contours.append(component.exterior)
        contours.extend(ring for ring in component.interiors
                        if Polygon(ring).area > np.finfo(float).eps * actual.area)
    boundary, reference = MultiLineString(contours), target.boundary
    a, actual_vertices = boundary_samples(boundary)
    b, reference_vertices = boundary_samples(reference)
    errors = np.r_[distance(a, reference), distance(b, boundary)]
    median, p90, maximum = (float(np.median(errors)), float(np.quantile(errors, .9)), float(max(errors.max(), distance(shapely_points(actual_vertices), reference).max(),
                                           distance(shapely_points(reference_vertices), boundary).max())))
    # The final rule explicitly expands only the area-test region by 10mm.
    # Outline errors above remain measured against the original fixed target.
    expanded = target.buffer(.010, quad_segs=128)
    inside = float(actual.intersection(expanded).area / actual.area)
    checks = dict(contour_median=median < .015, contour_p90=p90 < .025,
                  projected_area_in_expanded_target=inside >= .98)
    if level == 'hard':
        checks['maximum_contour_error'] = maximum < .010
    return dict(shape_success=all(checks.values()), checks=checks,
                metrics=dict(median_contour_error_m=median, p90_contour_error_m=p90,
                             maximum_contour_error_m=maximum,
                             in_target_area_fraction=float(actual.intersection(target).area/actual.area),
                             in_expanded_target_area_fraction=inside, target_expansion_m=.010,
                             projected_area_m2=float(actual.area)),
                fixed_target_vertices_xy=target_vertices.tolist(),
                scope='T0 measured triangle union in fixed table target; no pose fitting')
