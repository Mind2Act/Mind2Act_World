"""Millimetre-space PR3 target path. Ported unchanged from the baseline (source_id
pr3-original-source-mainline-pro-sourcepath: cases_unified/PR3/source/mind2act_pr3_marker_paths.py) plus
`reference` from cases_unified/PR3/geometry.py, without scipy."""
import math
import numpy as np


def make_path(level):
    bends, ratio = {'easy': (0, 0.), 'medium': (2, .14), 'hard': (4, .075)}[level]
    t = np.linspace(0, 1, 2401)
    points = np.column_stack((t-.5, ratio*np.sin(bends*math.pi*t)))
    raw_length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    span = 350.0/raw_length
    points *= span
    points[:, 1] -= (points[:, 1].min()+points[:, 1].max())/2
    if level == "easy":
        angle = math.radians(-30)
        points = points @ np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    stats = {
        'bends': bends,
        'length_mm': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        'horizontal_span_mm': float(np.ptp(points[:, 0])),
        'vertical_span_mm': float(np.ptp(points[:, 1])),
        'minimum_radius_mm': None if bends == 0 else span/(ratio*(bends*math.pi)**2),
        'maximum_curvature_per_mm': 0.0 if bends == 0 else ratio*(bends*math.pi)**2/span,
        'total_absolute_turn_degrees': math.degrees(2*bends*math.atan(ratio*bends*math.pi)),
        'centerline_width_mm': 2.5,
        'tolerance_half_width_mm': 3.0,
        'curve_family': ('straight line: 350 mm at -30 degrees in the paper plane' if bends == 0 else
                         'x=D(t-0.5), y=A*sin(bends*pi*t), centered by bounding box'),
    }
    assert abs(stats['length_mm']-350) < .01
    assert np.max(np.abs(points[:, 0])) < 190
    assert np.max(np.abs(points[:, 1])) < 105
    return points, stats


def reference(parameters):
    """Target path in board millimetres for one instance (level, mirror, angle_delta_deg, offset_xy_mm)."""
    points, _ = make_path(parameters['level'])
    if parameters['mirror']:
        if parameters['level'] == 'easy':
            raise ValueError('source Easy path cannot be mirrored')
        points[:, 1] *= -1
    a = math.radians(parameters['angle_delta_deg'])
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return points @ rotation.T + parameters['offset_xy_mm']
