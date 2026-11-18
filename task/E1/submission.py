"""Submission occupancy for E1 (source_id E-142-e-current-reconstruction-camera-v2).

Verbatim logic from engine physco_eval/e_submission.py; no answer identities or policy feedback.
"""
import numpy as np


def submission_height_bounds(geometry, station):
    """World-space floor top / rim top from the actual transport box inputs."""
    from scipy.spatial.transform import Rotation
    arm = next(r for r in station['robots'] if r['name'] == geometry['robot_arm'])
    rotation = Rotation.from_euler('xyz', arm['euler_deg'], degrees=True)
    boxes = geometry['collision_world']
    def top(box):
        center = np.asarray(arm['position']) + rotation.apply(box['pose'][:3])
        return float(center[2] + box['dims'][2] / 2)
    floor = top(boxes['submit_floor'])
    walls = [top(box) for name, box in boxes.items()
             if name.startswith('submit_') and name != 'submit_floor']
    if not walls or not np.isfinite([floor, *walls]).all() or max(walls) <= floor:
        raise ValueError('Missing or invalid submission tray geometry')
    return floor, max(walls)


def occupied_items(bounds, center_xy, half_xy, z_bounds, contacts, round_index):
    """Count volume intersection, plus objects stacked on a submitted object.

    A held/unsettled object in the tray still occupies it. An object wholly
    above the rim does not, unless connected to a tray occupant by live object
    contact (stacking). Side-wall contact alone is not submission occupancy.
    """
    z_low, z_high = z_bounds
    if not np.isfinite([z_low, z_high]).all() or z_low >= z_high:
        raise ValueError('Invalid submission tray height evidence')
    horizontal = set()
    occupied = set()
    for key, (center, extent) in bounds.items():
        center, extent = np.asarray(center), np.asarray(extent)
        if not np.isfinite(np.r_[center, extent]).all() or np.any(extent <= 0):
            raise ValueError('Invalid submission object bounds')
        if np.all(abs(center[:2] - center_xy) < half_xy + extent[:2] / 2):
            horizontal.add(key)
            if center[2] + extent[2] / 2 >= z_low and center[2] - extent[2] / 2 <= z_high:
                occupied.add(key)
    def owner(path):
        return next((key for key in bounds if path == key or path.startswith(key + '/')), None)
    graph = {key: set() for key in horizontal}
    floor = '/Batch' + str(round_index) + '/Collision/submit_floor'
    for pair, record in contacts.items():
        if not record['active']:
            continue
        owners = [owner(path) for path in pair]
        if any(floor in path for path in pair):
            occupied.update(key for key in owners if key in horizontal)
        a, b = owners
        if a in horizontal and b in horizontal:
            graph[a].add(b)
            graph[b].add(a)
    pending = list(occupied)
    while pending:
        for key in graph[pending.pop()] - occupied:
            occupied.add(key)
            pending.append(key)
    return occupied
