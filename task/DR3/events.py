"""Ordered item aperture passage and actual T0 delivery; FINAL-VERSION.md.

Robot pose, grip fraction, release, withdrawal and later door closure are not
score conditions. Missing native physical evidence remains unknown, never zero.
"""
import math
import numpy as np
from mind2act_eval.scoring import SCORING_VERSION
from .geometry import (finite_array, transform_vertices, swept_aperture, side,
                       measured_physics_transition, MEASUREMENT_CONTRACT)

CONTRACT = SCORING_VERSION


def event(status='NOT_OCCURRED', time=None, **fields):
    return dict(status=status, time_s=time, **fields)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite detector measurement required')
    return float(value)


def _contact_state(frame, manifest):
    root = manifest['item_root']
    support, robot = False, False
    for row in frame['contacts']:
        pair = row['pair']
        if len(pair) != 2 or type(row['active']) is not bool:
            raise ValueError('invalid native contact row')
        _number(row['impulse_ns'])  # Measurement validity, never a support threshold.
        if not row['active'] or not any(p == root or p.startswith(root+'/') for p in pair):
            continue
        robot |= any(p.startswith('/World/Robot') for p in pair)
        if any(p in manifest['tray_colliders'] for p in pair):
            normals = finite_array(row['item_upward_normals'])
            if not len(normals):
                raise ValueError('tray contact lacks measured support normal')
            support |= bool(np.any(normals > 0.))
    return dict(tray_support=bool(support), robot_contact=bool(robot))


def _center_in_tray(center_xy, tray):
    """Actual floor triangle union, including rounded edges and real holes."""
    if 'floor_triangles_xy' not in tray:
        raise ValueError('actual tray cavity geometry is missing')
    triangles = finite_array(tray['floor_triangles_xy'])
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2) or not len(triangles):
        raise ValueError('actual tray floor triangles are missing')
    edges = np.roll(triangles, -1, axis=1)-triangles
    offset = np.asarray(center_xy)-triangles
    cross = edges[..., 0]*offset[..., 1]-edges[..., 1]*offset[..., 0]
    area = (triangles[:, 1, 0]-triangles[:, 0, 0])*(triangles[:, 2, 1]-triangles[:, 0, 1]) \
        - (triangles[:, 1, 1]-triangles[:, 0, 1])*(triangles[:, 2, 0]-triangles[:, 0, 0])
    return bool(np.any((area != 0) & (np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1))))


def rebuild_events(rows, manifest, *, allow_test_certificates=False):
    if not rows:
        raise ValueError('missing native item trace')
    bodies, gates = manifest['bodies'], manifest['gates']
    items = [name for name, body in bodies.items() if body['role'] == 'item']
    if len(items) != 1 or len(gates) not in (1, 2):
        raise ValueError('one item and one/two ordered gates required')
    item_id, unknown, violation = items[0], [], False
    item = bodies[item_id]
    measured = (manifest.get('measurement_contract') == MEASUREMENT_CONTRACT or
        (manifest.get('schema') == 'dr3-collision-pose-evidence-v1' and
         manifest.get('motion_contract') == 'per-physics-step-origin-segment-full-rotation-envelope-v1'))
    coverage = manifest['coverage']
    if coverage['missing_colliders'] or set(coverage['expected_colliders']) != set(coverage['represented_colliders']):
        unknown.append('collision geometry coverage is incomplete')
    if manifest.get('tray_region_qualified', True) is not True:
        unknown.append('target tray cavity geometry is not qualified')
    if manifest.get('schema') == 'dr3-collision-pose-evidence-v2' and not measured:
        unknown.append('unsupported measured physics contract')
    dt = _number(manifest['physics_dt_s'])
    if measured and (abs(dt-1/120) > 1e-12 or any(
            body.get('vertices_are_actual_collision_points') is not True for body in bodies.values())):
        unknown.append('missing actual item collision points or 120Hz native clock')
    entries, passed = [event() for _ in gates], [event() for _ in gates]
    states, times, attempts, active = [], [], [], None
    for i, row in enumerate(rows):
        t = _number(row['simulation_time_s'])
        if type(row['physics_tick']) is not int or row['physics_tick'] != i:
            unknown.append('nonconsecutive detector physics ticks')
        if times and abs(t-times[-1]-dt) > 1e-7:
            unknown.append('detector simulation clock gap')
        times.append(t)
        frame = row['detector']
        if set(frame['poses']) != set(bodies) or set(frame['gates']) != {gate['id'] for gate in gates}:
            raise ValueError('missing native collision body pose or gate measurement')
        for name, body in bodies.items():
            transform_vertices(body['vertices'], frame['poses'][name])
        for position in frame['gates'].values():
            _number(position)
        if frame.get('contacts_complete') is not True:
            unknown.append('contact report coverage not qualified')
        if measured:
            receipt = frame.get('sampling', {})
            if (type(receipt.get('tick')) is not int or type(receipt.get('before_step_epoch')) is not int
                    or receipt.get('tick') != i or receipt.get('before_step_epoch') != i-1
                    or receipt.get('subscription_active') is not True or receipt.get('errors') != []):
                unknown.append('missing native physics substep receipt')
        states.append(_contact_state(frame, manifest))
        if not i:
            continue
        previous = rows[i-1]['detector']
        before, after = previous['poses'][item_id], frame['poses'][item_id]
        forward = after['position'][0] > before['position'][0]
        for j, gate in enumerate(gates):
            if measured:
                motion = measured_physics_transition(item['vertices'], before, after, gate,
                    actual_vertices=True, collision_parts=item.get('collision_parts'))
            else:
                certificates = frame.get('sweeps', {}).get(item_id, {})
                certificate = certificates if 'contract' in certificates else certificates.get(gate['id'])
                motion = swept_aperture(item['vertices'], before, after, gate, certificate,
                                       allow_test_certificates=allow_test_certificates)
            if motion['status'] == 'UNKNOWN':
                unknown.append('unresolved item aperture passage at '+gate['id'])
            # Gate occupancy must be legal in either travel direction (also
            # while the item turns or pauses in the slab). Only progress/entry
            # ordering below is restricted to forward travel.
            if motion.get('intersects'):
                opened = all(_number(f['gates'][gate['id']]) >=
                    gate['fully_open_position_m']-gate['open_tolerance_m'] for f in (previous, frame))
                if not opened or motion['status'] == 'VIOLATION':
                    violation = True
            a = side(item['vertices'], before, gate, gate['half_thickness_m'])
            b = side(item['vertices'], after, gate, gate['half_thickness_m'])
            begins = forward and a == 'OUTSIDE' and b != 'OUTSIDE'
            if begins:
                if j == 0:
                    active = dict(attempt_id=len(attempts), entry_s=[None]*len(gates),
                                  pass_s=[None]*len(gates), passed_gate_count=0, legal=True)
                    attempts.append(active)
                # Overlapping gate occupancy is legal: earlier ENTRY suffices.
                if active is None or j and active['entry_s'][j-1] is None:
                    violation = True
                    continue
                active['entry_s'][j] = t
                entries[j] = event('OCCURRED', t, attempt_id=active['attempt_id'])
            if active and active['entry_s'][j] is not None and b == 'INSIDE' and active['pass_s'][j] is None:
                active['pass_s'][j] = t
                active['passed_gate_count'] += 1
                passed[j] = event('OCCURRED', t, attempt_id=active['attempt_id'])
    valid_pass = not violation and any(a['passed_gate_count'] == len(gates) for a in attempts)
    tray = manifest['tray']
    vertices = transform_vertices(item['vertices'], rows[-1]['detector']['poses'][item_id])
    center = (vertices.min(axis=0) + vertices.max(axis=0))/2 - finite_array(tray['origin'], (3,))
    try:
        contained = _center_in_tray(center[:2], tray)
    except ValueError as exc:
        unknown.append(str(exc)); contained = False
    supported = states[-1]['tray_support']
    delivered = bool(contained and supported)
    unknown = list(dict.fromkeys(unknown))
    complete = not unknown
    success = bool(valid_pass and delivered)
    value = (100. if success else 5. if valid_pass else 0.) if complete else None
    return dict(evidence_complete=complete, raw_sampling_complete=complete,
        full_success_evidence_complete=complete, unknown_reasons=unknown, unknown_ranges=[],
        event_detector_contract=CONTRACT, measurement_contract=MEASUREMENT_CONTRACT if measured else 'legacy-bounded-envelope',
        intra_step_continuous_trajectory_claimed=False, success=success if complete else None, score=value,
        events=dict(entry=entries, passed=passed), attempts=attempts, windows=[],
        passed_aperture=bool(valid_pass), passed_aperture_count=max((a['passed_gate_count'] for a in attempts), default=0),
        violation_latched=bool(violation), delivered=delivered, terminal_tray_support=supported,
        center_inside_tray=bool(contained), cutoff_simulation_time_s=times[-1], zero_proof=None)
