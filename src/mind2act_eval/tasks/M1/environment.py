"""Environment choreography at 120 Hz: initial hold, cover lift/align/lower, carrier exchanges.

Ported from source `cases_unified/M/environment.py` (same frames; timings come from config/layout.json and
the case). Pure numpy: returns the per-tick pose writes the scene applies to the kinematic bodies, the swap
arrivals, final occupants and final poses. The target is routed only by these environment writes until the
physical transition; the robot never sees or controls this schedule.
"""
import numpy as np

from .exchange_hard import exchange_paths as hard_paths
from .exchange_legacy import exchange_paths as legacy_paths


def compile_environment(M, lay):
    C, carriers = M['case'], M['carriers']
    poses = {k: np.asarray(s['initial_pose_world'], float).copy() for k, s in M['bodies'].items()}
    frames, arrivals = [], []
    occupants = {v['initial_station']: k for k, v in carriers.items()}
    target = next(k for k, v in carriers.items() if v['workpiece_body'] == M['target_body_id'])
    route = [C['initial_target_station']]

    def hold(label, n):
        frames.extend((label, None, {}) for _ in range(n))

    def move(destinations, label, n, phase):
        starts = {k: poses[k].copy() for k in destinations}
        for i in range(n):
            u = (i + 1) / n
            e = u * u * (3 - 2 * u)
            writes = {}
            for k, dest in destinations.items():
                if not np.allclose(starts[k][3:], dest[3:]):
                    raise ValueError('environment move may not rotate bodies')
                poses[k] = np.r_[starts[k][:3] + (dest[:3] - starts[k][:3]) * e, starts[k][3:]]
                writes[k] = poses[k].copy()
            frames.append((label, phase, writes))

    hold('initial_target', lay['initial_hold_ticks'])
    n = round(float(C.get('cover_phase_seconds', lay['default_cover_phase_s'])) * 120)
    if n <= 0:
        raise ValueError('cover phase must be positive')
    raised = {v['cover_body']: poses[v['cover_body']].copy() for v in carriers.values()}
    for p in raised.values():
        p[2] += float(M['cover_transport_lift_m'])
    move(raised, 'environment_cover_lift', n, 'environment_close')
    aligned = {v['cover_body']: np.r_[v['cover_closed_pose_world'][:2], raised[v['cover_body']][2], raised[v['cover_body']][3:]]
               for v in carriers.values()}
    move(aligned, 'environment_cover_align', n, 'environment_close')
    move({v['cover_body']: np.asarray(v['cover_closed_pose_world']) for v in carriers.values()},
         'environment_cover_lower', n, 'environment_close')
    hold('covered_transport', lay['covered_transport_ticks'])
    for index, (sa, sb) in enumerate(C['swaps']):
        ids = [occupants[sa], occupants[sb]]
        starts = {ident: poses[ident].copy() for cid in ids
                  for ident in (carriers[cid]['carrier_body'], carriers[cid]['cover_body'], carriers[cid]['workpiece_body'])}
        transport = C['exchange_transport']
        if transport['style'] != 'compact_arcs':
            raise ValueError('unsupported exchange style')
        paths_for = hard_paths if C['difficulty'].lower() == 'hard' else legacy_paths
        trajectories = paths_for({k: v['center_xy'] for k, v in M['stations'].items()}, (sa, sb), fps=120,
                                 speed=float(transport.get('max_translation_speed_m_s', lay['default_swap_speed_m_s'])))
        carrier_at_start = {sa: ids[0], sb: ids[1]}
        for trajectory in trajectories:
            paths = trajectory['positions']
            for i in range(1, len(next(iter(paths.values())))):
                writes = {}
                for origin, path in paths.items():
                    cid = carrier_at_start[origin]
                    delta = np.r_[path[i] - np.asarray(M['stations'][origin]['center_xy']), 0.]
                    for ident in (carriers[cid]['carrier_body'], carriers[cid]['cover_body'], carriers[cid]['workpiece_body']):
                        poses[ident] = np.r_[starts[ident][:3] + delta, starts[ident][3:]]
                        writes[ident] = poses[ident].copy()
                frames.append(('environment_swap_' + str(index), 'environment_swap', writes))
        occupants[sa], occupants[sb] = ids[1], ids[0]
        station = next(s for s, c in occupants.items() if c == target)
        if target in ids:
            route.append(station)
        arrivals.append(dict(step=len(frames), swap_index=index, target_station=station, target_moved=target in ids))
        hold('covered_swap_stop_' + str(index), round(120 * float(transport.get('arrival_hold_seconds', lay['default_arrival_hold_s']))))
    if route != C['expected_path']:
        raise ValueError('environment route differs from the instance expected path')
    return frames, arrivals, occupants, poses
