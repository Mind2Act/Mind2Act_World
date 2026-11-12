"""Synthetic M1 instance rows: baseline shape (stations, carriers, covers, swaps), no real ids, hashes or paths."""
import copy


def route_of(target, swaps):
    holder = {s: s for s in {x for pair in swaps for x in pair} | {target}}  # station -> carrier id
    at, path = target, [target]
    for a, b in swaps:
        holder[a], holder[b] = holder[b], holder[a]
        now = next(s for s, c in holder.items() if c == target)
        if now != at:
            path.append(now)
            at = now
    return path


def row(difficulty='easy', stations='ABCD', target='D', swaps=(('D', 'B'), ('B', 'C'), ('C', 'A')), spacing=.22):
    n = len(stations)
    xs = {s: (i - (n - 1) / 2) * spacing for i, s in enumerate(stations)}
    bodies, carriers = {}, {}
    for s in stations:
        x = xs[s]
        for kind, z, dims, mass in (('carrier', .648, [.125, .125, .014], .25), ('cover', .662, [.125, .125, .06], .1),
                                    ('workpiece', .6845, [.06, .06, .03], .05)):
            ident = '%s_%s' % (kind, s)
            bodies[ident] = dict(prim_path='/World/Synthetic_' + ident, kind=kind, mass_kg=mass, dimensions=dims,
                                 center_offset_local=[0, 0, dims[2] / 2], initial_pose_world=[x, -.014, z, 1., 0., 0., 0.],
                                 support_prim='/World/Synthetic_%s/Collision' % ident)
        carriers[s] = dict(initial_station=s, carrier_body='carrier_' + s, cover_body='cover_' + s, workpiece_body='workpiece_' + s,
                           cover_closed_pose_world=[x, -.014, .662, 1., 0., 0., 0.])
    expected = route_of(target, [list(p) for p in swaps])
    manifest = dict(
        case=dict(difficulty=difficulty.capitalize(), swaps=[list(p) for p in swaps], expected_path=expected,
                  initial_target_station=target, exchange_transport=dict(style='compact_arcs'), stable_seconds=.5),
        bodies=bodies, carriers=carriers, stations={s: dict(center_xy=[xs[s], -.014]) for s in stations},
        submission_stations={s: dict(center_xy=[xs[s] + .06, -.265], pad_prim='/World/Synthetic_Station' + s, support_top_z=.652,
                                     clear_dimensions_xy=[.145, .09]) for s in stations},
        neutral=dict(center_xy=[0, -.19], support_top_z=.648, pad_prim='/World/Synthetic_Table', clear_dimensions_xy=[.08, .08]),
        cover_park=dict(center_xy=[0, .3], position_tolerance_m=.05, support_prim_paths=['/World/Synthetic_Park']),
        target_body_id='workpiece_' + target, target_cover_id='cover_' + target, robot_home_joint_degrees=[0, 65, 100, -65, 0, 0],
        environment_transport=dict(clearing_required_for_submission=False), cover_transport_lift_m=.23,
        static_collision_roots=['/World/Synthetic_Static'],
        station=dict(robots=[dict(name='Left', position=[-.3, -.45, .765], euler_deg=[0, 0, 90]),
                             dict(name='Right', position=[.3, -.45, .765], euler_deg=[0, 0, 90])]), replay_ready_prim='/World/Synthetic_Ready')
    return copy.deepcopy(dict(
        task_id='M', difficulty=difficulty, case_id='M_%s_synthetic' % difficulty, split='train',
        instruction='Track the target, uncover it, and move it through the stations in order.',
        binding=dict(expected_units=len(expected)),
        config=dict(source_job=dict(id='M_%s_synthetic' % difficulty), source_manifest_case=copy.deepcopy(manifest['case']),
                    native_inputs=dict(manifest=manifest))))


def hard_row():
    return row('hard', stations='ABCDEF', target='B', spacing=.12,
               swaps=(('B', 'E'), ('A', 'C'), ('E', 'A'), ('D', 'E'), ('A', 'B'), ('C', 'F'), ('E', 'D')))
