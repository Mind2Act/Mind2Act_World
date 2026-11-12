"""Synthetic DR1 instance row: same shape as a baseline spec, no real ids, hashes or paths."""
import copy

TIERS = {'easy': 3, 'medium': 5, 'hard': 7}


def row(difficulty='easy', *, case_id=None):
    n = TIERS[difficulty]
    items = [dict(index=i, label='P%d' % i, identity='parcel_%d' % i, arm='Left' if i % 2 == 0 else 'Right',
                  release_slot_xy=[-.27 + .09 * i, -.215], slot_bounds_m=[-.31 + .09 * i, -.23 + .09 * i, -.267, -.163],
                  closed_joint_target_m=.0137, spawn_center=[-.6 - .36 * i, -.035, .88], spawn_yaw_deg=-4.)
             for i in range(n)]
    goods = [dict(index=i, label='P%d' % i, identity='parcel_%d' % i, size_m=[.05, .06, .05], path='/World/Synthetic/Goods_%02d' % i)
             for i in range(n)]
    return copy.deepcopy(dict(
        task_id='FR1', difficulty=difficulty, case_id=case_id or 'FR1/%s/synthetic_0001' % difficulty, split='train',
        instruction='Pick up all %d packages and place them in the tray.' % n,
        config=dict(
            parameters=dict(count=n, speed_m_s=.08),
            source_recipe=dict(items=items, spawn_tick=0, source_centers=[.88] * n, source_budget_s=130.),
            native_inputs=dict(
                manifest=dict(layout=dict(belt_bounds_m=[-2.2, 2.2, -.105, .035, .842, .858],
                                          inspection_tray_bounds_m=[-.34, .34, -.275, -.155, .768, .784]),
                              stream_goods=[dict(g) for g in goods]),
                geometry_manifest=dict(stream_goods=goods), scene_sha256='0' * 64,
                station=dict(robots=[dict(name='Left', position=[-.3, -.45, .765], euler_deg=[0, 0, 90]),
                                     dict(name='Right', position=[.3, -.45, .765], euler_deg=[0, 0, 90])])),
            physics_budget=15601)))
