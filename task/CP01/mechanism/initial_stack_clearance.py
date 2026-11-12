"""Resolve the retained stacked-tray/conveyor version mismatch once, before contacts."""
def stack_clearance(stage,setup):
    if setup['difficulty'] not in ('medium','hard'):
        return dict(enabled=False,offset_m=[0.,0.,0.],scope='Nominal source initialization unchanged')
    from pxr import Usd,UsdGeom
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    bounds={}
    for path in ['/World/CP01/Conveyor/Chassis','/World/CP01/Conveyor/Rail0']:
        prim=stage.GetPrimAtPath(path)
        if not prim:raise ValueError('Missing original fixed fixture for initialization clearance: '+path)
        box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        bounds[path]=dict(min=list(box.GetMin()),max=list(box.GetMax()))
    front=min(row['min'][1] for row in bounds.values())
    rear=max(p[1]+setup['tray_spec']['tray_size_m'][1]/2 for p in setup['tray_initial_positions_m'])
    shift=min(0.,front-.003-rear)
    if abs(shift)>.02:raise ValueError('Initial geometry needs more than the documented small clearance adapter')
    return dict(enabled=True,offset_m=[0.,float(shift),0.],source_initial_positions=setup['tray_initial_positions_m'],
        fixture_bounds=bounds,clearance_m=.003,
        reason='Retained nominal stack intersects current conveyor chassis/rail; translate entire stack equally before first contact enable',
        scope='One initialization only; no observation/rollout pose writes; original IDs/seeds/source files/goals/physics/scoring unchanged')
