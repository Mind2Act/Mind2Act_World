"""CP06 independent terminal units and physical ZERO, FINAL-VERSION.md."""
import math
import numpy as np
from mind2act_eval.scoring import PressEdge, SCORING_VERSION


def object_placed(path, side, config, raw, initial=None):
    if not raw:
        return False
    obj=raw[-1]['objects'][path]
    center=np.asarray(obj['pan_center_local_m'][side],float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise ValueError('CP06 measured object center in pan frame required')
    return bool(obj['pan_supported'][side] and np.linalg.norm(center[:2])<=config['pan_inner_radius_m'])


def terminal_units(config, state, zero, dropped=()):
    fruits=[o['path'] for o in config['all_objects'] if o['kind']=='fruit']
    weights=[o['path'] for o in config['all_objects'] if o['kind']=='weight']
    pf=[p for p in fruits if p not in dropped and object_placed(p,'Left',config,[state])]
    pw=[p for p in weights if p not in dropped and object_placed(p,'Right',config,[state])]
    steps=([bool(zero)] if config['difficulty']!='easy' else [])
    steps += [p in pf for p in fruits]+[i<len(pw) for i in range(2)]
    if config['difficulty']=='hard': steps.append(abs(state['rider_m']-.09)<=.003)
    loaded=len(pf)==len(fruits) and len(pw)>=2
    green=config['green_angle_interval_deg'][0]<=state['corrected_angle_deg']<=config['green_angle_interval_deg'][1]
    steps.append(bool(green))
    return steps,pf,pw


class DropFacts:
    """A proven free fall below an actually occupied pan, never a contact gap."""
    def __init__(self,config,initial):
        self.config=config;self.seen={p:set() for p in initial['objects']}
        self.dropped=set();self.events=[]
        self.append(initial,0)

    def append(self,state,tick):
        for path,obj in state['objects'].items():
            for side,on in obj['pan_supported'].items():
                if on:self.seen[path].add(side)
            if path in self.dropped or not self.seen[path] or obj['robot_contact'] or obj['external_supported']:
                continue
            if any(obj['pan_supported'].values()) or obj['linear_velocity_m_s'][2]>=0:
                continue
            for side in self.seen[path]:
                if obj['pan_center_local_m'][side][2]<state['pan_bottom_local_m'][side]:
                    self.dropped.add(path)
                    self.events.append(dict(path=path,tick=tick,pan=side,event='proven_fall_off_pan'))
                    break


def validate(config,ticks,initial):
    expected={o['path'] for o in config['all_objects']}
    if set(initial['objects'])!=expected:
        raise ValueError('CP06 complete initial object coverage required')
    for obj in initial['objects'].values():
        p=np.asarray(obj['position'],float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('CP06 nonfinite initial position')
    previous=initial['simulation_time_s']
    for index,row in enumerate(ticks,1):
        state=row['raw'];now=row['simulation_time_s']
        if (row['physics_tick']!=index or state['simulation_time_s']!=now
                or not math.isfinite(now) or abs(now-previous-1/120)>1e-7
                or set(state['objects'])!=expected):raise ValueError('CP06 complete 120 Hz physical clock required')
        previous=now
        if not np.isfinite([state['beam_deg'],state['corrected_angle_deg'],state['rider_m']]).all():
            raise ValueError('CP06 nonfinite physical measurement')
        if set(state['pan_bottom_local_m'])!={'Left','Right'} or not np.isfinite(list(state['pan_bottom_local_m'].values())).all():
            raise ValueError('CP06 actual pan bottom measurements required')
        for obj in state['objects'].values():
            p=np.asarray(obj['position'],float)
            if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('CP06 nonfinite object position')
            if type(obj['external_supported']) is not bool or type(obj['robot_contact']) is not bool or np.asarray(obj['linear_velocity_m_s']).shape!=(3,) or not np.isfinite(obj['linear_velocity_m_s']).all():
                raise ValueError('CP06 actual free-fall contact measurements required')
            for side in ('Left','Right'):
                center=np.asarray(obj['pan_center_local_m'][side],float)
                if center.shape!=(3,) or not np.isfinite(center).all() or type(obj['pan_supported'][side]) is not bool:
                    raise ValueError('CP06 actual support and pan center measurements required')


def evaluate(config,ticks,initial):
    validate(config,ticks,initial)
    level=config['difficulty'];n={'easy':4,'medium':6,'hard':8}[level]
    zero=level=='easy';zero_tick=None;edge=PressEdge(.004);drops=DropFacts(config,initial)
    for row in ticks:
        state=row['raw'];drops.append(state,row['physics_tick'])
        if level!='easy' and edge.update(state['control']['button_travel_m']):
            empty=not any(any(o['pan_supported'].values()) for o in state['objects'].values())
            if empty:zero=True;zero_tick=row['physics_tick']
    if not ticks:
        return dict(success=False,k=0,n=n,first_error=None,cutoff_tick=0,steps=[False]*n,
                    placed_fruits=[],placed_weights=[],physical_zero=zero,scoring_version=SCORING_VERSION)
    steps,pf,pw=terminal_units(config,ticks[-1]['raw'],zero,drops.dropped)
    if len(steps)!=n:raise ValueError('CP06 authority denominator differs')
    k=sum(steps)
    # Required units may all be correct after an extra weight fell. The explicit
    # no-drop rule still prevents100; it does not erase independent partial units.
    return dict(success=all(steps) and not drops.dropped,k=k,n=n,steps=steps,first_error=None,
                cutoff_tick=ticks[-1]['physics_tick'],placed_fruits=pf,placed_weights=pw,
                physical_zero=zero,zero_tick=zero_tick,dropped_objects=sorted(drops.dropped),drop_events=drops.events,scoring_version=SCORING_VERSION)
