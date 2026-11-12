"""Versioned mobile shot sampler and physical guide clearance contract."""
import math
import random


def shot_kinds(config):
    s=config['arena']['rebound_series']
    formal=(['single']*s['formal_single_wall_shots']+['double']*s['formal_two_wall_shots']+
            ['direct']*(s['formal_shots']-s['formal_sidewall_shots']))
    random.Random(f"cp10-mobile-order-v1/{config['difficulty']}/{config['seed']}").shuffle(formal)
    return ['direct','single']+formal


def safe_pose(arena,x,yaw):
    l=arena['launcher']
    if not all(math.isfinite(v) for v in (x,yaw)):return False
    if not l['slide_limits_m'][0]<=x<=l['slide_limits_m'][1]:return False
    if not l['yaw_limits_deg'][0]<=yaw<=l['yaw_limits_deg'][1]:return False
    if x*yaw>1e-9:return False  # left position aims right, right position aims left
    t=math.radians(yaw);half=arena['court_size'][0]/2
    # All corners of the actual rectangular head, rather than only the ball,
    # must remain inside the sidewall's inner face throughout the lowered pose.
    for u in (-l['guide_half_width_m'],l['guide_half_width_m']):
        for v in (-l['guide_length_m'],.044):
            xx=x+u*math.cos(t)-v*math.sin(t)
            if abs(xx)>half-.012:return False
    # Emission begins on the supported court, with a full ball-radius margin.
    y=l['base_xy'][1]-l['muzzle_distance_m']*math.cos(t)
    return y+arena['ball_diameter_m']/2<arena['court_center'][1]+arena['court_size'][1]/2


def sample_pose(config,index):
    a=config['arena'];kind=shot_kinds(config)[index]
    rng=random.Random(f"{a['heading_sampling']['stream']}/{config['difficulty']}/{config['seed']}/{index}")
    half=a['court_size'][0]/2-a['ball_diameter_m']/2
    plane=config['objects']['paddle']['position'][1]+.007+a['ball_diameter_m']/2
    distance=a['launcher']['base_xy'][1]-plane
    for _ in range(1000):
        sign=-1 if rng.random()<.5 else 1
        x=-sign*rng.uniform(.10,.20)
        target=rng.uniform(-.20,.20)
        raw=target if kind=='direct' else sign*2*half-target if kind=='single' else sign*4*half+target
        yaw=round(math.degrees(math.atan2(raw-x,distance)),6);x=round(x,6)
        if safe_pose(a,x,yaw):return x,yaw
    raise ValueError('No safe continuous launcher pose for requested rebound kind')
