"""Versioned CP10 inbound shot mix, independent of the hidden ball colors."""
import math
import random

from .arena_preparation import launch_pose


def shot_kinds(config):
    spec=config['arena']['rebound_series']
    formal=['sidewall']*spec['formal_sidewall_shots']+['direct']*(spec['formal_shots']-spec['formal_sidewall_shots'])
    random.Random(f"cp10-rebound-order-v1/{config['difficulty']}/{config['seed']}").shuffle(formal)
    return ['direct','sidewall']+formal


def sample_rebound_heading(config,shot_index):
    kind=shot_kinds(config)[shot_index]
    spec=config['arena']['heading_sampling']
    rng=random.Random(f"{spec['stream']}/{config['difficulty']}/{config['seed']}/{shot_index}")
    magnitude=rng.uniform(*spec[kind+'_magnitude_deg'])
    return round(magnitude*(-1 if rng.random()<.5 else 1),6)


def inbound_geometry(config,yaw_deg,slide_x_m=0.):
    """Unfold the fixed parallel walls up to the near paddle's front face."""
    a=config['arena'];origin,direction=launch_pose(a,yaw_deg,slide_x_m)
    plane=config['objects']['paddle']['position'][1]+config['objects']['paddle']['size'][1]/2+a['ball_diameter_m']/2
    half=a['court_size'][0]/2-a['ball_diameter_m']/2
    raw=origin[0]+(plane-origin[1])*direction[0]/direction[1]
    crossings=math.floor((abs(raw)+half)/(2*half))
    t=(raw+half)%(4*half)
    return dict(sidewall_bounces=crossings,intercept_x_m=-half+t if t<=2*half else 3*half-t)
