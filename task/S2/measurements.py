"""Validate raw measurement completeness before the unchanged source rules."""
import math
from .source_rules.body import vector


def validate_frame(config, frame):
    if type(frame['tick']) is not int or frame['tick']<0:
        raise ValueError('invalid S2 physical tick')
    if not math.isfinite(frame['simulation_time_s']):
        raise ValueError('invalid S2 simulation clock')
    vector(frame['stage_drive_target_m'],2,'XY drive target')
    if not isinstance(frame['contact_events'],list):
        raise ValueError('missing current physical contact batch')
    for event in frame['contact_events']:
        if (type(event['active']) is not bool or type(event['contact_count']) is not int
                or event['contact_count']<0 or len(event['points'])!=event['contact_count']
                or len(event['pair'])!=2 or any(type(p) is not str for p in event['pair'])):
            raise ValueError('invalid raw contact batch')
        if not math.isfinite(event['min_separation_m']):
            raise ValueError('invalid raw contact separation')
        actual_min = min((p['separation'] for p in event['points']),default=1.)
        if abs(actual_min-event['min_separation_m'])>1e-9:
            raise ValueError('raw minimum separation differs from contact points')
        for point in event['points']:
            vector(point['position'],3,'contact point')
            vector(point['normal'],3,'contact normal')
            vector(point['impulse'],3,'contact impulse')
            if not math.isfinite(point['separation']):
                raise ValueError('invalid raw contact point separation')
    buttons = frame['direction_buttons']
    if set(buttons)!=set(config['direction_control']['buttons']):
        raise ValueError('Missing directional button measurements')
    for key, spec in config['direction_control']['buttons'].items():
        row = buttons[key]
        p = vector(row['position_m'],3,'cap position')
        quat = vector(row['orientation_wxyz'],4,'cap orientation')
        if not .99 <= math.sqrt(sum(q*q for q in quat)) <= 1.01:
            raise ValueError('invalid cap quaternion')
        vector(row['linear_velocity_m_s'],3,'cap velocity')
        vector(row['angular_velocity_rad_s'],3,'cap angular velocity')
        if not math.isfinite(row['depth_m']) or abs(spec['center'][2]-p[2]-row['depth_m'])>1e-9:
            raise ValueError('cap depth differs from actual measured pose')
        path = '/World/Task/Design/Controls/'+key+'/Cap'
        touch = any(e['active'] and e['contact_count']>0 and e['min_separation_m']<=.0001
            and any(own.startswith(path+'/') and other.startswith('/World/Robot')
                    for own,other in (e['pair'],e['pair'][::-1])) for e in frame['contact_events'])
        if type(row['robot_contact']) is not bool or row['robot_contact']!=touch:
            raise ValueError('button contact differs from current raw manifold')
    for row in frame['bodies'].values():
        vector(row['position'],3,'body position')
        vector(row['orientation_wxyz'],4,'body orientation')
    if set(frame['q'])!={'Left','Right'}:
        raise ValueError('missing native arm state')
    for q in frame['q'].values():
        vector(q,8,'native joint state')
