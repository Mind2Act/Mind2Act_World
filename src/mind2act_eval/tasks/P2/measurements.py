"""Raw poses and signed contact manifolds; no expert/controller annotations."""
import math


def vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError('P2 missing '+label)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError('P2 nonfinite '+label)
    return value


def owner(path):
    return path.split('/')[3] if path.startswith('/World/Task/') else path


def tilt(quaternion):
    w,x,y,z = vector(quaternion,4,'body quaternion')
    norm2 = w*w+x*x+y*y+z*z
    if norm2 == 0:
        raise ValueError('P2 zero body quaternion')
    return math.degrees(math.acos(max(-1.,min(1.,1-2*(x*x+y*y)/norm2))))


def validate_frame(row, objects):
    if type(row['physics_tick']) is not int or row['physics_tick'] < 0 or not math.isfinite(row['simulation_time_s']):
        raise ValueError('P2 invalid physical clock')
    if set(row['poses']) != set(objects):
        raise ValueError('P2 measured object identity mismatch')
    for pose in row['poses'].values():
        vector(pose['position'],3,'body position')
        tilt(pose['quaternion_wxyz'])
        vector(pose['linear_velocity_m_s'],3,'body linear velocity')
        vector(pose['angular_velocity_rad_s'],3,'body angular velocity')
    vector(row['preplaced_xyz'],3,'preplaced base position')
    if set(row['q']) != {'Left','Right'} or set(row['qd']) != {'Left','Right'}:
        raise ValueError('P2 missing native joint state')
    for side in ('Left','Right'):
        vector(row['q'][side],8,'native joint position')
        vector(row['qd'][side],8,'native joint velocity')
    for c in row['contacts']:
        if (type(c['active']) is not bool or len(c['pair']) != 2
                or any(type(p) is not str for p in c['pair']) or not isinstance(c['points'],list)):
            raise ValueError('P2 invalid native contact manifold')
        for p in c['points']:
            vector(p['position'],3,'contact position')
            vector(p['normal'],3,'signed contact normal')
            vector(p['impulse'],3,'contact impulse')
            if not math.isfinite(p['separation']):
                raise ValueError('P2 nonfinite contact separation')


def active(contact):
    return contact['active'] and any(math.dist(p['impulse'],[0.,0.,0.]) > 0.
                                     or p['separation'] <= 0 for p in contact['points'])


def robot_paths(row, name):
    return [p for c in row['contacts'] if active(c) and any(owner(p)==name for p in c['pair'])
            for p in c['pair'] if p.startswith('/World/Robot')]


def bilateral(row, name):
    paths = robot_paths(row,name)
    return any(all(any('/Robot'+side+'/' in p and '/'+link+'/' in p for p in paths)
                   for link in ('link7','link8')) for side in ('Left','Right'))


def supports(row, name):
    result = []
    for c in row['contacts']:
        if not active(c):
            continue
        for own,other,sign in ((c['pair'][0],c['pair'][1],1),(c['pair'][1],c['pair'][0],-1)):
            if owner(own)==name and not other.startswith('/World/Robot'):
                if any(sign*p['normal'][2] > 0. for p in c['points']):
                    result.append(owner(other))
    return result
