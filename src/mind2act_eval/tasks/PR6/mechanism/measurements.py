"""Task evidence from recorded native poses/contact manifolds only."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .evidence import vector, EvidenceError


def pose(frame,name):
    row=frame['bodies'][name]
    q=np.asarray(vector(row['orientation_wxyz'],4,'orientation'))
    if abs(np.linalg.norm(q)-1)>.001:raise EvidenceError('Non-unit measured quaternion')
    return np.asarray(vector(row['position'],3,'position')),Rotation.from_quat(q[[1,2,3,0]])


def angle(frame,name,axis):
    _,r=pose(frame,name)
    return float(r.as_rotvec()[axis]*180/math.pi)


def robot_touch(frame,name):
    prefix='/World/Task/Objects/'+name
    return any(e['active'] and e.get('contact_count',0)>0 and e.get('min_separation_m',1)<=.0001
        and any(a.startswith(prefix+'/') and b.startswith('/World/Robot')
            for a,b in (e['pair'],list(reversed(e['pair'])))) for e in frame.get('contact_events',[]))


def task_contact(frame,first,second):
    a='/World/Task/Objects/'+first;b='/World/Task/Objects/'+second
    return any(e['active'] and e.get('contact_count',0)>0 and e.get('min_separation_m',1)<=.0001
        and any(x.startswith(a+'/') and y.startswith(b+'/')
            for x,y in (e['pair'],list(reversed(e['pair'])))) for e in frame.get('contact_events',[]))


def supported_by(body,path,tolerance=.004):
    return any(p.path==path and p.normal[2]>=.5 and abs(p.separation)<=tolerance for p in body.supports or ())
