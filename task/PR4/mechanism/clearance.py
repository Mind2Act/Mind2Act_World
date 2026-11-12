"""Original PR4 clearance from actual finger body poses; motion planners excluded."""
import numpy as np


def observed_cloth_clearance(row,points,hulls,sides):
    """Read actual finger body transforms, independent of commanded openings."""
    from scipy.spatial.transform import Rotation
    result={}
    for side in sides:
        for j in (7,8):
            key=side+'/link'+str(j);body=row['finger_contact_envelopes'][key];q=body['quaternion']
            rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix();local=(np.asarray(points)-body['position'])@rotation
            result[key]=float(np.min([np.max(local@np.asarray(h)[:,:3].T+np.asarray(h)[:,3],axis=1) for h in hulls[key]]))
    return result
