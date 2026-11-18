"""Original sphere/tray containment; unchanged geometry and tolerances."""
import numpy as np


def containment(local,polygon,radius,floor,rim):
    p=np.asarray(polygon);e=np.roll(p,-1,axis=0)-p;norm=np.linalg.norm(e,axis=1)
    # Determine inward side independently of polygon winding.
    center=p.mean(0);orient=np.sign(np.mean(e[:,0]*(center[1]-p[:,1])-e[:,1]*(center[0]-p[:,0])))
    xy=orient*(e[:,0]*(local[1]-p[:,1])-e[:,1]*(local[0]-p[:,0]))/norm
    margin=float(xy.min()-radius)
    inside=margin>=-.002 and floor-.004<=local[2]<=rim+radius+.025
    return margin,bool(inside)
