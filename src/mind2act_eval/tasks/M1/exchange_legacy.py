"""Easy/medium exchange arcs (source: cases_unified/M/source_exchange_legacy.py), unchanged."""
import numpy as np

def smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*(3-2*t)

def exchange_paths(stations, pair, fps=120, speed=.9):
    points={k:np.asarray(v,dtype=float) for k,v in stations.items()}
    left,right=sorted(pair,key=lambda k:points[k][0]);a,b=points[left],points[right]
    between=[k for k,v in points.items() if a[0]<v[0]<b[0]]
    amplitude=.131 if between else (.085 if b[0]-a[0]<.17 else .075)
    def arc(start,end,sign):
        def path(u):
            e=smooth(u)
            # Broader crest clears intermediate carriers without larger excursions.
            bend=1-(2*e-1)**8 if between else np.sin(np.pi*e)
            return start+(end-start)*e[:,None]+np.stack([np.zeros_like(e),sign*amplitude*bend],axis=1)
        return path
    paths={left:arc(a,b,1),right:arc(b,a,-1)}
    u=np.linspace(0,1,4001)
    vmax=max(float(np.linalg.norm(np.diff(f(u),axis=0),axis=1).max()*4000) for f in paths.values())
    n=int(np.ceil(max(1.2,vmax/speed)*fps));t=np.linspace(0,1,n+1)
    return [{'label':'continuous_opposite_arcs','positions':{k:f(t) for k,f in paths.items()},'seconds':n/fps}]
