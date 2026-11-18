"""Geometry-based free setup inside a band; there are no hidden target slots."""
import numpy as np
from functools import lru_cache
from .measurements import pose,supported_by

@lru_cache(maxsize=32)
def _segments(points):
    poly=np.asarray(points);a=poly[:-1];delta=np.diff(poly,axis=0)
    length=np.linalg.norm(delta,axis=1)
    return a,delta,length,np.r_[0.,np.cumsum(length)[:-1]]

def project_vectorized(point,c):
    a,d,length,travel=_segments(tuple(tuple(v) for v in c['route_region']['centerline']))
    t=np.clip(np.sum((np.asarray(point)-a)*d,axis=1)/(length*length),0,1)
    distances=np.linalg.norm(np.asarray(point)-(a+t[:,None]*d),axis=1)
    i=int(np.argmin(distances))
    return float(distances[i]),float(travel[i]+t[i]*length[i]),d[i]/length[i]

def project(point,c):
    if c.get('native_four_case'):return project_vectorized(point,c)
    poly=np.asarray(c['route_region']['centerline']);best=None;travel=0.
    for a,b in zip(poly,poly[1:]):
        d=b-a;length=float(np.linalg.norm(d));t=float(np.clip(np.dot(np.asarray(point)-a,d)/(length*length),0,1))
        distance=float(np.linalg.norm(np.asarray(point)-(a+t*d)))
        if best is None or distance<best[0]:best=(distance,travel+t*length,d/length)
        travel+=length
    return best
def standing(task,frame,name):
    c=task.config;b=task.bodies[name];p,r=pose(frame,name);dist,along,tangent=project(p[:2],c)
    corners=[(x,y) for x in (b.lower[0],b.upper[0]) for y in (b.lower[1],b.upper[1])]
    return (b.released and b.quiet(c) and supported_by(b,'/World/Table')
            and max(project(point,c)[0] for point in corners)<=c['route_region']['half_width_m']
            and r.apply([0,0,1])[2]>.99
            and abs(np.dot(r.apply([1,0,0])[:2],tangent))>.98)
def advance_setup(task,frame):
    c=task.config;tiles=[n for n in c['objects'] if n.startswith('domino')]
    good=all(standing(task,frame,n) for n in tiles)
    assignment=sorted(tiles,key=lambda n:project(task.bodies[n].center[:2],c)[1])
    task.ready_ticks=(task.ready_ticks+1 if assignment==task.setup_assignment else 1) if good else 0
    task.setup_assignment=assignment if good else []
    if task.ready_ticks>=c['stable_ticks']:
        task.chain=list(assignment);task.phase='chain';task.last_progress=task.tick
        task.event('physical_domino_setup_ready',ordered_tiles=task.chain,placement_contract='continuous_curve_band',no_hidden_slots=True)
