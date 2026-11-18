"""Hard-tier exchange arcs (source: cases_unified/M/source_exchange.py, sha256 prefix in staging ledger).
Unchanged except: module-level PROFILE is the constant used by the baseline default (no environment read)."""
import numpy as np
VERSION='M-hard-six-equal-speed-v5-worker415-candidate'
SPEED=1.08
ACCELERATION=30.
PROFILES={'smooth':{'minimum_seconds':.75,'arrival_hold_seconds':.35,'degree':5},'swift':{'minimum_seconds':.65,'arrival_hold_seconds':.25,'degree':3}}
PROFILE='swift'

def spatial(stations,pair,e):
    points={k:np.asarray(v,float) for k,v in stations.items()}
    left,right=sorted(pair,key=lambda k:points[k][0]);a,b=points[left],points[right]
    between=any(a[0]<v[0]<b[0] for v in points.values());w=2*e-1
    bend=1-w**32;bd=-64*w**31;bdd=-3968*w**30
    result={}
    for name,start,end,h in [(left,a,b,.262 if between else .075),(right,b,a,.131 if between else -.075)]:
        d=end-start;p=start+e[:,None]*d+np.stack([np.zeros_like(e),h*bend],axis=1)
        pe=np.broadcast_to(d,p.shape).copy();pe[:,1]+=h*bd
        pee=np.zeros_like(p);pee[:,1]=h*bdd
        result[name]=(p,pe,pee)
    return result

def curves(stations,pair,u,profile=None):
    cfg=PROFILES[profile or PROFILE]
    grid=np.linspace(0,1,100001);sp=spatial(stations,pair,grid);primary=next(iter(sp))
    ve,ae=sp[primary][1:];base=np.linalg.norm(ve,axis=1)
    curvature=np.abs(ve[:,0]*ae[:,1]-ve[:,1]*ae[:,0])/base**3
    g=base*np.sqrt(1+.08*curvature)
    arc=np.r_[0,np.cumsum((g[:-1]+g[1:])*.5*np.diff(grid))];length=arc[-1]
    if cfg['degree']==5:
        s=u**3*(10+u*(-15+6*u));sd=30*u*u*(1-u)**2;sdd=60*u*(1-u)*(1-2*u)
    else:s=u*u*(3-2*u);sd=6*u*(1-u);sdd=6-12*u
    # C2 inverse interpolant: positions, first and second derivatives agree
    # across every interpolation knot. No piecewise-linear velocity jumps.
    ge=np.gradient(g,grid,edge_order=2);x=s*length
    j=np.clip(np.searchsorted(arc,x,side='right')-1,0,len(grid)-2)
    h=arc[j+1]-arc[j];z=(x-arc[j])/h;d=grid[j+1]-grid[j]
    m0=h/g[j];m1=h/g[j+1];a0=-ge[j]/g[j]**3*h*h;a1=-ge[j+1]/g[j+1]**3*h*h
    c0=grid[j];c1=m0;c2=a0/2;c3=10*d-6*m0-4*m1-1.5*a0+.5*a1
    c4=-15*d+8*m0+7*m1+1.5*a0-a1;c5=6*d-3*m0-3*m1-.5*a0+.5*a1
    e=c0+z*(c1+z*(c2+z*(c3+z*(c4+z*c5))))
    ex=(c1+z*(2*c2+z*(3*c3+z*(4*c4+z*5*c5))))/h
    exx=(2*c2+z*(6*c3+z*(12*c4+z*20*c5)))/(h*h)
    ed=ex*length*sd;edd=exx*(length*sd)**2+ex*length*sdd
    result=spatial(stations,pair,e)
    return {k:(p,ve*ed[:,None],ae*ed[:,None]**2+ve*edd[:,None]) for k,(p,ve,ae) in result.items()}

def exchange_paths(stations,pair,fps=120,speed=SPEED,profile=None):
    cfg=PROFILES[profile or PROFILE]
    if not 0<speed<=SPEED+1e-10:raise ValueError('speed exceeds candidate cap')
    if abs(stations[pair[0]][0]-stations[pair[1]][0])>.6400001:raise ValueError('Unsupported direct A-F inherited collision case')
    dense=curves(stations,pair,np.linspace(0,1,40001),profile)
    metrics={k:{'path_length_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),'normalized_peak_speed':float(np.linalg.norm(v,axis=1).max()),'normalized_peak_acceleration':float(np.linalg.norm(a,axis=1).max())} for k,(p,v,a) in dense.items()}
    vs=max(v['normalized_peak_speed']/speed for v in metrics.values())*1.002
    ac=max(np.sqrt(v['normalized_peak_acceleration']/ACCELERATION) for v in metrics.values())*1.002
    n=int(np.ceil(max(cfg['minimum_seconds'],vs,ac)*fps));p=curves(stations,pair,np.linspace(0,1,n+1),profile)
    return [{'label':'speed415_common_arclength_'+(profile or PROFILE),'positions':{k:v[0] for k,v in p.items()},'seconds':n/fps,'metrics':metrics,'limits':{'speed_m_s':speed,'acceleration_m_s2':ACCELERATION,'minimum_seconds':cfg['minimum_seconds']},'duration_components':{'speed':vs,'acceleration':float(ac),'visibility_floor':cfg['minimum_seconds']}}]
