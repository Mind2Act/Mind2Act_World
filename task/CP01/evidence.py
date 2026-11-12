"""CP01 terminal measurements from FINAL-VERSION.md.

Transport, release and time histories are diagnostic only. Food credit uses
actual tray support and its measured geometric center inside the tray cavity.
"""
from collections import Counter,deque
from copy import deepcopy
from mind2act_eval.scoring import SCORING_VERSION
import numpy as np
from mind2act_eval.scoring import container_contains


class Stable:
    def __init__(self,require_transport=True):
        self.require_transport=require_transport
        self.grasped=False;self.lifted=False;self.positions=deque(maxlen=120)

    def update(self,p,*,bilateral,elevated,released,supported,correct):
        self.grasped|=bool(bilateral);self.lifted|=bool(bilateral and elevated)
        self.positions.clear()
        if supported and correct:
            self.positions.append(np.asarray(p, float))

    @property
    def complete(self):
        return bool(self.positions)


def rotation(row):
    w,x,y,z=np.asarray(row['quaternion'],float)/np.linalg.norm(row['quaternion'])
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def separated_trays(transforms,size):
    """Non-overlapping tabletop footprints, including in-plane tray rotation."""
    half=np.asarray(size,float)/2
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
    footprints=[p[:2]+corners@rot[:2,:2].T for p,rot in transforms]
    separated=[True]*len(footprints)
    for i,a in enumerate(footprints):
        for j in range(i+1,len(footprints)):
            b=footprints[j];apart=False
            for polygon in (a,b):
                for edge in np.roll(polygon,-1,axis=0)-polygon:
                    axis=np.array([-edge[1],edge[0]])
                    if np.linalg.norm(axis)<1e-12:continue
                    pa,pb=a@axis,b@axis
                    if pa.max()<=pb.min() or pb.max()<=pa.min():apart=True;break
                if apart:break
            if not apart:separated[i]=separated[j]=False
    return separated


class Evidence:
    def __init__(self,setup,max_food_tilt):
        self.setup=setup;self.max_tilt=float(max_food_tilt)
        self.foods=[Stable() for _ in setup['items']];self.trays=[Stable(require_transport=False) for _ in setup['orders']]
        self.phase="observation";self.initial=None;self.last_tick=0;self.violations=set();self.inside=[];self.ready=[]
        self.tray_first_ready={};self.food_first_complete={}

    def sample(self,row):
        tick=row['physics_tick']
        if type(tick)!=int or tick!=self.last_tick+1:raise ValueError('CP01 evidence clock gap')
        self.last_tick=tick
        self.terminal_measurements=deepcopy(row)
        bodies=row['bodies'];count=len(self.foods)
        if len(bodies)!=count+len(self.trays):raise ValueError('CP01 measured body coverage differs')
        for body in bodies:
            values=[*body['position'],*body['quaternion'],*body['local_center'],*body['half_extents']]
            if not np.isfinite(values).all():raise ValueError('Nonfinite CP01 pose/geometry')
        if self.initial is None:
            self.initial=[np.asarray(b['position'],float) for b in bodies]
        transforms=[(np.asarray(body['position']),rotation(body)) for body in bodies[count:]]
        separated=separated_trays(transforms,self.setup['tray_spec']['tray_size_m'])
        self.ready=[]
        for j,gate in enumerate(self.trays):
            body=bodies[count+j];p,rot=transforms[j]
            correct=bool(separated[j])
            gate.update(p,bilateral=body['bilateral'],elevated=False,
                        released=not body['touch_robot'],supported=body['touch_table'],correct=correct)
            self.ready.append(gate.complete)
            if self.ready[-1] and j not in self.tray_first_ready:self.tray_first_ready[j]=tick
        self.inside=[[] for _ in self.trays]
        sx,sy=self.setup['tray_spec']['tray_size_m']
        for i,gate in enumerate(self.foods):
            body=bodies[i];p=np.asarray(body['position']);rot=rotation(body);fit=None;local=p
            for j,(tp,tr) in enumerate(transforms):
                candidate=tr.T@(p-tp);relative=tr.T@rot
                center=candidate+relative@np.asarray(body['local_center'])
                # The authored 4mm rim is physical cavity geometry, not an
                # extra inset or whole-object containment margin.
                if container_contains(supported=j in body['touch_trays'], center=center[:2],
                        lower=[-sx/2+.004, -sy/2+.004], upper=[sx/2-.004, sy/2-.004]):
                    fit=j;local=candidate;self.inside[j].append(i);break
            gate.update(local,bilateral=body['bilateral'],elevated=p[2]>self.initial[i][2]+.025,
                released=not body['touch_robot'],supported=fit is not None and fit in body['touch_trays'],
                correct=fit is not None)
        for i,gate in enumerate(self.foods):
            if gate.complete and i not in self.food_first_complete:self.food_first_complete[i]=tick
        self.phase=row['phase'];return self.grade()

    def grade(self):
        """Native facts for the 2026-10-29 rule: required steps satisfied, minus wrong/extra items.

        Tray table support/separation and food placement count independently.
        First-ready timestamps do not gate terminal credit.
        """
        matches=[];credited=0;wrong=0;tray_steps=0
        easy=self.setup['difficulty']=='easy'
        for j,(ids,order) in enumerate(zip(self.inside,self.setup['orders'])):
            kinds=Counter(self.setup['items'][i]['kind'] for i in ids)
            matches.append(kinds==Counter(order))
            wrong+=len(ids)-sum((kinds&Counter(order)).values())
            if self.ready[j] and not easy:tray_steps+=1
            complete=[i for i in ids if self.foods[i].complete]
            credited+=sum((Counter(self.setup['items'][i]['kind'] for i in complete)&Counter(order)).values())
        n=sum(map(len,self.setup['orders']));selected=[i for ids in self.inside for i in ids]
        success=bool(self.phase=='execution' and self.ready and all(self.ready)
            and all(matches) and all(self.foods[i].complete for i in selected) )
        required_steps=n+(0 if easy else len(self.setup['orders']))
        satisfied=tray_steps+credited
        return dict(scoring_version=SCORING_VERSION, terminal_measurements=getattr(self,'terminal_measurements',None),
                    credited_foods=credited,required_foods=n,tray_steps=tray_steps,required_steps=required_steps,
                    wrong_or_extra_items=wrong,steps_satisfied=satisfied,k=max(0,satisfied-wrong),
                    task_success=success,tray_readiness=list(self.ready),order_matches=matches,
                    inside=[list(x) for x in self.inside],protocol_violations=sorted(self.violations))


def evaluate(setup,rows,max_food_tilt):
    evidence=Evidence(setup,max_food_tilt)
    if not rows:raise ValueError('CP01 measured trace is empty')
    for row in rows:evidence.sample(row)
    return evidence.grade()
