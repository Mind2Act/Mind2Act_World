"""Source-backed transfers and stability; no mat assignment or score policy."""
from collections import deque
import math
import numpy as np
from .measurements import bilateral,robot_paths,supports,tilt,validate_frame


def landed(row,name,expected):
    p=row['poses'][name]['position']
    return (math.dist(p[:2],expected[:2])<=.01+1e-12 and abs(p[2]-expected[2])<=.003+1e-12
            and tilt(row['poses'][name]['quaternion_wxyz'])<=5
            and not robot_paths(row,name) and bool(supports(row,name)))


class PhysicalFacts:
    def __init__(self,source,initial):
        self.names=[*[f'Tier{i+1}' for i in range(len(source['target_foods_bottom_to_top']))],'Lid']
        self.decoys=list(source['decoy_xyz'])
        self.objects=self.names+self.decoys
        validate_frame(initial,self.objects)
        if initial['physics_tick']!=0:
            raise ValueError('P2 native reset sample must be tick0')
        self.tick=0;self.initial=initial
        self.previous_time=initial['simulation_time_s']
        self.carries={n:dict(touch=False,origin=initial['poses'][n]['position'],qualified=False,
                            released_position=initial['poses'][n]['position']) for n in self.objects}
        self.decoy_motion={n:0. for n in self.decoys}
        self.base_motion=0.
        self.illegal_target_contact=False
        self.tail=deque(maxlen=360);self.unstable_until=-1
        self.last=initial

    def sample(self,row):
        validate_frame(row,self.objects)
        if row['physics_tick']!=self.tick+1:
            raise ValueError('P2 consecutive native physical samples required')
        if abs(row['simulation_time_s']-self.previous_time-1/120)>1e-6:
            raise ValueError('P2 native samples must advance at120Hz')
        self.previous_time=row['simulation_time_s']
        self.tick=row['physics_tick'];lifts=[];releases=[]
        self.base_motion=max(self.base_motion,math.dist(row['preplaced_xyz'],self.initial['preplaced_xyz']))
        for name in self.objects:
            carry=self.carries[name];p=row['poses'][name]['position'];paths=robot_paths(row,name)
            touched=bool(paths)
            if touched and not carry['touch']:
                carry.update(origin=carry['released_position'],qualified=False)
            if (touched and bilateral(row,name) and p[2]-carry['origin'][2]>=.03-1e-12
                    and not carry['qualified']):
                carry['qualified']=True
                lifts.append(dict(item=name,tick=self.tick,origin=list(carry['origin']),position=list(p)))
            if not touched and carry['qualified'] and supports(row,name):
                releases.append(dict(item=name,tick=self.tick,origin=list(carry['origin']),position=list(p)))
                carry['qualified']=False
            if not touched and supports(row,name):
                carry['released_position']=p
            carry['touch']=touched
            if name in self.decoys:
                self.decoy_motion[name]=max(self.decoy_motion[name],math.dist(p,self.initial['poses'][name]['position']))
            else:
                self.illegal_target_contact |= any(not any('/'+link+'/' in path for link in ('link7','link8')) for path in paths)
        positions=np.asarray([row['poses'][n]['position'] for n in self.names],dtype=float)
        if self.tail:
            old=np.asarray([p for _,p in self.tail])
            bad=np.any(np.linalg.norm(old-positions,axis=2)>.0005+1e-12,axis=1)
            self.unstable_until=max(self.unstable_until,max((t+359 for (t,_),b in zip(self.tail,bad) if b),default=-1))
        if any(robot_paths(row,n) or not supports(row,n) for n in self.names):
            self.unstable_until=max(self.unstable_until,self.tick+359)
        self.tail.append((self.tick,positions));self.last=row
        return dict(lifts=lifts,releases=releases)

    @property
    def stable(self):
        return len(self.tail)==360 and self.tick>self.unstable_until
