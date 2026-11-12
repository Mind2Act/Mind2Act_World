"""CP06 source physical measurements on existing shared simulation/robot owners."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import collider_vertices

BALANCE='/World/Task/Balance'


class Sensors:
    def __init__(self,ctx,config,bodies,arms,flanges,controls,latch):
        self.ctx,self.config,self.bodies,self.arms,self.flanges=ctx,config,bodies,arms,flanges
        self.controls,self.latch=controls,latch
        self.contacts=ctx.sim.contacts
        self.definitions={r['path']:r for r in config['all_objects']}
        self.collider_points={p:collider_vertices(ctx.sim.stage,p) for p in self.definitions}
        self.collider_centers={p:(v.min(axis=0)+v.max(axis=0))/2 for p,v in self.collider_points.items()}
        self.pan_bottom={s:float(collider_vertices(ctx.sim.stage,BALANCE+'/Pan'+s)[:,2].min()) for s in ('Left','Right')}
        self.initial_base={s:np.asarray(a.get_world_pose()[0]).copy() for s,a in arms.items()}
        self.initial_balance=self.pose(BALANCE+'/Base')[0].copy()
        self.control=dict(reading_valid=config['difficulty']=='easy',display_and_score_angle_deg=0.)
        self.events=[];self.event_index=0

    def pose(self,path):
        p,q=self.bodies[path].get_world_pose()
        return np.asarray(p),Rotation.from_quat(np.asarray(q)[[1,2,3,0]])

    def tcp(self,side):
        positions,quaternions=self.flanges[side].get_world_poses()
        r=Rotation.from_quat(np.asarray(quaternions[0])[[1,2,3,0]])
        return np.asarray(positions[0])+r.apply([.145,0.,0.]),r

    def beam_angle(self):
        axis=(self.pose(BALANCE+'/Base')[1].inv()*self.pose(BALANCE+'/Beam')[1]).apply([1,0,0])
        return math.degrees(math.atan2(-axis[2],axis[0]))

    def before_step(self):
        if self.latch:self.latch.step(push_n=0.)

    def after_step(self):
        from .display import update_cursor
        beam=self.beam_angle()
        if self.controls:
            empty=not any(self.contacts.supported(p,other=BALANCE+'/Pan'+s)
                          for p in self.definitions for s in ('Left','Right'))
            self.control=self.controls.step(beam,empty,self.latch.state()[0]/.003 if self.latch else 0.,1/120)
            self.events=self.controls.events[self.event_index:]
            self.event_index=len(self.controls.events)
        update_cursor(self.ctx.sim.stage,BALANCE,self.control['display_and_score_angle_deg'] if self.controls else beam)

    def read(self):
        contacts=[dict(pair=list(pair),impulse_ns=impulse,separation_m=separation)
                  for pair,(active,impulse,separation) in self.contacts.pairs.items()
                  if active and (impulse>1e-5 or separation<-.0005)]
        def contains(contact,root):
            return any(p==root or p.startswith(root+'/') for p in contact['pair'])
        objects={}
        for path,definition in self.definitions.items():
            p,r=self.pose(path);side='Left' if definition['kind']=='fruit' else 'Right'
            tcp,tr=self.tcp(side);pp,pr=self.pose(BALANCE+'/Pan'+side)
            local=pr.inv().apply(p+r.apply(self.collider_points[path])-pp)
            objects[path]=dict(robot_contact=bool(self.contacts.touch(path,robot_only=True)),
                external_supported=bool(self.contacts.supported(path)),position=p.tolist(),orientation_xyzw=r.as_quat().tolist(),
                linear_velocity_m_s=np.asarray(self.bodies[path].get_linear_velocity()).tolist(),
                finger_contacts=[self.contacts.touch(path,other='/World/Robot'+side,link=l) for l in ('link7','link8')],
                pan_supported={s:self.contacts.supported(path,other=BALANCE+'/Pan'+s) for s in ('Left','Right')},
                pan_center_local_m={s:self.pose(BALANCE+'/Pan'+s)[1].inv().apply(
                    p+r.apply(self.collider_centers[path])-self.pose(BALANCE+'/Pan'+s)[0]).tolist() for s in ('Left','Right')},
                origin_in_tcp_m=tr.inv().apply(p-tcp).tolist(),
                feature_in_tcp_m=tr.inv().apply(p+r.apply(definition['grasp_feature_local_m'])-tcp).tolist(),
                max_pan_local_radius_m=float(np.linalg.norm(local[:,:2],axis=1).max()),
                min_pan_local_z_m=float(local[:,2].min()))
        unexpected=[];neighbors=[];rims=[]
        for contact in contacts:
            pair=contact['pair'];robot_hit=any('/Robot' in p for p in pair)
            mount=any('/base_link/' in p for p in pair) and any('/Pedestal' in p for p in pair)
            finger=any('/Robot' in p and any('/'+l+'/' in p for l in ('link7','link8')) for p in pair)
            permitted=finger and any(contains(contact,p) for p in
                [*self.definitions,'/World/Task/CalibrationButton',BALANCE+'/Rider/Grip'])
            if robot_hit and not mount and not permitted:unexpected.append(contact)
            fruits=[p for p,d in self.definitions.items() if d['kind']=='fruit' and contains(contact,p)]
            if len(fruits)>=2 and any(any(objects[p]['finger_contacts']) for p in fruits):neighbors.append(contact)
            if fruits and any('/RimCollider' in p and '/FruitDish/' not in p for p in pair):rims.append(contact)
        rider=self.latch.state() if self.latch else (0.,0.)
        beam=self.beam_angle()
        return dict(pan_bottom_local_m=self.pan_bottom,simulation_time_s=float(self.ctx.world.current_time),objects=objects,contacts=contacts,
            beam_deg=beam,corrected_angle_deg=float(self.control['display_and_score_angle_deg']) if self.controls else beam,
            reading_valid=bool(self.control['reading_valid']),control=dict(self.control),new_control_events=list(self.events),
            rider_m=float(rider[0]),rider_velocity_m_s=float(rider[1]),
            rider_fingers=[self.contacts.touch(BALANCE+'/Rider/Grip',other='/World/RobotRight',link=l) for l in ('link7','link8')],
            unexpected_robot_contacts=unexpected,carried_fruit_neighbor_contacts=neighbors,steady_rim_contacts=rims,
            carried_fruit_rim_contacts=[c for c in rims if any(any(objects[p]['finger_contacts']) and contains(c,p)
                for p,d in self.definitions.items() if d['kind']=='fruit')],
            robot_base_motion_m=max(float(np.linalg.norm(np.asarray(a.get_world_pose()[0])-self.initial_base[s]))
                                    for s,a in self.arms.items()),
            balance_base_motion_m=float(np.linalg.norm(self.pose(BALANCE+'/Base')[0]-self.initial_balance)))
