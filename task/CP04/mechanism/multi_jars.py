from .geometry import RELEASE_AXIAL_TOLERANCE
"""Independent passive condiment bodies/caps and reciprocal screw constraints."""
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,PhysxSchema
import numpy as np
import os,math
from scipy.spatial.transform import Rotation as R
from .geometry import CAP_Z,PITCH,RELEASE_ANGLE,drawer_jar_position,CAP_INERTIA
from .screw_state import cleared_neck
ROOT='/World/Cooking'

class JarSet:

    @staticmethod
    def pose(body):
        p,q=body.get_world_pose();return p,R.from_quat([*q[1:],q[0]]).as_matrix()

    def add_primary(self,index,jar,lid,guide,body_path,lid_path):
        self.items[index]=dict(jar=jar,lid=lid,guide=guide,released=False,theta=0.,last_angle=0.,body_path=body_path,lid_path=lid_path)

    def build_batch(self,world):
        from isaacsim.core.prims import RigidPrim
        paths=[item[key] for item in self.items.values() for key in ['body_path','lid_path']]
        self.batch=world.scene.add(RigidPrim(prim_paths_expr=paths,name='condiment_batch'))
        self.path_indices={path:i for i,path in enumerate(self.batch.prim_paths)}
        # scale492 solver-side passive friction: zero stiffness, zero target
        # relative velocity, bounded0.04Nm torque. USD angular damping uses
        # degree coordinates;0.801Nms/rad becomes0.801*pi/180 authored units.
        # This participates in the same120Hz constraint solve as grasp contacts,
        # including position-solver motion absent from velocity tensors.
        for item in self.items.values():
            drive=UsdPhysics.DriveAPI.Apply(item['guide'].GetPrim(),'rotZ')
            drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(.801*math.pi/180.)
            drive.CreateMaxForceAttr(.04);drive.CreateTargetVelocityAttr(0.)


    def pre_step(self):
        positions,quaternions=self.batch.get_world_poses()
        rotations=R.from_quat(quaternions[:,[1,2,3,0]]).as_matrix()
        linear=self.batch.get_linear_velocities();angular=self.batch.get_angular_velocities()
        forces=np.zeros_like(positions);torques=np.zeros_like(positions);application=positions.copy()
        for item in self.items.values():
            bi=self.path_indices[item['body_path']];li=self.path_indices[item['lid_path']]
            bp,br=positions[bi],rotations[bi];lp,lr=positions[li],rotations[li];rr=br.T@lr;angle=np.arctan2(rr[1,0],rr[0,0])
            item['theta']+=(angle-item['last_angle']+np.pi)%(2*np.pi)-np.pi;item['last_angle']=angle
            theta=item['theta'];relative=br.T@(lp-bp);rise=relative[2]-CAP_Z
            if item['released']:
                if cleared_neck(float(rise),float(np.linalg.norm(relative[:2]))):item['recap_armed']=True
                if not self.enable_recap or not item.get('recap_armed',False):continue
                spin=float(br[:,2]@(angular[li]-angular[bi]))
                # A loose cap engages only when geometrically aligned and
                # actually turned clockwise. No trajectory phase attaches it.
                aligned=np.linalg.norm(relative[:2])<.0015 and rr[2,2]>.999
                lowering=float(br[:,2]@(linear[li]-linear[bi]))<-.0005
                if aligned and .0025<rise<.0055 and spin<-.05 and lowering:
                    start_angle=rise/PITCH*2*np.pi;offset=angle-start_angle
                    item['guide_offset']=float(offset)
                    item['guide'].CreateLocalRot0Attr(Gf.Quatf(float(np.cos(offset/2)),0,0,float(np.sin(offset/2))))
                    item['guide'].CreateJointEnabledAttr(True);item['released']=False;item['recapping']=True;item['recap_armed']=False
                    item['theta']=theta=start_angle
                else:continue
            if item.get('recapping') and theta<np.deg2rad(5) and abs(rise)<.001:
                item['recapped']=True;item['recapping']=False
            if not item.get('recapping') and np.degrees(theta)>=RELEASE_ANGLE and rise>=PITCH*RELEASE_ANGLE/360-RELEASE_AXIAL_TOLERANCE:
                item['guide'].CreateJointEnabledAttr(False);item['released']=True;item['recap_armed']=False;continue
            axis=br[:,2];bv=linear[bi];lv=linear[li];bw=angular[bi];lw=angular[li]
            lead=PITCH/(2*np.pi);error=rise-lead*theta;speed=axis@(lv-bv)-lead*(axis@(lw-bw))
            # scale492: backward-Euler axial spring, same passive1800N/m,4Ns/m,
            # reciprocal12N cap and one actual world.step per120Hz sample.
            # Explicit spring is unstable at dt1/120 for the25g cap: observed
            # -4.66/+8.11mm oscillation under transport before finger contact.
            from .geometry import CAP_MASS,JAR_RADIUS
            inv_mass=1/.14+1/CAP_MASS+lead*lead*(1/CAP_INERTIA[2]+1/(.14*JAR_RADIUS**2/2))
            dt=self.physics_dt
            f=float(np.clip((-1800*(error+dt*speed)-4*speed)/(1+dt*(4+1800*dt)*inv_mass),-12,12))
            spin=float(axis@(lw-bw));torque=-lead*f-.001*spin-.04*np.tanh(spin/.05)
            # Helical reciprocal torque retained. Friction itself is solved
            # by the bounded zero-stiffness guide drive, never a pose target.
            torque=-lead*f
            for index,sgn in [(bi,-1),(li,1)]:forces[index]=sgn*f*axis;torques[index]=sgn*torque*axis;application[index]=lp
        self.batch.apply_forces_and_torques_at_pos(forces=forces,torques=torques,positions=application,is_global=True)

    def snapshot(self):
        return {str(i):dict(released=item['released'],recap_armed=item.get('recap_armed',False),recapped=item.get('recapped',False),recapping=item.get('recapping',False),guide_offset=item.get('guide_offset',0.),theta=item['theta'],body_path=item['body_path'],lid_path=item['lid_path'],**{n:dict(position=self.pose(item[n])[0].tolist(),rotation=self.pose(item[n])[1].tolist()) for n in ['jar','lid']}) for i,item in self.items.items()}
