"""Single removable flat board, two low side loops and measured table seating.

Robot drives carry the same board from its platform. Resting stabilization
engages only after measured table contact.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf,Sdf,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
from .shapes import material as make_material
from .shapes import rounded,cylinder
from .retrieval_geometry import (BODY_Z,FRAME_X,FRAME_Y,FRAME_Z,REVISION,
    STORAGE_CENTER,RACK_FRONT_Y,MACHINE_RETRACTION_M,
    )

ROOT='/World/CP09/Board'



def quat(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3])))


class BoardSupport:
    def __init__(self,env):
        self.env=env
        self.enabled=False
        self.hard=False
        self.initial_y=.065
        self.body=None;self.events=[];self.samples=[]
        self.latches={'Left':False,'Right':False}
        self.release_pending={'Left':False,'Right':False}
        self.contact_run={'Left':0,'Right':0}
        self.supported_ink_ticks={'eraser':0,'pen':0}
        self.unsupported_ink_ticks={'eraser':0,'pen':0}
        self.minimum_y=self.initial_y;self.maximum_y=self.initial_y
        self.ink_before_move=0;self.ink_after_move=0
        self.moved=False;self.handover_verified=False
        self.first_ink_y=None
        self.retrieval_events=[];self.retrieval_samples=[]
        self.storage_released=False;self.extracted=False;self.placed=False
        self.seat_ticks=0;self.seated_contact=False;self.machine_retract_start=None
        self.machine_offset=0.;self.machine_motion_board_positions=[]
        self.desk_released=False;self.reseat_ticks=0;self.reoriented=False
        self.maximum_yaw_deg=0.;self.maximum_tilt_deg=0.
        self.minimum_machine_clearance=float('inf');self.initial_retrieval_pose=None
        st=env.stage_usd
        body=UsdGeom.Xform.Define(st,ROOT)
        body.AddTranslateOp().Set(Gf.Vec3d(*map(float,STORAGE_CENTER)))
        body.AddOrientOp().Set(Gf.Quatf(1.))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
        api=PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
        api.CreateDisableGravityAttr(False);api.CreateSolverPositionIterationCountAttr(128)
        api.CreateSolverVelocityIterationCountAttr(8)
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.25)
        dark=make_material(st,'CP09FrameGrip',(.025,.035,.043),0,.62)
        metal=make_material(st,'CP09FrameGripStem',(.44,.49,.52),.68,.43)
        material=UsdShade.Material.Define(st,ROOT+'/GripPhysics')
        api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(2.);api.CreateDynamicFrictionAttr(1.8);api.CreateRestitutionAttr(0.)
        for side,sign in [('Left',-1),('Right',1)]:
            x=sign*FRAME_X
            # A single low, continuous U-shaped side handle, integrated into
            # the frame. No vertical posts or extra front extraction grips.
            for suffix,dy in [('Near',-.030),('Far',.030)]:
                rounded(st,'/World/CP09/Work/'+side+suffix+'HandleMount',
                        (sign*.249,FRAME_Y+dy,FRAME_Z),(.037,.012,.014),.005,metal)
            cylinder(st,'/World/CP09/Work/'+side+'HandleBridge',
                     (x,FRAME_Y,FRAME_Z),.0085,.060,dark,'Y')
            grip=rounded(st,ROOT+'/'+side+'FrameGrip',
                         (x,FRAME_Y,FRAME_Z-BODY_Z),(.017,.026,.020),.004,dark)
            UsdGeom.Imageable(grip.GetPrim()).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(grip.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(grip.GetPrim()).CreateApproximationAttr('convexHull')
            PhysxSchema.PhysxCollisionAPI.Apply(grip.GetPrim()).CreateContactOffsetAttr(.0001)
            UsdShade.MaterialBindingAPI.Apply(grip.GetPrim()).Bind(material,materialPurpose='physics')
        # Full board backing and four feet are real rigid-body support shapes.
        parts=[('Backing',(0,0,0),(.480,.330,.020))]
        parts += [(f'Foot{sx+1}{sy+1}',(sx*.205,sy*.133,-.012),(.035,.025,.008))
                  for sx in [-1,1] for sy in [-1,1]]
        for name,point,size in parts:
            shape=UsdGeom.Cube.Define(st,ROOT+'/'+name);shape.CreateSizeAttr(1.)
            shape.AddTranslateOp().Set(Gf.Vec3d(*point));shape.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim()).CreateContactOffsetAttr(.0001)
            UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
        clip=UsdPhysics.FixedJoint.Define(st,ROOT+'/StorageClip')
        clip.CreateBody0Rel().SetTargets(['/World/CP09/StorageMachine'])
        clip.CreateBody1Rel().SetTargets([ROOT])
        clip.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,STORAGE_CENTER)))
        clip.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
        clip.CreateExcludeFromArticulationAttr(True)
        col=UsdGeom.Cube.Define(st,ROOT+'/BoardContact');col.CreateSizeAttr(1.)
        col.AddTranslateOp().Set(Gf.Vec3d(0,0,env.board_surface_z-BODY_Z-.0015))
        col.AddScaleOp().Set(Gf.Vec3f(.440,.285,.003))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim()).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim()).CreateContactOffsetAttr(.0001)
        UsdGeom.Imageable(col.GetPrim()).MakeInvisible()

        # Visuals and collision shapes are children of one physical board.
        # Moving the rigid body therefore also moves its actual ink mesh.
        edit=Sdf.BatchNamespaceEdit();edit.Add('/World/CP09/Work',ROOT+'/Visuals')
        if not st.GetRootLayer().Apply(edit):raise RuntimeError('Cannot parent single board visuals')
        visual=UsdGeom.Xformable(st.GetPrimAtPath(ROOT+'/Visuals'))
        visual.GetOrderedXformOps()[0].Set(Gf.Vec3d(0,0,-BODY_Z))
        env.w['ink_meshes']['Work']=UsdGeom.Mesh(st.GetPrimAtPath(ROOT+'/Visuals/ActualInk'))
        env.w['roots'].pop('Work',None)
        for side in ('Left','Right'):
            st.RemovePrim(ROOT+'/'+side+'FrameGrip')
            for name in (side+'HandleBridge',side+'NearHandleMount',side+'FarHandleMount'):
                prim=st.GetPrimAtPath(ROOT+'/Visuals/'+name)
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                if prim.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr(.0001);api.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')

    def _seat_guide(self):
        # Stabilize measured resting contact; never drive a board pose.
        st=self.env.stage_usd;p=self.position()
        joint=UsdPhysics.FixedJoint.Define(st,ROOT+'/DeskClamp')
        joint.CreateBody1Rel().SetTargets([ROOT])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,p)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
        joint.CreateLocalRot0Attr().Set(quat(self.rotation()))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
        joint.CreateExcludeFromArticulationAttr(True)

    def rotation(self):
        q=self.body.get_world_pose()[1]
        return Rotation.from_quat([*q[1:],q[0]]).as_matrix()

    def surface_point(self,xy,offset=0.):
        return self.position()+self.rotation()@np.r_[xy,self.env.board_surface_z-BODY_Z+offset]

    def register(self,world):
        from isaacsim.core.prims import SingleRigidPrim
        self.body=world.scene.add(SingleRigidPrim(prim_path=ROOT,name='CP09_SupportedBoard'))
        self.machine_motion=UsdGeom.Xformable(self.env.stage_usd.GetPrimAtPath('/World/CP09/StorageMachine')).GetOrderedXformOps()[0]

    def position(self):
        return np.asarray(self.body.get_world_pose()[0])

    def grip_position(self,side):
        return self.position()+self.rotation()@np.array([(-1 if side=='Left' else 1)*FRAME_X,FRAME_Y,FRAME_Z-BODY_Z+.016])

    def contacts(self,side):
        tcp,_=self.env.kin[side].fk(self.env.robots[side].get_joint_positions()[:6])
        if np.linalg.norm(tcp-self.grip_position(side))>.035:return [False,False]
        return [any(ROOT+'/Visuals/'+side+'HandleBridge' in pair and
                    any(p.startswith('/World/Robot'+side+'/Asset/'+link+'/collisions/') for p in pair)
                    and active and separation<=.00005 and (impulse>1e-8 or separation<=0.)
                    for pair,(active,impulse,separation) in self.env.contact_pairs.items())
                for link in ('link7','link8')]

    def _attach(self,side):
        if not self.storage_released and self.env.tick<600:return
        st=self.env.stage_usd
        p,q=self.body.get_world_pose();r=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
        tcp,rotation=self.env.kin[side].fk(self.env.robots[side].get_joint_positions()[:6])
        flange=tcp-rotation@np.array([.145,0,0])
        joint=UsdPhysics.FixedJoint.Define(st,ROOT+'/'+side+'VerifiedFrameGrasp')
        joint.CreateBody0Rel().SetTargets(['/World/Robot'+side+'/Asset/link6'])
        joint.CreateBody1Rel().SetTargets([ROOT])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,rotation.T@(p-flange))))
        joint.CreateLocalRot0Attr().Set(quat(rotation.T@r))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
        joint.CreateExcludeFromArticulationAttr(True)
        filtered=UsdPhysics.FilteredPairsAPI.Apply(st.GetPrimAtPath(ROOT)).CreateFilteredPairsRel()
        for link in ['link6','link7','link8']:filtered.AddTarget('/World/Robot'+side+'/Asset/'+link)
        self.latches[side]=True
        if not self.storage_released:
            st.RemovePrim(ROOT+'/StorageClip')
            self.storage_released=True
            self.initial_retrieval_pose=np.asarray(p).copy()
            self.retrieval_events.append(dict(tick=self.env.tick,event='storage_clip_released_after_bilateral_grasp',side=side,physical_contact=True,board_position_m=np.asarray(p).tolist()))
        measured=self.env.robots[side].get_joint_positions()[6:].copy()
        self.env.gripper_opening[side]=float(np.mean(measured));self.env.commands[side][6:]=measured
        event=dict(tick=self.env.tick,event='frame_grasp_verified',side=side,bilateral_contact_ticks=self.contact_run[side],
                   board_center_y_m=float(p[1]),physical_contact=True,stabilization_after_contact=True)
        self.events.append(event);print('CP09_BOARD_SUPPORT',event,flush=True)
        if all(self.latches.values()):
            self.handover_verified=True
            self.events.append(dict(tick=self.env.tick,event='two_hand_support_overlap',physical_grasps=True))

    def release(self,side):
        if not self.latches[side]:return
        self.env.stage_usd.RemovePrim(ROOT+'/'+side+'VerifiedFrameGrasp')
        self.release_pending[side]=True
        self.latches[side]=False;self.contact_run[side]=0
        self.events.append(dict(tick=self.env.tick,event='frame_released',side=side,other_hand_support=self.latches['Right' if side=='Left' else 'Left']))

    def sample(self):
        for side,tool in [('Left','eraser'),('Right','pen')]:
            if self.release_pending[side] and min(self.env.robots[side].get_joint_positions()[6:])>.024:
                relation=UsdPhysics.FilteredPairsAPI(self.env.stage_usd.GetPrimAtPath(ROOT)).GetFilteredPairsRel()
                for link in ('link6','link7','link8'):relation.RemoveTarget('/World/Robot'+side+'/Asset/'+link)
                self.release_pending[side]=False
            empty=not self.env.grasp_latches[tool]
            contact=all(self.contacts(side)) and self.env.gripper_opening[side]<.018 and empty
            self.contact_run[side]=self.contact_run[side]+1 if contact else 0
            if not self.latches[side] and self.contact_run[side]>=24:self._attach(side)
        self._sample_retrieval()
        y=float(self.position()[1])
        if self.placed:self.minimum_y=min(self.minimum_y,y);self.maximum_y=max(self.maximum_y,y)
        if self.env.tick%20==0:
            self.samples.append([self.env.tick,y,*self.contacts('Left'),*self.contacts('Right'),
                                 int(self.latches['Left']),int(self.latches['Right']),int(self.moved)])

    def record_ink(self,tool):
        if not self.enabled:return True
        if self.first_ink_y is None:self.first_ink_y=float(self.position()[1])
        other='Right' if tool=='eraser' else 'Left'
        supported=self.latches[other]
        target=self.supported_ink_ticks if supported else self.unsupported_ink_ticks
        target[tool]+=1
        if self.moved:self.ink_after_move+=1
        else:self.ink_before_move+=1
        return supported

    def observation(self):
        return dict(enabled=self.enabled,center_y_m=float(self.position()[1]),
                    retrieval_grip_world_m={s:self.grip_position(s).tolist() for s in ('Left','Right')},
                    retrieval=self.retrieval_observation(),
                    frame_grip_world_m={s:self.grip_position(s).tolist() for s in ('Left','Right')},
                    finger_contacts={s:self.contacts(s) for s in ('Left','Right')},
                    supporting_hands=self.latches.copy(),moved=self.moved,initial_center_y_m=self.initial_y,
                    rotation_matrix=self.rotation().tolist(),reoriented=self.reoriented)

    def evidence(self):
        fractions={t:self.supported_ink_ticks[t]/max(1,self.supported_ink_ticks[t]+self.unsupported_ink_ticks[t]) for t in ('eraser','pen')}
        success=not self.enabled or (self.handover_verified and min(fractions.values())>=.98)
        success=success and (not self.hard or (self.reoriented and self.ink_before_move>0 and self.ink_after_move>0))
        return dict(enabled=self.enabled,success=success,supported_ink_ticks=self.supported_ink_ticks,
                    unsupported_ink_ticks=self.unsupported_ink_ticks,support_fraction=fractions,
                    handover_verified=self.handover_verified,initial_center_y_m=self.initial_y,
                    maximum_translation_m=self.initial_y-self.minimum_y,ink_ticks_before_move=self.ink_before_move,
                    maximum_yaw_deg=self.maximum_yaw_deg,maximum_tilt_deg=self.maximum_tilt_deg,reoriented=self.reoriented,
                    ink_ticks_after_move=self.ink_after_move,events=self.events,
                    motion_source='measured robot frame grasp and table-contact resting stabilization',
                    grasp_assistance='fixed frame-grasp stabilization only after 24 consecutive bilateral-contact ticks',
                    automatic_board_position_drive=False,resting_contact_stabilization=True)

    def retrieval_observation(self):
        return dict(storage_released=self.storage_released,extracted=self.extracted,placed=self.placed,
                    machine_retracted=self.machine_offset>=MACHINE_RETRACTION_M-.001,
                    board_position_m=self.position().tolist(),desk_center_y_m=self.initial_y)

    def _sample_retrieval(self):
        p=self.position()
        rotation=self.rotation()
        rear=p[1]+float(np.dot(np.abs(rotation[1]),[.240,.165,.016]))
        clearance=RACK_FRONT_Y+self.machine_offset-rear
        if self.storage_released and not self.extracted and clearance>=.012 and p[2]>=BODY_Z+.08 and any(self.latches.values()):
            self.extracted=True
            self.retrieval_events.append(dict(tick=self.env.tick,event='board_completely_clear_of_rack',clearance_m=float(clearance),board_position_m=p.tolist(),held=True))
        table_contact=self._table_foot_contact()
        seated=self.extracted and abs(p[1]-self.initial_y)<.008 and abs(p[2]-BODY_Z)<.006 and table_contact
        self.seat_ticks=self.seat_ticks+1 if seated else 0
        if not self.placed and self.seat_ticks>=24:
            self.placed=True;self.seated_contact=True;self._seat_guide()
            self.machine_retract_start=self.env.tick
            self.retrieval_events.append(dict(tick=self.env.tick,event='board_seated_on_independent_table',board_position_m=p.tolist(),actual_foot_table_contact=True,consecutive_contact_ticks=self.seat_ticks,rack_clearance_m=float(clearance)))
        if self.placed:
            self.minimum_machine_clearance=min(self.minimum_machine_clearance,float(clearance))
            if self.machine_offset<MACHINE_RETRACTION_M:
                alpha=min(1.,(self.env.tick-self.machine_retract_start)/180.)
                self.machine_offset=MACHINE_RETRACTION_M*alpha*alpha*(3-2*alpha)
                self.machine_motion.Set(Gf.Vec3d(0.,self.machine_offset,0.))
                self.machine_motion_board_positions.append(p.copy())
                if alpha>=1.:
                    self.retrieval_events.append(dict(tick=self.env.tick,event='empty_machine_retracted',travel_m=self.machine_offset,board_position_m=p.tolist(),rack_clearance_m=float(clearance)))
        if self.env.tick%4==0:
            self.retrieval_samples.append([self.env.tick,*p,self.machine_offset,clearance,int(table_contact),int(self.storage_released),int(self.extracted),int(self.placed)])

    def _table_foot_contact(self):
        return any(any(q.startswith(ROOT+'/Foot') for q in pair) and
                   any(q.startswith('/World/Worktable') for q in pair) and active and (impulse>1e-8 or separation<.0002)
                   for pair,(active,impulse,separation) in self.env.contact_pairs.items())

    def retrieval_evidence(self):
        positions=np.asarray(self.machine_motion_board_positions)
        drift=float(np.max(np.linalg.norm(positions-positions[0],axis=1))) if len(positions) else None
        events=self.retrieval_events
        success=(self.storage_released and self.extracted and self.placed and self.seated_contact
                 and self.machine_offset>=MACHINE_RETRACTION_M-.001 and drift is not None and drift<.003
                 and self.minimum_machine_clearance>.012)
        return dict(required=True,success=bool(success),initial_board_position_m=STORAGE_CENTER.tolist(),
                    final_board_position_m=self.position().tolist(),events=events,
                    storage_clip_removed=not self.env.stage_usd.GetPrimAtPath(ROOT+'/StorageClip').IsValid(),
                    extracted_before_lowering=self.extracted,actual_foot_table_contact=self.seated_contact,
                    machine_retraction_m=self.machine_offset,minimum_machine_clearance_m=self.minimum_machine_clearance if self.placed else None,
                    board_drift_during_machine_retraction_m=drift,board_parented_to_machine=False,
                    motion_source='measured PhysX rigid board held by actual-contact-verified robot frame grasp; no board pose writes')
