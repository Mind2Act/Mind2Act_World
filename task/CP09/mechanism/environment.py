"""Measured source contact/presentation mechanisms; no controller or robot configuration."""
import math,hashlib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from pxr import Gf,UsdPhysics
from .surface import InkSurface
from .props import write_ink_mesh
from .retrieval_geometry import BODY_Z
from .presentation import MEMORY_END,ANNOTATION_TICK,REVEAL_END,shutter_fraction,SHUTTER_OPEN_Y,SHUTTER_CLOSED_Y

def gf_quat(r):
    q=Rot.from_matrix(r).as_quat()
    return Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3])))

class Environment:
    def __init__(self,w,robots,kin,world,tools_module):
        self.w=w;self.stage_usd=w['stage'];self.case=w['case'];self._target=w['target']
        self.surface=InkSurface(w['target']);self.surface_identity=id(self.surface.ink)
        self.initial_ink_sha256=hashlib.sha256(self.surface.ink.tobytes()).hexdigest()
        self.presentation_events=[];self.annotation_applied=False;self.tick=0;self.phase='memorize'
        self.robots=robots;self.kin=kin;self.world=world
        self.contact_counts={'eraser':0,'pen':0};self.last_contact={'eraser':False,'pen':False}
        self.contact_pairs={};self.contact_pair_last_report_tick={};self.contact_log=[]
        self.board_surface_z=w['board_z'];self.board_center_xy=np.array([0.,.065])
        self.commands={s:np.r_[np.radians([0,65,100,-65,0,0]),.044,.044] for s in ("Left","Right")}
        self.gripper_opening={s:float(np.mean(q[6:])) for s,q in self.commands.items()}
        self.active_pen=1;self.pen_verified={i:False for i in range(1,self.case['color_count']+1)}
        self.pen_contact_counts={i:0 for i in self.pen_verified};self.pickup_events=[]
        self.pickup_verified={'eraser':False,'pen':False};self.grasp_latches={'eraser':False,'pen':False}
        self.grasp_contact_ticks={'eraser':0,'pen':0};self.lift_ticks={'eraser':0,'pen':0}
        self.initial_tool_positions={};self.initial_tools_unattached=True
        self.tool_root_for=tools_module.root_for;self.grip_height=tools_module.GRIP_HEIGHT

    def _tool_attachment_paths(self):
        tools={self.tool_root_for('eraser'),*[self.tool_root_for('pen',i) for i in self.pen_verified]}
        paths=[]
        for prim in self.stage_usd.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint=UsdPhysics.Joint(prim)
                bodies=set(map(str,joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets()))
                if bodies&tools and any(p.startswith('/World/Robot') for p in bodies) and joint.GetJointEnabledAttr().Get()!=False:paths.append(str(prim.GetPath()))
        return paths

    def tool_root(self,tool):return self.tool_root_for(tool,self.active_pen)

    def select_pen(self,color):
        if self.grasp_latches['pen']:raise RuntimeError('Return the current marker before changing color')
        if color not in self.pen_verified:raise ValueError('No marker with that color')
        self.active_pen=int(color);self.tools['pen']=self.all_tools['pen'+str(color)]
        self.pickup_verified['pen']=self.pen_verified[color]
        self.initial_tool_positions['pen']=self.initial_pen_positions[color].copy()
        self.grasp_contact_ticks['pen']=0;self.lift_ticks['pen']=0

    def _finger_contacts(self,tool):
        root=self.tool_root(tool)+'/'
        side='Left' if tool=='eraser' else 'Right'
        tcp,_=self.kin[side].fk(self.robots[side].get_joint_positions()[:6])
        position,rotation=self._tool_pose(tool)
        if np.linalg.norm(tcp-(position+rotation@np.array([0,0,self.grip_height[tool]])))>.035:return [False,False]
        return [any(any(p.startswith(root) for p in pair) and
                    any(p.startswith('/World/Robot'+side+'/Asset/'+link+'/') for p in pair)
                    and active and (impulse>1e-8 or separation<.00005)
                    for pair,(active,impulse,separation) in self.contact_pairs.items())
                for link in ('link7','link8')]

    def _stabilize_verified_grasp(self,tool):
        """A latch is allowed only AFTER a real bilateral-contact free lift."""
        if not self.pickup_verified[tool]:raise RuntimeError('Cannot latch an unverified pickup')
        side='Left' if tool=='eraser' else 'Right'
        p,r=self._tool_pose(tool)
        tcp,flange_r=self.kin[side].fk(self.robots[side].get_joint_positions()[:6])
        flange_p=tcp-flange_r@np.array([.145,0,0])
        root=self.tool_root(tool)
        joint=UsdPhysics.FixedJoint.Define(self.stage_usd,root+'/VerifiedGraspLatch')
        joint.CreateBody0Rel().SetTargets(['/World/Robot'+side+'/Asset/link6'])
        joint.CreateBody1Rel().SetTargets([root])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,flange_r.T@(p-flange_p))))
        joint.CreateLocalRot0Attr().Set(gf_quat(flange_r.T@r))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateJointEnabledAttr(True)
        pairs=UsdPhysics.FilteredPairsAPI.Apply(self.stage_usd.GetPrimAtPath(root)).CreateFilteredPairsRel()
        for link in ['link6','link7','link8']:pairs.AddTarget('/World/Robot'+side+'/Asset/'+link)
        self.grasp_latches[tool]=True
        measured_opening=self.robots[side].get_joint_positions()[6:].copy()
        self.gripper_opening[side]=float(np.mean(measured_opening))
        self.commands[side][6:]=measured_opening

    def _measure_pickup(self):
        for tool in ('eraser','pen'):
            pair=self._finger_contacts(tool)
            if all(pair):self.grasp_contact_ticks[tool]+=1
            if tool=='pen' and all(pair) and self.grasp_contact_ticks[tool]>=24:
                self.stage_usd.RemovePrim(self.tool_root(tool)+'/DockRetention')
            p,_=self._tool_pose(tool)
            lifted=p[2]-self.initial_tool_positions[tool][2]>.040
            self.lift_ticks[tool]=self.lift_ticks[tool]+1 if lifted and all(pair) else 0
            if not self.grasp_latches[tool] and self.lift_ticks[tool]>=24:
                self.pickup_verified[tool]=True
                if tool=='pen':self.pen_verified[self.active_pen]=True
                event={'tick':self.tick,'tool':tool,'color_id':self.active_pen if tool=='pen' else 0,'two_finger_contact':True,
                       'lift_m':float(p[2]-self.initial_tool_positions[tool][2]),
                       'grip_contact_ticks':self.grasp_contact_ticks[tool], 'consecutive_lift_ticks':self.lift_ticks[tool],
                       'tool_position_m':p.tolist(),'attachment_joint_created_before_lift':False,
                       'stabilization_joint_created_after_lift':True}
                self._stabilize_verified_grasp(tool)
                self.pickup_events.append(event)
                print('CP09_PHYSICAL_PICKUP',event,flush=True)

    def _contact_report(self,headers,data):
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools
        if hasattr(self,'robot_motion_audit'):self.robot_motion_audit.contacts(headers,data)
        for h in headers:
            pair=tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0,h.collider1)))
            if not any('/World/CP09/' in p for p in pair):continue
            points=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
            impulse=max((math.hypot(*p.impulse) for p in points),default=0.)
            separation=min((float(p.separation) for p in points),default=1.)
            active=h.type!=ContactEventType.CONTACT_LOST
            self.contact_pairs[pair]=(active,impulse,separation)
            self.contact_pair_last_report_tick[pair]=self.tick

    def _tool_pose(self,tool):
        p,q=self.tools[tool].get_world_pose()
        return np.asarray(p),Rot.from_quat([*q[1:],q[0]]).as_matrix()

    def _boards(self):
        fraction=shutter_fraction(self.tick)
        y=SHUTTER_OPEN_Y+(SHUTTER_CLOSED_Y-SHUTTER_OPEN_Y)*fraction
        self.w['roots']['Shutter'].Set(Gf.Vec3d(0,float(y),0))
        if self.tick>=ANNOTATION_TICK and not self.annotation_applied:
            if fraction!=1.:raise RuntimeError('Annotations require a completely closed shutter')
            before=hashlib.sha256(self.surface.ink.tobytes()).hexdigest()
            self.surface.ink[:]=self.w['work']
            self.annotation_applied=True
            self.presentation_events.append(dict(tick=self.tick,event='same_board_annotated_under_closed_shutter',
                board_prim='/World/CP09/Board',ink_prim='/World/CP09/Board/Visuals/ActualInk',
                shutter_closed=True,original_ink_sha256=before,
                annotated_ink_sha256=hashlib.sha256(self.surface.ink.tobytes()).hexdigest(),
                same_raster_object=id(self.surface.ink)==self.surface_identity))
        if self.tick<MEMORY_END:self.phase='memorize'
        elif self.tick<REVEAL_END:self.phase='occlude'
        else:
            r=self.board_support.retrieval_observation()
            self.phase='restore' if r['placed'] and r['machine_retracted'] else 'retrieve'
        self.board_center_xy=np.asarray(self.board_support.position()[:2]).copy()

    def same_board_evidence(self):
        stage=self.stage_usd
        faces=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/CP09/') and p.GetName()=='BoardFace']
        identity=(len(faces)==1 and stage.GetPrimAtPath('/World/CP09/Board').IsValid()
                  and stage.GetPrimAtPath('/World/CP09/Board/Visuals/ActualInk').IsValid()
                  and not stage.GetPrimAtPath('/World/CP09/Work').IsValid()
                  and not stage.GetPrimAtPath('/World/CP09/Reference').IsValid()
                  and id(self.surface.ink)==self.surface_identity)
        success=(identity and len(self.presentation_events)==1
                 and self.presentation_events[0]['original_ink_sha256']==self.initial_ink_sha256
                 and self.presentation_events[0]['shutter_closed'])
        return dict(success=bool(success),board_count=len(faces),visible_board_faces=faces,board_prim='/World/CP09/Board',
            ink_prim='/World/CP09/Board/Visuals/ActualInk',same_physical_and_visual_board=True,
            initial_ink_sha256=self.initial_ink_sha256,events=self.presentation_events)

    def tool_return_status(self):
        result={}
        for tool in ('eraser','pen'):
            p,_=self._tool_pose(tool)
            prefixes=[self.tool_root(tool)+'/']+([f'/World/CP09/Cap{self.active_pen}/'] if tool=='pen' and self.caps.enabled else [])
            supported=any(any(any(q.startswith(prefix) for prefix in prefixes) for q in pair)
                and any('/World/CP09/ToolTray' in q or '/World/CP09/PenSocket' in q for q in pair)
                and active and (impulse>1e-8 or sep<.0002)
                for pair,(active,impulse,sep) in self.contact_pairs.items())
            result[tool]=bool(not self.grasp_latches[tool] and np.linalg.norm(p-self.initial_tool_positions[tool])<.02 and supported)
        return result
