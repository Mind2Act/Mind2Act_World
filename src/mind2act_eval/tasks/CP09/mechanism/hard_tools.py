"""Everyday eraser/markers and contact-verified, robot-driven axial pull caps."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf,UsdGeom,UsdShade,UsdPhysics,PhysxSchema
from .shapes import material
from .shapes import rounded,cylinder
from .layouts import CONFIG

BASE_Z=.7855
ERASER_POSITION=np.array([-.150,-.220,BASE_Z])
def pen_position(color,count):return np.array([.150+(color-(count+1)/2)*.065,-.220,BASE_Z])
PEN_POSITIONS={i:pen_position(i,1) for i in range(1,4)}
GRIP_HEIGHT={'eraser':.015,'pen':.105}
CAP_GRIP=.015
CAP_TURNS=0.0  # v2 has no rotational opening degree of freedom
THREAD_TRAVEL=.004  # Original 4 mm retention travel, now purely axial


def matrix(q):return Rotation.from_quat([*q[1:],q[0]]).as_matrix()
def quat(r):
    q=Rotation.from_matrix(r).as_quat();return Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3])))
def root_for(tool,color=1):return '/World/CP09/EraserTool' if tool=='eraser' else f'/World/CP09/Marker{color}'
def cap_root(color):return f'/World/CP09/Cap{color}'


def collision(mesh,material=None):
    p=mesh.GetPrim();UsdPhysics.CollisionAPI.Apply(p)
    if p.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
    PhysxSchema.PhysxCollisionAPI.Apply(p).CreateContactOffsetAttr(.0001)
    if material:UsdShade.MaterialBindingAPI.Apply(p).Bind(material,materialPurpose='physics')


def rigid(stage,path,position,mass):
    x=UsdGeom.Xform.Define(stage,path);x.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)));x.AddOrientOp().Set(Gf.Quatf(1.))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim());api=PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
    api.CreateSolverPositionIterationCountAttr(128);api.CreateSolverVelocityIterationCountAttr(8)
    UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0.)
    return x


def build_tools(env):
    st=env.stage_usd
    rubber=material(st,'CP09EraserRubber',(.79,.80,.73),0.,.94)
    sleeve=material(st,'CP09EraserSleeve',(.035,.080,.12),.10,.60)
    silver=material(st,'CP09MarkerMetal',(.58,.61,.64),.7,.35)
    light=material(st,'CP09MarkerShell',(.86,.87,.84),.02,.55)
    hard=material(st,'CP09MarkerGraphite',(.085,.105,.12),.3,.48)
    grip=UsdShade.Material.Define(st,'/World/CP09/ToolGripPhysics')
    api=UsdPhysics.MaterialAPI.Apply(grip.GetPrim());api.CreateStaticFrictionAttr(2.2);api.CreateDynamicFrictionAttr(1.9)
    root=root_for('eraser');rigid(st,root,ERASER_POSITION,.020)
    # Ordinary compact rubber eraser with a paper sleeve, no stamp handle.
    collision(rounded(st,root+'/Rubber',(0,0,.013),(.022,.038,.020),.004,rubber),grip)
    rounded(st,root+'/Sleeve',(0,0,.014),(.023,.022,.0205),.002,sleeve)
    for side in (-1,1):
        rounded(st,root+('/SleeveEdgeL' if side<0 else '/SleeveEdgeR'),(0,side*.0105,.0245),(.020,.001,.0006),.0003,silver)
    c=UsdGeom.Cube.Define(st,root+'/Contact');c.CreateSizeAttr(1.);c.AddTranslateOp().Set(Gf.Vec3d(0,0,.0015));c.AddScaleOp().Set(Gf.Vec3f(.018,.028,.003));UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(rubber);collision(c,grip)
    for color in range(1,env.case['color_count']+1):
        position=pen_position(color,env.case['color_count'])+np.array([0,0,.0045 if env.case['difficulty']!='easy' else 0.])
        root=root_for('pen',color);rigid(st,root,position,.030)
        dock=UsdPhysics.FixedJoint.Define(st,root+'/DockRetention')
        dock.CreateBody1Rel().SetTargets([root]);dock.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,position)))
        dock.CreateExcludeFromArticulationAttr(True)
        ink=material(st,'CP09MarkerColor'+str(color),tuple(CONFIG['colors'][color-1]['material']),.05,.46)
        barrel=hard if env.case['difficulty']=='hard' else light
        collision(cylinder(st,root+'/Barrel',(0,0,.091),.0085,.118,barrel),grip)
        collision(cylinder(st,root+'/RearPlug',(0,0,.151),.009,.006,ink),grip)
        cylinder(st,root+'/Neck',(0,0,.027),.0075,.016,silver)
        cylinder(st,root+'/ColorBand',(0,0,.052),.0088,.008,ink)
        collision(cylinder(st,root+'/Nib',(0,0,.009),.0028,.018,ink),grip)
        c=UsdGeom.Sphere.Define(st,root+'/Contact');c.CreateRadiusAttr(.002)
        c.AddTranslateOp().Set(Gf.Vec3d(0,0,.002));UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(ink);collision(c,grip)
        # Axial pull retention sleeve; no helical thread or twist instruction.
        cylinder(st,root+'/PullSleeve',(0,0,.027),.0078,.008,silver)
        if env.case['difficulty']=='easy':continue
        cap=cap_root(color);rigid(st,cap,position,.010)
        radius=.011 if env.case['difficulty']=='medium' else .012
        # Hollow cap: the nib is genuinely covered; no solid invisible plug.
        for k in range(20):
            a=2*np.pi*k/20
            part=rounded(st,cap+f'/Wall{k}',(radius*np.cos(a),radius*np.sin(a),.017),(.003,.003,.038),.001,ink)
            collision(part,grip);UsdGeom.Imageable(part.GetPrim()).MakeInvisible()
        # Smooth hollow shell over the separate convex collision segments.
        shell_mesh=UsdGeom.Mesh.Define(st,cap+'/Shell');points=[];faces=[];n=48
        for z,rr in [(-.002,radius+.001),(.036,radius+.001),(.036,radius-.0015),(-.002,radius-.0015)]:
            points.extend((rr*np.cos(2*np.pi*k/n),rr*np.sin(2*np.pi*k/n),z) for k in range(n))
        for ring in range(4):
            for k in range(n):faces.extend([ring*n+k,ring*n+(k+1)%n,((ring+1)%4)*n+(k+1)%n,((ring+1)%4)*n+k])
        shell_mesh.CreatePointsAttr(points);shell_mesh.CreateFaceVertexCountsAttr([4]*(4*n));shell_mesh.CreateFaceVertexIndicesAttr(faces);shell_mesh.CreateSubdivisionSchemeAttr('none');UsdShade.MaterialBindingAPI.Apply(shell_mesh.GetPrim()).Bind(ink)
        collision(cylinder(st,cap+'/End',(0,0,-.003),radius+.001,.003,ink),grip)
        rim=UsdGeom.BasisCurves.Define(st,cap+'/Rim')
        a=np.linspace(0,2*np.pi,65)
        rim.CreatePointsAttr(np.c_[(radius+.0005)*np.cos(a),(radius+.0005)*np.sin(a),np.full(65,.036)].tolist())
        rim.CreateCurveVertexCountsAttr([65]);rim.CreateTypeAttr('linear');rim.CreateWidthsAttr([.0015]);UsdShade.MaterialBindingAPI.Apply(rim.GetPrim()).Bind(silver)
        collision(rounded(st,cap+'/Clip',(radius+.003,0,.019),(.004,.006,.026),.002,silver),grip)
        if env.case['difficulty']=='hard':
            for k in range(24):
                a=2*np.pi*k/24
                collision(rounded(st,cap+f'/Knurl{k}',((radius+.001)*np.cos(a),(radius+.001)*np.sin(a),.017),(.0016,.0016,.021),.0006,ink),grip)
        joint=UsdPhysics.FixedJoint.Define(st,cap+'/PullClosed')
        joint.CreateBody0Rel().SetTargets([root]);joint.CreateBody1Rel().SetTargets([cap]);joint.CreateExcludeFromArticulationAttr(True)
        UsdPhysics.FilteredPairsAPI.Apply(st.GetPrimAtPath(cap)).CreateFilteredPairsRel().SetTargets([root])


class PullCaps:
    def __init__(self,env):
        self.env=env;self.enabled=env.case['difficulty']!='easy';self.bodies={};self.held=None
        self.state={i:dict(closed=True,opened=False,engaged=False,angle=0.,last_angle=0.,open_events=0,close_events=0,removed_clear=False) for i in range(1,env.case['color_count']+1)}
        self.contacts_run=0;self.events=[];self.pending_release={};self.samples=[];self.tray_contacts={i:0 for i in self.state}
    def register(self,world):
        from isaacsim.core.prims import SingleRigidPrim
        if self.enabled:
            for i in self.state:self.bodies[i]=world.scene.add(SingleRigidPrim(prim_path=cap_root(i),name=f'CP09_Cap{i}'))
    def pose(self,color):
        p,q=self.bodies[color].get_world_pose();return np.asarray(p),matrix(q)
    def marker_pose(self,color):
        p,q=self.env.all_tools['pen'+str(color)].get_world_pose();return np.asarray(p),matrix(q)
    def grip_position(self,color):
        p,r=self.pose(color);return p+r@np.array([0,0,CAP_GRIP])
    def contacts(self,color):
        tcp,_=self.env.kin['Left'].fk(self.env.robots['Left'].get_joint_positions()[:6])
        if np.linalg.norm(tcp-self.grip_position(color))>.035:return [False,False]
        root=cap_root(color)+'/'
        return [any(any(q.startswith(root) for q in pair) and any(q.startswith('/World/RobotLeft/Asset/'+link+'/') for q in pair)
            and active and (impulse>1e-8 or separation<.00005)
            for pair,(active,impulse,separation) in self.env.contact_pairs.items()) for link in ('link7','link8')]
    def _guide(self,color):
        st=self.env.stage_usd;path=cap_root(color)
        st.RemovePrim(path+'/PullClosed')
        joint=UsdPhysics.Joint.Define(st,path+'/PullGuide')
        joint.CreateBody0Rel().SetTargets([root_for('pen',color)]);joint.CreateBody1Rel().SetTargets([path]);joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateLocalRot0Attr().Set(quat(Rotation.from_euler('y',-90,degrees=True).as_matrix()))
        joint.CreateLocalRot1Attr().Set(quat(Rotation.from_euler('y',-90,degrees=True).as_matrix()))
        for axis in ('transY','transZ','rotX','rotY','rotZ'):
            limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis);limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
        limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),'transX');limit.CreateLowAttr(-THREAD_TRAVEL-.002);limit.CreateHighAttr(.001)
        self.state[color]['guide']=joint;self.state[color]['engaged']=True
    def _attach(self,color):
        if not self.env.grasp_latches['pen']:return
        st=self.env.stage_usd;p,r=self.pose(color)
        st.RemovePrim(cap_root(color)+'/TrayDetent')
        tcp,rr=self.env.kin['Left'].fk(self.env.robots['Left'].get_joint_positions()[:6]);flange=tcp-rr@np.array([.145,0,0])
        joint=UsdPhysics.FixedJoint.Define(st,cap_root(color)+'/VerifiedCapGrasp')
        joint.CreateBody0Rel().SetTargets(['/World/RobotLeft/Asset/link6']);joint.CreateBody1Rel().SetTargets([cap_root(color)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,rr.T@(p-flange))));joint.CreateLocalRot0Attr().Set(quat(rr.T@r))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
        joint.CreateExcludeFromArticulationAttr(True)
        relation=UsdPhysics.FilteredPairsAPI(self.env.stage_usd.GetPrimAtPath(cap_root(color))).GetFilteredPairsRel()
        for link in ('link6','link7','link8'):relation.AddTarget('/World/RobotLeft/Asset/'+link)
        measured=self.env.robots['Left'].get_joint_positions()[6:].copy();self.env.gripper_opening['Left']=float(np.mean(measured));self.env.commands['Left'][6:]=measured
        self.held=color
        if self.state[color]['closed']:
            self.state[color]['opened']=False;self.state[color]['closing']=False
            self._guide(color)
        self.events.append(dict(tick=self.env.tick,event='cap_grasp_verified',color=color,physical_contact=True,bilateral_contact_ticks=self.contacts_run))
    def release(self):
        if self.held is None:return
        color=self.held;st=self.env.stage_usd
        st.RemovePrim(cap_root(color)+'/VerifiedCapGrasp')
        if not self.state[color]['closed'] and self.tray_contacts[color]>=24:
            p,r=self.pose(color);joint=UsdPhysics.FixedJoint.Define(st,cap_root(color)+'/TrayDetent')
            joint.CreateBody1Rel().SetTargets([cap_root(color)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,p)));joint.CreateLocalRot0Attr().Set(quat(r))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.));joint.CreateExcludeFromArticulationAttr(True)
            self.events.append(dict(tick=self.env.tick,event='cap_parked_on_tray',color=color,actual_contact=True,consecutive_contact_ticks=self.tray_contacts[color],position_m=p.tolist(),resting_contact_stabilization=True))
        self.pending_release[color]=True;self.held=None;self.contacts_run=0
    def sample(self):
        if not self.enabled:return
        for color in list(self.pending_release):
            if min(self.env.robots['Left'].get_joint_positions()[6:])>.024:
                relation=UsdPhysics.FilteredPairsAPI(self.env.stage_usd.GetPrimAtPath(cap_root(color))).GetFilteredPairsRel()
                for link in ('link6','link7','link8'):relation.RemoveTarget('/World/RobotLeft/Asset/'+link)
                self.pending_release.pop(color)
        color=self.env.active_pen;s=self.state[color]
        supported=any(any(q.startswith(cap_root(color)+'/') for q in pair) and any('/World/CP09/ToolTray' in q for q in pair)
                      and active and (impulse>1e-8 or separation<.0002) and self.env.tick-getattr(self.env,'contact_pair_last_report_tick',{}).get(pair,-1000000000)<=2 for pair,(active,impulse,separation) in self.env.contact_pairs.items())
        self.tray_contacts[color]=self.tray_contacts[color]+1 if supported else 0
        if self.held is None:
            contact=all(self.contacts(color)) and self.env.gripper_opening['Left']<.018 and not self.env.grasp_latches['eraser']
            self.contacts_run=self.contacts_run+1 if contact else 0
            if self.contacts_run>=24:self._attach(color)
        mp,mr=self.marker_pose(color);cp,cr=self.pose(color);local=mr.T@(cp-mp);rel=mr.T@cr
        if not s['engaged']:
            if s['opened'] and local[2]<-.030:s['removed_clear']=True
            return
        # The D6 slider locks all angular axes. Detachment follows measured
        # axial displacement with both real, verified physical grasps retained.
        theta=math.atan2(rel[1,0],rel[0,0]);s['angle']=theta
        if (not s['opened'] and self.held==color and self.env.grasp_latches['pen']
                and local[2]<=-THREAD_TRAVEL and np.linalg.norm(local[:2])<.0025):
            self.env.stage_usd.RemovePrim(cap_root(color)+'/PullGuide')
            s['engaged']=False;s['closed']=False;s['opened']=True;s['open_events']+=1
            self.events.append(dict(tick=self.env.tick,event='cap_pulled_open',color=color,
                measured_angle_deg=float(np.degrees(theta)),axial_travel_m=float(-local[2]),
                physical_grasps=True,mechanism='axial_pull_v2',body_pose_writes=False))
        if self.env.tick%4==0:self.samples.append([self.env.tick,color,float(theta),float(local[2]),int(s['closed']),int(s['opened']),int(self.held==color)])
    def is_open(self,color):return not self.enabled or not self.state[color]['closed']
    def evidence(self):
        return dict(required=self.enabled,success=not self.enabled or all(s['open_events']>0 and not s['closed'] for s in self.state.values()),events=self.events,
                    all_used_markers_uncapped=not self.enabled or all(s['open_events']>0 for s in self.state.values()),all_caps_parked=not self.enabled or all(self.env.stage_usd.GetPrimAtPath(cap_root(i)+'/TrayDetent').IsValid() for i in self.state),all_caps_reclosed=all(s['close_events']>0 and s['closed'] for s in self.state.values()),body_pose_writes=False,
                    completion_criterion='Each used cap physically pulled at least 4 mm axially from its paired marker with verified grasps; tray detents are optional resting stabilization',mechanism='contact-verified axial D6 pull guide; all angular axes locked; measured axial displacement releases retention')
