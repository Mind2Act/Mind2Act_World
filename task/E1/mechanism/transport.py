"""Kinematic-carrier transport with independent dynamic items.

Source: E-142 baseline engine/simulation/scripts/physco_dynamic_transport.py (verbatim, except that
`settle` receives hold/step callables instead of an Isaac World and robot articulations, so robot
control stays in the shared robot module).
The transport also retains its parent tensor simulation view after scene construction.
"""
import json
from pathlib import Path
import numpy as np


def build_transport(stage, manifest, plan, scene, intervals):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    from scipy.spatial.transform import Rotation
    layout = manifest['case']['layout']
    origin = np.array([0., layout['carrier_center'][1], layout['carrier_center'][2]])
    initial_x = np.array([intervals[i][0]-.00003-i*1.15 for i in range(len(manifest['rounds']))])
    arm = next(r for r in manifest['station']['robots'] if r['name'] == plan['robot_arm'])
    base = np.asarray(arm['position']); rotation = Rotation.from_euler('xyz', arm['euler_deg'], degrees=True).as_matrix()
    mat = UsdShade.Material.Define(stage, '/World/CarrierPhysicsMaterial')
    material = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    material.CreateStaticFrictionAttr(.9); material.CreateDynamicFrictionAttr(.8); material.CreateRestitutionAttr(0.)
    paths = []
    submission_floors = []
    for rn, batch in enumerate(manifest['rounds']):
        root_path = '/World/Task/Batch'+str(rn)
        root = stage.GetPrimAtPath(root_path)
        carrier = UsdGeom.Xform.Define(stage, '/World/Carriers/Batch'+str(rn))
        carrier.AddTranslateOp().Set(Gf.Vec3d(*(origin+[initial_x[rn],0,0])))
        xf = UsdGeom.Xformable(root)
        xf.GetOrderedXformOps()[0].Set(Gf.Vec3d(*origin))
        UsdGeom.Xform.Define(stage, '/World/Items/Batch'+str(rn))
        item_paths = []
        for item in batch['items']:
            old_path = item['prim_path']
            matrix = UsdGeom.Xformable(stage.GetPrimAtPath(old_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            matrix.SetTranslateOnly(matrix.ExtractTranslation()+Gf.Vec3d(float(initial_x[rn]), 0., .004))
            new_path = '/World/Items/Batch'+str(rn)+'/'+old_path.rsplit('/', 1)[-1]
            Sdf.CopySpec(stage.GetRootLayer(), old_path, stage.GetRootLayer(), new_path)
            new = stage.GetPrimAtPath(new_path)
            transform = UsdGeom.Xformable(new); transform.ClearXformOpOrder(); transform.AddTransformOp(opSuffix='world').Set(matrix)
            item['source_prim_path'] = old_path; item['prim_path'] = new_path
            item_paths.append(new_path); stage.RemovePrim(old_path)
        paths.append(item_paths)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI): prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdGeom.Mesh): UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
        submission_floor = None
        for name, b in plan['collision_world'].items():
            if name == 'belt_surface' or (rn == 0 and name.startswith('submit')): continue
            center = base+rotation@np.asarray(b['pose'][:3])-origin
            dims = np.asarray(b['dims']).copy()
            if name == 'carrier' and rn == 0:
                dims[:2] = layout['observation_carrier_size_xy']; center[0] = layout['source_center_xy'][0]
            cube = UsdGeom.Cube.Define(stage, str(carrier.GetPath())+'/Collision/'+name.replace('-', '_minus_'))
            if name == 'submit_floor':
                submission_floor = str(cube.GetPath())
            cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center)); cube.AddScaleOp().Set(Gf.Vec3f(*dims))
            UsdGeom.Imageable(cube).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, materialPurpose='physics')
        if rn > 0 and submission_floor is None:
            raise ValueError('E1 answer batch is missing its submission floor')
        submission_floors.append(submission_floor)
        rb = UsdPhysics.RigidBodyAPI.Apply(carrier.GetPrim()); rb.CreateRigidBodyEnabledAttr(True); rb.CreateKinematicEnabledAttr(True)
        xf.GetOrderedXformOps()[0].Set(Gf.Vec3d(*(origin+[initial_x[rn], 0, 0])))
        UsdGeom.Imageable(root).MakeInvisible()
        UsdGeom.Imageable(stage.GetPrimAtPath('/World/Items/Batch'+str(rn))).MakeInvisible()
    return DynamicTransport(stage, manifest, intervals, initial_x, paths, submission_floors)


class DynamicTransport:
    def __init__(self, stage, manifest, intervals, initial_x, paths, submission_floors):
        self.submission_floors = submission_floors
        self.stage=stage; self.manifest=manifest; self.intervals=intervals
        self.initial_x=initial_x; self.carrier_x=initial_x.tolist(); self.paths=paths
        self.motions={}; self.retired={}; self.visible=set(); self.samples=[]; self.transport_audits=[]
        self.phase=None; self.phase_ref=None; self.phase_worst_xy=0.; self.phase_worst_z=0.
        self.one=np.array([0], dtype=np.int32)

    def initialize(self, sim):
        # The scene builder's local SimulationView must outlive the child views
        # used by the transport after build returns and cameras become ready.
        self._simulation_view = sim
        self.carriers=[sim.create_rigid_body_view('/World/Carriers/Batch'+str(i)) for i in range(len(self.paths))]
        self.item_views=[sim.create_rigid_body_view('/World/Items/Batch'+str(i)+'/Item*') for i in range(len(self.paths))]
        self.targets=[v.get_transforms().copy() for v in self.carriers]
        from pxr import UsdGeom
        self.visual_ops=[UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Task/Batch'+str(i))).GetOrderedXformOps()[0] for i in range(len(self.paths))]
        assert all(v.count == len(paths) for v,paths in zip(self.item_views,self.paths))

    def show(self, rn):
        from pxr import UsdGeom
        for path in ['/World/Task/Batch','/World/Items/Batch']:
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path+str(rn))).MakeVisible()
        self.visible.add(rn)

    def hide(self, rn):
        from pxr import UsdGeom
        for path in ['/World/Task/Batch','/World/Items/Batch']:
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path+str(rn))).MakeInvisible()
        self.visible.discard(rn)

    def before_step(self, t):
        for i in range(len(self.carriers)):
            x=self.carrier_x[i]
            if i in self.motions:
                start, end, t0, duration=self.motions[i]
                u=float(np.clip((t-t0)/duration,0,1))
                x=start+(end-start)*(10*u**3-15*u**4+6*u**5)
            if i in self.retired:
                start,t0=self.retired[i]; elapsed=max(0,t-t0); ramp=2.
                x=start+(.38*elapsed**2/(2*ramp) if elapsed < ramp else .38*(elapsed-ramp/2))
            self.targets[i][0,0]=x
            self.carriers[i].set_kinematic_targets(self.targets[i],self.one)
            self.carrier_x[i]=float(x)

    def start_motion(self, index, start, end, time, speed):
        distance=abs(end-start)
        duration=max(1.875*distance/speed,np.sqrt(5.774*distance/.6),1/120)
        duration=np.ceil(duration*120)/120
        self.motions[index]=(start,end,time,duration)
        return duration

    def advance_queue(self, index, time):
        # Off-screen batches advance while the preceding batch is observed/handled.
        # At least a carrier length separates neighbors throughout.
        for j in range(index+1,len(self.paths)):
            end=self.intervals[j][0]-.00003-(j-index-1)*1.15
            self.start_motion(j,self.carrier_x[j],end,time,.30)

    def retire(self, index, time):
        self.retired[index]=(self.carrier_x[index],time)

    def settle(self, hold_rest, step):
        window=[]; excursion=float('inf'); angle=float('inf')
        for n in range(1200):
            hold_rest()
            self.before_step(0); step()
            current=np.concatenate([v.get_transforms().copy() for v in self.item_views])
            window.append(current); window=window[-61:]
            if len(window)==61:
                poses=np.asarray(window)
                excursion=float(np.linalg.norm(poses[:,:,:3]-poses[0,:,:3],axis=-1).max())
                q=poses[:,:,3:].astype(float); q/=np.linalg.norm(q,axis=-1,keepdims=True)
                angle=float(np.degrees(2*np.arccos(np.clip(abs(np.sum(q*q[0],axis=-1)),0,1))).max())
                if excursion<.0005 and angle<.5: break
        self.settle_report={'steps':n+1,'window_seconds':.5,'max_position_excursion_m':excursion,'max_angle_excursion_deg':angle,'pass':bool(excursion<.0005 and angle<.5)}
        print('TRANSPORT_SETTLE',json.dumps(self.settle_report),flush=True)
        if not self.settle_report['pass']: raise RuntimeError('Out-of-view physical settling did not converge; refusing demo recording')

    def after_step(self,t,skill):
        from pxr import Gf
        # Graphics only: mirror the measured rigid-body pose, never a scripted item pose.
        for view,op in zip(self.carriers,self.visual_ops):
            op.Set(Gf.Vec3d(*map(float,view.get_transforms()[0,:3])))
        is_transport=skill.endswith('_enter') or skill.endswith('_exit')
        if is_transport:
            index=int(skill.split('_')[0][5:])-1
            poses=self.item_views[index].get_transforms().copy()
            carrier=self.carriers[index].get_transforms()[0].copy()
            relative=poses[:,:3]-carrier[:3]
            if self.phase!=skill:
                self.flush_phase(); self.phase=skill; self.phase_ref=relative
            delta=relative-self.phase_ref
            self.phase_worst_xy=max(self.phase_worst_xy,float(np.linalg.norm(delta[:,:2],axis=-1).max()))
            self.phase_worst_z=max(self.phase_worst_z,float(abs(delta[:,2]).max()))
        else:
            self.flush_phase()
        if round(t*120)%10==0:
            self.samples.append(np.r_[t, self.carrier_x, np.concatenate([v.get_transforms().ravel().copy() for v in self.item_views])])

    def flush_phase(self):
        if self.phase is not None:
            self.transport_audits.append({'phase':self.phase,'worst_relative_xy_m':self.phase_worst_xy,'worst_relative_z_m':self.phase_worst_z,'pass':bool(self.phase_worst_xy<.015 and self.phase_worst_z<.005)})
        self.phase=None; self.phase_ref=None; self.phase_worst_xy=0.;self.phase_worst_z=0.

    def finish(self,output):
        self.flush_phase()
        np.savez_compressed(output/'transport_states.npz',states=np.asarray(self.samples))
        report={'settle':self.settle_report,'phases':self.transport_audits,'measured_item_pose_writes':0,'item_mode_changes':0,'pass':bool(self.settle_report['pass'] and all(p['pass'] for p in self.transport_audits)), 'mechanism':'kinematic carrier + independent dynamic objects; contact/friction; no parent animation or attachment'}
        (output/'transport_report.json').write_text(json.dumps(report,indent=2))
        return report
