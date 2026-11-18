"""Original free surface FEM cloth on shared World/X5; no task controller."""
from collections import deque
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull
from mind2act_eval.contracts import Health,Observation
from .instances import validate
from .source_admission import validate_case_source
from .evidence import Recorder,rotation


def collision_hulls(stage):
    from pxr import Usd,UsdGeom,UsdPhysics
    out={}
    for side in ('Left','Right'):
        for link in ('link7','link8'):
            key=side+'/'+link;root=stage.GetPrimAtPath('/World/Robot'+side+'/Asset/'+link);values=[]
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    matrix=np.asarray(UsdGeom.XformCache().ComputeRelativeTransform(prim,root)[0])
                    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)@matrix[:3,:3]+matrix[3,:3]
                    values.append(np.unique(np.round(ConvexHull(points).equations,8),axis=0).tolist())
            if not values:raise ValueError('PR4 missing original finger collision geometry '+key)
            out[key]=values
    return out


def _body_views(stage):
    from mind2act_eval.robot.read_only import robot_body_readers
    keys = {s+'/'+link: 'PR4_'+s+'_'+link for s in ('Left','Right')
            for link in ('link6','link7','link8')}
    readers = robot_body_readers(stage, {name: '/World/Robot'+key.split('/')[0]+'/Asset/'+key.split('/')[1]
                                       for key, name in keys.items()})
    for view in readers.values():
        view.initialize()
    return {key: readers[name] for key, name in keys.items()}


def _record_fem_binding_failure(stage,body_path,fem,material,directory,world):
    """Read the failed native binding without changing the source or its guard."""
    def json_value(value):
        if isinstance(value,(np.ndarray,np.generic)):value=value.tolist()
        if isinstance(value,float) and not np.isfinite(value):return str(value)
        if isinstance(value,list):return [json_value(v) for v in value]
        if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
        return value
    def read(getter):
        try:return dict(value=json_value(getter()))
        except Exception as error:return dict(error=type(error).__name__+': '+str(error))
    def source_binding(path):
        prim=stage.GetPrimAtPath(path)
        return dict(exists=bool(prim),apiSchemas=str(prim.GetMetadata('apiSchemas')),
            registered_apis=prim.GetAppliedSchemas(),
            attributes={a.GetName():a.Get() for a in prim.GetAuthoredAttributes()
                        if a.GetName() in ('omniphysics:youngsModulus','omniphysics:dynamicFriction')},
            relationships={r.GetName():[str(v) for v in r.GetTargets()] for r in prim.GetRelationships()})
    def current_stage():
        import omni.usd
        context=omni.usd.get_context()
        return dict(stage_id=context.get_stage_id(),matches_task_stage=context.get_stage()==stage)
    readbacks={name:read(getter) for name,getter in (
        ('fem_count',lambda:fem.count),
        ('max_simulation_nodes_per_body',lambda:fem.max_simulation_nodes_per_body),
        ('material_count',lambda:material.count))}
    for name,getter in (('youngs_modulus',material.get_youngs_modulus),
                        ('dynamic_friction',material.get_dynamic_friction)):
        readbacks[name]=(read(getter) if readbacks['material_count'].get('value')==1
                         else dict(unavailable='material_count is not 1'))
    from mind2act_eval.sim.physics_profile import runtime_diagnostics
    data=dict(schema='mind2act_eval.pr4_native_fem_binding_failure.v1',
        body_path=body_path,material_path='/World/ClothMaterial',
        task_stage_root=stage.GetRootLayer().identifier,current_stage=read(current_stage),
        physics_after_source_settle=read(lambda:runtime_diagnostics(world,stage)),
        body_source=read(lambda:source_binding(body_path)),
        material_source=read(lambda:source_binding('/World/ClothMaterial')),readbacks=readbacks)
    with (directory/'native-fem-binding-failure.json').open('x') as stream:
        json.dump(data,stream,allow_nan=False,indent=2)


class Contacts:
    def __init__(self):self.rows={}
    def receive(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            pair=tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0,h.collider1)))
            points=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
            if points:self.rows[pair]=np.sum([[float(p.impulse[j]) for j in range(3)] for p in points],axis=0).tolist()
    def clear(self):self.rows.clear()
    def snapshot(self):return [dict(pair=list(pair),impulse=list(v)) for pair,v in sorted(self.rows.items())]


class ClothScene:
    def __init__(self,spec,ctx,mesh,initial,settled,triangles,hulls,views,articulations,fem,contacts,subscription,settle_receipt,support_geometry):
        self.spec,self.ctx,self.mesh=spec,ctx,mesh
        self.views,self.articulations,self.fem=views,articulations,fem
        self.contacts,self.subscription=contacts,subscription
        self.settle_receipt=settle_receipt
        self.support_geometry=support_geometry
        self.recorder=Recorder(spec.instance_id,spec.config['source_case'],initial,settled,triangles,hulls,ctx.output_dir/'PR4',support_geometry=support_geometry)
        try:self._sample(0)
        except BaseException:self.recorder.close();raise

    def points(self):
        from pxr import UsdGeom
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.mesh.GetPrim()))
        return np.asarray(self.mesh.GetPointsAttr().Get(),np.float32)@matrix[:3,:3]+matrix[3,:3]

    def _sample(self,tick):
        points=self.points()
        # Isaac 5.1's source surface view reports zero tensor node capacity even
        # in the successful collection replays. The source reads its single
        # simulation mesh from USD and compares tensors only when exposed.
        error=None
        if self.fem.max_simulation_nodes_per_body>=6561:
            physical=np.asarray(self.fem.get_simulation_nodal_positions())[0,:6561]
            if physical.shape!=(6561,3):raise ValueError('PR4 missing native FEM simulation nodes')
            error=float(np.linalg.norm(physical-points,axis=1).max())
        fingers={};tcps={}
        for key,body in self.views.items():
            p,q=body.get_world_pose();p,q=np.asarray(p),np.asarray(q)
            if key.endswith('link6'):tcps[key.split('/')[0]]=(p+rotation(q)@np.array([.145,0.,0.])).tolist()
            else:fingers[key]=dict(position=p.tolist(),quaternion=q.tolist())
        raw=dict(tick=tick,time_s=float(self.ctx.world.current_time),joints={s:np.asarray(a.get_joint_positions()).tolist() for s,a in self.articulations.items()},
            tcp=tcps,finger_poses=fingers,rigid_contact_impulses_ns=self.contacts.snapshot(),
            point_source='native_surface_simulation_mesh_usd',render_physics_error_m=error)
        self.recorder.record(raw,points)
        self._last_shape_points = np.asarray(points, dtype='<f4').copy()

    def before_step(self,tick):self.contacts.clear()
    def after_step(self,tick):self._sample(tick)
    def observe(self,tick):
        return Observation(tick=tick,sim_time_s=tick/120,instruction=self.spec.config['instruction'],
            protocol_version=self.spec.protocol_version,**dict(self.ctx.robot.observe()))
    def health(self):return Health(True)
    def evidence(self,tick):return self.recorder.evidence(tick)
    def is_success(self):
        """Check the live measured mesh in the fixed target, without replay I/O."""
        if not hasattr(self, '_last_shape_points'):
            return False
        if getattr(self, '_success_check_rows', None) != self.recorder.rows:
            from .mechanism.terminal_shape import terminal_shape
            case = self.recorder.case
            origin = case['origin']
            shape = terminal_shape(self._last_shape_points, self.recorder.triangles,
                                   case['level'], origin['fold_target_center'], origin['width'])
            from .mechanism.support import support_witness
            support=support_witness(self._last_shape_points,self.support_geometry)
            self._success_at_tick = bool(shape['shape_success'] and support['supported'])
            self._success_check_rows = self.recorder.rows
        return self._success_at_tick
    def is_complete(self):return self.is_success()
    def result_metadata(self):
        return dict(source_id=self.spec.source_id,source_spec_sha256=self.spec.config['source_spec_sha256'],
            source_config_sha256=self.spec.config['source_config_sha256'],portable_config_sha256=self.spec.config['portable_config_sha256'],
            scene_sha256=self.spec.config['scene_sha256'],source_initialization=self.settle_receipt,
            scoring_version=self.spec.config['scoring_version'],rule_decision='Measured cloth coverage and corner placement',
            source_global_physics_expectation=dict(gpu_dynamics=True,broadphase='GPU'),global_physics_owner='shared simulator')
    def close(self):
        try:
            if self.subscription is not None:self.subscription.unsubscribe()
        finally:self.subscription=None;self.recorder.close()


def build(spec,ctx):
    from pxr import UsdGeom
    from .mechanism import fold_layout
    import omni.physics.tensors as tensors
    from omni.physx import get_physx_simulation_interface
    validate_case_source(spec)
    validate(spec)
    from mind2act_eval.sim.source_assets import source_file
    source_file(ctx.assets, 'station')
    c=spec.config['source_case'];origin=c['origin'];stage=ctx.sim.stage
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(origin['body_path']))
    initial=np.asarray(mesh.GetPointsAttr().Get(),np.float32)
    if initial.shape!=(6561,3) or hashlib.sha256(initial.tobytes()).hexdigest()!=c['initial_points_sha256']:
        raise ValueError('PR4 original cloth material geometry differs')
    triangles=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
    ctx.robot.prepare_stage(stage)
    hulls=collision_hulls(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    ctx.robot.initialize(ctx.world)
    articulations=ctx.robot.articulations(ctx.world)  # public idempotent shared handle accessor, read only
    views=_body_views(stage)
    def points():
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
        return np.asarray(mesh.GetPointsAttr().Get(),np.float32)@matrix[:3,:3]+matrix[3,:3]
    # Preserve the source physical initialization. Record its diagnostics; no
    # historical settle gate is promoted into a new qualification/scoring gate.
    recent=deque(maxlen=360);history=[];converged=False
    for step in range(7200):
        ctx.sim.step(render=False);recent.append(points().astype(np.float32))
        if step>=1439 and (step+1)%120==0:
            span=float(np.linalg.norm(np.stack(recent)-recent[-1],axis=2).max());history.append(dict(tick=step+1,motion_m=span))
            if span<.0015:converged=True;break
    settled=points();rear=np.arange(81*40,len(initial))
    if spec.difficulty=='medium':rear=fold_layout.diagonal(81)['stationary_ids']
    drift=float(np.quantile(np.linalg.norm((settled-initial)[rear,:2],axis=1),.95))
    receipt=dict(source_settle_converged=converged,history=history,rear_drift_p95_m=drift,
        rear_min_z_m=float(settled[rear,2].min()),qualification_gate_added=False)
    directory=ctx.output_dir/'PR4';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'source-initialization.json').open('x') as stream:json.dump(receipt,stream,allow_nan=False,indent=2)
    view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
    fem=view.create_surface_deformable_body_view(origin['body_path'])
    material=view.create_deformable_material_view('/World/ClothMaterial')
    capacity=fem.max_simulation_nodes_per_body
    if (fem.count!=1 or (capacity!=0 and capacity<6561) or material.count!=1
            or not np.isclose(material.get_youngs_modulus()[0,0],1e7)
            or not np.isclose(material.get_dynamic_friction()[0,0],1.)):
        failure=ValueError('PR4 source native FEM material/body binding missing')
        try:_record_fem_binding_failure(stage,origin['body_path'],fem,material,directory,ctx.world)
        except Exception as diagnostic_error:raise failure from diagnostic_error
        raise failure
    with (directory/'native-fem.json').open('x') as stream:
        json.dump(dict(body_count=int(fem.count),material_count=int(material.count),
            material=dict(young=float(material.get_youngs_modulus()[0,0]),
                          friction=float(material.get_dynamic_friction()[0,0])),
            simulation_nodes_capacity=int(capacity),body_path=origin['body_path'],
            point_source='native_surface_simulation_mesh_usd',
            tensor_comparison_available=capacity>=6561),stream,allow_nan=False,indent=2)
    from .mechanism.support import capture_support
    support_geometry=capture_support(stage,origin)
    contacts=Contacts();subscription=get_physx_simulation_interface().subscribe_contact_report_events(contacts.receive)
    try:
        scene=ClothScene(spec,ctx,mesh,initial,settled,triangles,hulls,views,articulations,fem,contacts,subscription,receipt,support_geometry)
        scene.tensor_view=view  # keep native measurement owner alive
        return scene
    except BaseException:subscription.unsubscribe();raise
