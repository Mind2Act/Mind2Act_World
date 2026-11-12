"""CPU source/geometry/audit work independent of unresolved region semantics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from mind2act_eval.contracts import InstanceSpec,to_json
from mind2act_eval.tasks.PR4.instances import spec_from_source,validate,digest,source_rest
from mind2act_eval.tasks.PR4.evidence import Recorder,checked_path,POINT_BYTES,contact_envelopes
from mind2act_eval.robot import X5Robot


def geometry():
    y,x=np.mgrid[:81,:81];initial=np.stack([x*.0025,y*.0025,np.ones_like(x)],axis=-1).reshape(-1,3).astype('<f4')
    a=np.arange(80*81).reshape(80,81)[:,:80];b=a+1;c=a+81;d=c+1
    triangles=np.concatenate([np.stack([a,b,c],axis=-1).reshape(-1,3),np.stack([b,d,c],axis=-1).reshape(-1,3)])
    planes=[[1,0,0,-.01],[-1,0,0,-.01],[0,1,0,-.01],[0,-1,0,-.01],[0,0,1,-.01],[0,0,-1,-.01]]
    hulls={s+'/link'+str(j):[planes] for s in ('Left','Right') for j in (7,8)}
    return initial,triangles,hulls


def case():
    initial,_,_=geometry();sha=hashlib.sha256(initial.tobytes()).hexdigest()
    return dict(level='easy',source_seed=410000,initial_points_sha256=sha,origin=dict(grid=[81,81],particles=6561,
        physical_model='native_surface_FEM_cloth',attachments=False,kinematic_nodes=False,adhesion=0,
        youngs_modulus_pa=1e7,dynamic_friction=1.,body_path='/World/DustCloth',initial_points_sha256=sha,
        historical_source='/source-provenance-only/script.py',panel_source='/source-provenance-only/panel.usd',
        panel_prim='/World/PR4/Support/Normalized/Asset'))


def row(tick):
    return dict(tick=tick,time_s=tick/120,joints={s:source_rest() for s in ('Left','Right')},
        tcp={s:[0,0,0] for s in ('Left','Right')},finger_poses={s+'/link'+str(j):dict(position=[0,0,0],quaternion=[1,0,0,0]) for s in ('Left','Right') for j in (7,8)},
        rigid_contact_impulses_ns=[],render_physics_error_m=0.)


def test_producer_preserves_geometry_strips_only_unused_host_provenance():
    c=case();r=dict(task_id='PR4',case_id='PR4-easy-train-seed410000',difficulty='easy',binding=dict(
        declared_version='pr4-seeded-translation-v1',task_definition_version='height-study-raised20mm-seeded-translation-v1',config_sha256=digest(c)),
        config=dict(source_case=c,scene_sha256='a'*64));raw=json.dumps(r).encode();sha=hashlib.sha256(raw).hexdigest()
    s=spec_from_source(raw,source_spec_sha256=sha,assets=dict(station='pr4/station.usda',x5='robot/x5.usda'),asset_version='fixture');validate(s)
    assert '/source-provenance-only' not in json.dumps(to_json(s))
    assert s.config['source_config_sha256']==digest(c)
    assert s.config['source_case']['initial_points_sha256']==c['initial_points_sha256']
    assert tuple(X5Robot(rest=s.config['robot_rest']).rest_pose())==tuple(source_rest())
    with pytest.raises(ValueError):spec_from_source(raw+b' ',source_spec_sha256=sha,assets=dict(station='s',x5='x'),asset_version='fixture')
    changed=to_json(s);changed['config']['source_case']['origin']['dynamic_friction']=.1
    with pytest.raises(ValueError):validate(InstanceSpec.from_dict(changed))


def test_native_binary_audit_is_append_only_and_bounded_receipt(tmp_path):
    initial,triangles,hulls=geometry();r=Recorder('fixture',case(),initial,initial,triangles,hulls,tmp_path)
    sizes=[]
    for t in range(120):
        r.record(row(t),initial)
        if t in (59,119):sizes.append(len(json.dumps(to_json(r.evidence(t)))))
    assert abs(sizes[0]-sizes[1])<15
    e=r.evidence(119);r.close()
    assert checked_path(e.values['points']).stat().st_size==120*POINT_BYTES
    checked_path(e.values['raw_trace']);checked_path(e.values['initial'])
    p=Path(e.values['points']['path']);data=p.read_bytes();p.write_bytes(data[:-4]+b'abcd')
    with pytest.raises(ValueError):checked_path(e.values['points'])
    with pytest.raises(FileExistsError):Recorder('fixture',case(),initial,initial,triangles,hulls,tmp_path)


@pytest.mark.parametrize('mutation',['gap','clock','nan','phase','joints','finger','fem'])
def test_native_measurement_failures(tmp_path,mutation):
    initial,triangles,hulls=geometry();r=Recorder('fixture',case(),initial,initial,triangles,hulls,tmp_path);r.record(row(0),initial)
    f=row(1)
    if mutation=='gap':f['tick']=2
    if mutation=='clock':f['time_s']=.1
    if mutation=='nan':initial[0,0]=float('nan')
    if mutation=='phase':f['expert_phase']='success'
    if mutation=='joints':f['joints']['Left']=[]
    if mutation=='finger':del f['finger_poses']['Right/link8']
    if mutation=='fem':f['render_physics_error_m']=-.000021
    with pytest.raises(ValueError):r.record(f,initial)
    r.close()


def test_envelopes_derived_from_measured_fingers_not_reported_material_ids():
    initial,_,hulls=geometry();f=row(0);p=initial.copy();p[:,2]=0.
    result=contact_envelopes(f,p,hulls,np.arange(81))
    assert result['Left/link7']['hem_contact_ids']
    f['finger_poses']['Left/link7']['position']=[1,0,0]
    result=contact_envelopes(f,p,hulls,np.arange(81));assert not result['Left/link7']['hem_contact_ids']


@pytest.mark.parametrize('bad_field',[None,'source_surface_zero_capacity','fem_count','max_nodes','youngs_modulus','dynamic_friction','empty_material',
                                    'physics_getter','physics_drift','second_scene'])
def test_real_build_cpu_fem_seam_has_no_task_global_setters(tmp_path,monkeypatch,bad_field):
    """CPU fakes check guard/diagnostics only; they cannot reproduce native FEM."""
    import sys
    from types import SimpleNamespace
    from pxr import Usd,UsdGeom,Sdf
    from mind2act_eval.tasks.PR4 import scene as module
    # This synthetic FEM fixture tests native readback failures, not admission.
    # Exact source admission is exercised separately against actual releases.
    monkeypatch.setattr(module,'validate_case_source',lambda spec:None)
    initial,triangles,hulls=geometry();c=case();stage=Usd.Stage.CreateInMemory()
    mesh=UsdGeom.Mesh.Define(stage,'/World/DustCloth');mesh.CreatePointsAttr(initial)
    mesh.CreateFaceVertexCountsAttr([3]*len(triangles));mesh.CreateFaceVertexIndicesAttr(triangles.ravel())
    from .support_fixture import author_support
    author_support(stage,top=1.)
    material=stage.DefinePrim('/World/ClothMaterial','Material')
    for name,value in (('omniphysics:youngsModulus',1e7),('omniphysics:dynamicFriction',1.)):
        material.CreateAttribute(name,Sdf.ValueTypeNames.Float).Set(value)
    station=tmp_path/'station';station.write_bytes(b'original source')
    r=dict(task_id='PR4',case_id='fixture',difficulty='easy',binding=dict(declared_version='pr4-seeded-translation-v1',
        task_definition_version='height-study-raised20mm-seeded-translation-v1',config_sha256=digest(c)),
        config=dict(source_case=c,scene_sha256=hashlib.sha256(station.read_bytes()).hexdigest()))
    raw=json.dumps(r).encode();s=spec_from_source(raw,source_spec_sha256=hashlib.sha256(raw).hexdigest(),
        assets=dict(station='pr4/scene',x5='robot/x5'),asset_version='fixture')
    def gpu_dynamics():
        if bad_field=='physics_getter':raise RuntimeError('SDK readback unavailable')
        return bad_field!='physics_drift'
    physics=SimpleNamespace(prim_path='/physicsScene',device='cpu',use_gpu_pipeline=False,
        is_gpu_dynamics_enabled=gpu_dynamics,get_broadphase_type=lambda:'GPU',get_solver_type=lambda:'TGS')
    readback_calls=[]
    def get_context():readback_calls.append(True);return physics
    world=SimpleNamespace(current_time=0.,stage=stage,backend='numpy',device='cpu',
        get_physics_context=get_context,get_physics_dt=lambda:1/120,get_rendering_dt=lambda:1/30)
    physics_scene=stage.DefinePrim('/physicsScene','PhysicsScene')
    physics_scene.CreateAttribute('physxScene:enableGPUDynamics',Sdf.ValueTypeNames.Bool).Set(True)
    physics_scene.CreateAttribute('physxScene:broadphaseType',Sdf.ValueTypeNames.Token).Set('GPU')
    if bad_field=='second_scene':stage.DefinePrim('/secondPhysicsScene','PhysicsScene')
    steps=[]
    def step(**kwargs):world.current_time+=1/120;steps.append(kwargs)
    sim=SimpleNamespace(stage=stage,step=step)
    robot=SimpleNamespace(prepare_stage=lambda _:None,initialize=lambda _:None,
        articulations=lambda _:{side:SimpleNamespace(get_joint_positions=lambda:np.asarray(source_rest())) for side in ('Left','Right')})
    ctx=SimpleNamespace(world=world,sim=sim,robot=robot,assets=SimpleNamespace(resolve=lambda _:station),output_dir=tmp_path/'output')
    views={side+'/'+link:SimpleNamespace(get_world_pose=lambda:(np.array([1.,1.,1.]),np.array([1.,0.,0.,0.]))) for side in ('Left','Right') for link in ('link6','link7','link8')}
    monkeypatch.setattr(module,'collision_hulls',lambda _:hulls);monkeypatch.setattr(module,'_body_views',lambda _:views)
    def native_nodes():
        assert bad_field!='source_surface_zero_capacity','surface tensor nodes are unavailable in the successful source runtime'
        return initial[None]
    fem=SimpleNamespace(count=0 if bad_field in ('fem_count','empty_material','physics_getter','physics_drift','second_scene') else 1,
        max_simulation_nodes_per_body=(0 if bad_field=='source_surface_zero_capacity' else 6560 if bad_field=='max_nodes' else 6561),
        get_simulation_nodal_positions=native_nodes)
    def youngs_modulus():
        assert bad_field!='empty_material','do not read native tensors from an empty material view'
        return np.array([[0. if bad_field=='youngs_modulus' else 1e7]])
    mat=SimpleNamespace(count=0 if bad_field=='empty_material' else 1,get_youngs_modulus=youngs_modulus,
        get_dynamic_friction=lambda:np.array([[0. if bad_field=='dynamic_friction' else 1.]]))
    view=SimpleNamespace(set_subspace_roots=lambda _:None,create_surface_deformable_body_view=lambda _:fem,create_deformable_material_view=lambda _:mat)
    tensor=SimpleNamespace(create_simulation_view=lambda _:view);closed=[]
    physx=SimpleNamespace(get_physx_simulation_interface=lambda:SimpleNamespace(subscribe_contact_report_events=lambda _:SimpleNamespace(unsubscribe=lambda:closed.append(True))))
    monkeypatch.setitem(sys.modules,'omni',SimpleNamespace(physics=SimpleNamespace(tensors=tensor),physx=physx))
    monkeypatch.setitem(sys.modules,'omni.physics',SimpleNamespace(tensors=tensor));monkeypatch.setitem(sys.modules,'omni.physics.tensors',tensor);monkeypatch.setitem(sys.modules,'omni.physx',physx)
    if bad_field not in (None,'source_surface_zero_capacity'):
        with pytest.raises(ValueError,match='PR4 source native FEM material/body binding missing'):
            module.build(s,ctx)
        receipt=json.loads((ctx.output_dir/'PR4/native-fem-binding-failure.json').read_text())
        assert receipt['body_path']=='/World/DustCloth'
        diagnostic=receipt['physics_after_source_settle']['value']
        assert readback_calls==[True]
        expected_gpu=({'error':'RuntimeError: SDK readback unavailable'} if bad_field=='physics_getter'
                      else {'value':bad_field!='physics_drift'})
        assert diagnostic['context']['gpu_dynamics']==expected_gpu
        assert diagnostic['context']['broadphase']=={'value':'GPU'}
        assert diagnostic['context']['solver']=={'value':'TGS'}
        assert diagnostic['context']['use_gpu_pipeline']=={'value':False}
        assert diagnostic['world']['matches_task_stage']=={'value':True}
        expected_scenes=[{
            'path':'/physicsScene','attributes':{'physxScene:enableGPUDynamics':True,
                                               'physxScene:broadphaseType':'GPU'}}]
        if bad_field=='second_scene':expected_scenes.append({'path':'/secondPhysicsScene','attributes':{}})
        assert diagnostic['scenes']['value']==expected_scenes
        assert receipt['material_source']['value']['attributes']=={
            'omniphysics:youngsModulus':1e7,'omniphysics:dynamicFriction':1.}
        expected=dict(fem_count=dict(value=fem.count),
            max_simulation_nodes_per_body=dict(value=fem.max_simulation_nodes_per_body),
            material_count=dict(value=mat.count))
        if bad_field=='empty_material':
            expected.update({name:dict(unavailable='material_count is not 1')
                             for name in ('youngs_modulus','dynamic_friction')})
        else:
            expected.update(youngs_modulus=dict(value=mat.get_youngs_modulus().tolist()),
                dynamic_friction=dict(value=mat.get_dynamic_friction().tolist()))
        assert receipt['readbacks']==expected
        assert closed==[]
        return
    native=module.build(s,ctx)
    assert readback_calls==[]
    assert not (ctx.output_dir/'PR4/native-fem-binding-failure.json').exists()
    assert len(steps)==1440 and native.settle_receipt['source_settle_converged']
    assert native.evidence(0).tick==0
    assert native.result_metadata()['global_physics_owner']=='shared simulator'
    if bad_field=='source_surface_zero_capacity':
        trace=json.loads((ctx.output_dir/'PR4/raw-trace.jsonl').read_text().splitlines()[0])
        assert trace['render_physics_error_m'] is None
        assert trace['point_source']=='native_surface_simulation_mesh_usd'
        binding=json.loads((ctx.output_dir/'PR4/native-fem.json').read_text())
        assert binding['body_count']==binding['material_count']==1
        assert binding['simulation_nodes_capacity']==0 and not binding['tensor_comparison_available']
    world.current_time+=1/120;native.after_step(1);checked_path(native.evidence(1).values['points'])
    native.close();assert closed==[True]


def test_unavailable_tensor_comparison_requires_explicit_native_mesh_source():
    from mind2act_eval.tasks.PR4.evidence import validate_row
    sample=row(0);sample['render_physics_error_m']=None
    with pytest.raises(ValueError,match='requires the native surface mesh source'):validate_row(sample,0)
    sample['point_source']='native_surface_simulation_mesh_usd';validate_row(sample,0)
    sample['point_source']='expert_or_interpolated_points'
    with pytest.raises(ValueError,match='unsupported native point source'):validate_row(sample,0)
