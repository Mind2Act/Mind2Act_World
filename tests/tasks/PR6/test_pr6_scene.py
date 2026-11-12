"""CPU-only producer, native scene seam and immutable prop configuration."""
import copy
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from mind2act_eval.tasks.PR6.instances import spec_from_source, validate, digest, source_rest
from mind2act_eval.tasks.PR6.scene import configure_props, DominoScene
from mind2act_eval.tasks.PR6.mechanism.contacts import Contacts
from mind2act_eval.tasks.PR6.scorer import score
from mind2act_eval.robot import X5Robot
from test_pr6_rules import case


def source_row():
    c = case()
    return dict(task_id='PR8',case_id='PR8-easy-fixture',difficulty='easy',binding=dict(
        declared_version='pr8-four-case-native-v4',source_case_sha256=c['case_sha256'],seed=0,
        source_config_bytes_sha256='a'*64),config=dict(source_case=c,native_inputs=dict(manifest=dict(scene_sha256='b'*64))))


def spec(row=None):
    raw=json.dumps(row or source_row()).encode()
    return spec_from_source(raw,source_spec_sha256=hashlib.sha256(raw).hexdigest(),
                            assets=dict(station='pr6/station.usda',x5='robots/x5.usda'),asset_version='fixture')


def test_source_bound_public_alias_inputs_and_shared_rest():
    s=spec(); validate(s)
    assert s.task_id=='PR6' and s.instance_id.startswith('PR8-')
    assert len(s.config['goal_recipe'])>1000 and '20:' in s.config['goal_recipe']
    assert tuple(X5Robot(rest=s.config['robot_rest']).rest_pose())==tuple(source_rest())
    assert source_rest()[-2:]==[.035,.035]


@pytest.mark.parametrize('change',['sha','version','seed','count','asset','instruction','rest','case'])
def test_wrong_source_binding_is_rejected(change):
    row=source_row()
    if change=='version': row['binding']['declared_version']='old'
    if change=='seed': row['binding']['seed']=9
    if change=='count': del row['config']['source_case']['objects']['domino5']
    if change=='case': row['config']['source_case']['bell']['ring_angle_deg']=1
    if change in ('version','seed','count','case'):
        with pytest.raises(ValueError): spec(row)
    elif change=='sha':
        with pytest.raises(ValueError): spec_from_source(b'{}',source_spec_sha256='a'*64,assets={},asset_version='fixture')
    elif change=='asset':
        raw=json.dumps(row).encode()
        with pytest.raises(ValueError): spec_from_source(raw,source_spec_sha256=hashlib.sha256(raw).hexdigest(),assets=dict(station='/root/private',x5='x5'),asset_version='fixture')
    else:
        from mind2act_eval.contracts import InstanceSpec,to_json
        data=to_json(spec());data['config']['instruction' if change=='instruction' else 'robot_rest']='changed' if change=='instruction' else [0]*8
        with pytest.raises(ValueError): validate(InstanceSpec.from_dict(data))


def record_physics(monkeypatch):
    import pxr
    # usd-core has no Isaac PhysxSchema: record native authoring calls only.
    recorded = {}
    class API:
        def __init__(self,prim): self.path=str(prim.GetPath())
        @classmethod
        def Apply(cls,prim): return cls(prim)
        def __getattr__(self,name):
            return lambda value: recorded.__setitem__((self.path,name),value)
    monkeypatch.setattr(pxr,"PhysxSchema",SimpleNamespace(**{n:API for n in ("PhysxSceneAPI","PhysxRigidBodyAPI","PhysxContactReportAPI","PhysxCollisionAPI")}),raising=False)
    return recorded


def test_exact_prop_physics_without_robot_mass_geometry_or_friction_changes(monkeypatch):
    recorded=record_physics(monkeypatch)
    stage=Usd.Stage.CreateInMemory();UsdPhysics.Scene.Define(stage,'/World/Physics')
    for name in case()['objects']: stage.DefinePrim('/World/Task/Objects/'+name,'Xform')
    prop=UsdGeom.Cube.Define(stage,'/World/Task/Objects/domino1');prop.CreateSizeAttr(.08)
    UsdPhysics.RigidBodyAPI.Apply(prop.GetPrim());UsdPhysics.CollisionAPI.Apply(prop.GetPrim())
    UsdPhysics.MassAPI.Apply(prop.GetPrim()).CreateMassAttr(.1)
    material=UsdPhysics.MaterialAPI.Apply(stage.DefinePrim('/World/Material','Material'));material.CreateStaticFrictionAttr(.6)
    robot=stage.DefinePrim('/World/RobotLeft/Asset/link6','Xform');UsdPhysics.RigidBodyAPI.Apply(robot)
    configure_props(stage,case())
    root='/World/Task/Objects/domino1'
    assert recorded[(root,'CreateSolverPositionIterationCountAttr')]==128
    assert recorded[(root,'CreateSolverVelocityIterationCountAttr')]==8
    assert not any(path=='/World/Physics' for path,_ in recorded)
    assert recorded[(root,'CreateContactOffsetAttr')]==.001
    assert not any(path.startswith('/World/Robot') for path,_ in recorded)
    assert prop.GetSizeAttr().Get()==.08 and UsdPhysics.MassAPI(prop.GetPrim()).GetMassAttr().Get()==pytest.approx(.1)
    assert material.GetStaticFrictionAttr().Get()==pytest.approx(.6)


def test_scene_native_readout_zero_action_and_cleanup(tmp_path):
    s=spec(); c=s.config['source_case'];names=list(c['objects'])
    objects=SimpleNamespace(prim_paths=['/World/Task/Objects/'+n for n in names],
        get_world_poses=lambda:(np.array([c['objects'][n]['position'] for n in names]),np.tile([1,0,0,0],(len(names),1))),
        get_velocities=lambda:np.zeros((len(names),6)))
    closed=[];ctx=SimpleNamespace(world=SimpleNamespace(current_time=2/120),output_dir=tmp_path)
    scene=DominoScene(s,ctx,objects,Contacts(),SimpleNamespace(unsubscribe=lambda:closed.append(True)))
    assert score(scene.evidence(0)).value==0 and not scene.is_complete()
    assert scene.result_metadata()['public_notice_interface_pending'] is False
    scene.close();assert closed==[True]


@pytest.mark.parametrize('tick,enabled', [(0,False), (596,False), (600,True), (604,True)])
def test_e2e_handoff_instruction_is_static_across_notice_boundary(tick,enabled):
    from mind2act_eval.tasks.PR6.mechanism.dominos import Dominos
    s=spec(); rules=Dominos(s.config['source_case']); rules.tick=tick
    # These scoring facts must not reach the public observation.
    rules.phase='private-phase-canary'; rules.chain=['private-chain-canary']
    robot=SimpleNamespace(observe=lambda:dict(left_arm=(0.,)*6,right_arm=(0.,)*6,
                                             left_gripper=1.,right_gripper=1.,
                                             rgb={k:np.zeros((1,1,3),dtype=np.uint8) for k in ('head','left_wrist','right_wrist')}))
    scene=object.__new__(DominoScene)
    scene.spec=s; scene.ctx=SimpleNamespace(robot=robot); scene.recorder=SimpleNamespace(rules=rules)
    before=copy.deepcopy({name:value for name,value in vars(rules).items() if name!='config'})
    observation=scene.observe(tick)
    from mind2act_eval.adapters.xpolicylab import public_observation,CAMERAS
    assert observation.instruction==s.config['instruction']
    assert public_observation(observation,CAMERAS)['task']==s.config['instruction']
    assert rules.action_enabled is enabled
    assert 'private-' not in observation.instruction
    assert {name:value for name,value in vars(rules).items() if name!='config'}==before
    assert rules.config is s.config['source_case']  # No scene/controller state changes.
    assert observation.tick==tick and observation.sim_time_s==tick/120


def test_actual_build_writes_only_owned_objects(tmp_path,monkeypatch):
    import sys
    from mind2act_eval.tasks.PR6 import scene as module
    recorded=record_physics(monkeypatch)
    stage=Usd.Stage.CreateInMemory()
    for path in ('/World/Physics','/World/Table','/World/OtherTask/Prop','/World/RobotLeft/Asset/link6'):
        prim=stage.DefinePrim(path,'Xform');UsdPhysics.RigidBodyAPI.Apply(prim);UsdPhysics.CollisionAPI.Apply(prim)
    for name in case()['objects']:
        prim=stage.DefinePrim('/World/Task/Objects/'+name,'Xform');UsdPhysics.RigidBodyAPI.Apply(prim);UsdPhysics.CollisionAPI.Apply(prim)
    station=tmp_path/'station.usda';station.write_bytes(b'source scene')
    row=source_row();row['config']['native_inputs']['manifest']['scene_sha256']=hashlib.sha256(station.read_bytes()).hexdigest();s=spec(row)
    c=s.config['source_case'];names=list(c['objects'])
    objects=SimpleNamespace(prim_paths=['/World/Task/Objects/'+n for n in names],initialize=lambda:None,
        get_world_poses=lambda:(np.array([c['objects'][n]['position'] for n in names]),np.tile([1,0,0,0],(len(names),1))),
        get_velocities=lambda:np.zeros((len(names),6)),set_velocities=lambda _:None)
    monkeypatch.setattr(module,'_objects',lambda _:objects)
    closed=[];physics=SimpleNamespace(get_physx_interface=lambda:SimpleNamespace(update_transformations=lambda *args:None))
    monkeypatch.setitem(sys.modules,'omni',SimpleNamespace(physx=physics));monkeypatch.setitem(sys.modules,'omni.physx',physics)
    from mind2act_eval.tasks.PR6.mechanism import contacts
    monkeypatch.setattr(contacts,'subscribe',lambda _:SimpleNamespace(unsubscribe=lambda:closed.append(True)))
    world=SimpleNamespace(current_time=2/120,physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda:None))
    ctx=SimpleNamespace(sim=SimpleNamespace(stage=stage),world=world,output_dir=tmp_path/'run',
        assets=SimpleNamespace(resolve=lambda _:station),robot=SimpleNamespace(prepare_stage=lambda _:None,initialize=lambda _:None))
    native=module.build(s,ctx)
    assert score(native.evidence(0)).value==0
    assert recorded and all(path.startswith('/World/Task/Objects/') for path,_ in recorded)
    assert native.result_metadata()['source_global_physics_expectation']['solver_type']=='TGS'
    native.close();assert closed==[True]


def test_model_has_no_notice_hold_and_preserves_source_instruction():
    from mind2act_eval.tasks.PR6.scene import TASK_PROTOCOL_VERSION
    scene = object.__new__(DominoScene)
    scene.spec = spec()
    assert not hasattr(scene, 'initial_model_hold_ticks')
    assert TASK_PROTOCOL_VERSION == 'mind2act-final-scoring-20261105'
    assert scene.spec.config['instruction'].endswith('using as few pushes as possible.')
