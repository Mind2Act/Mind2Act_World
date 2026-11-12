"""Source-limited shared authoring, active-instance binding and mutation rejection."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import hashlib,json,math
import pxr
from pxr import Usd,UsdGeom,UsdPhysics,Sdf
import pytest
from mind2act_eval.robot import X5Robot,PROFILE
from mind2act_eval.robot.task_collision_overlay import expected_descriptor,snapshot,SourceCollisionGuard
from mind2act_eval.tasks.CP09.instances import spec_for_selected,config


@pytest.fixture(autouse=True)
def physx_metadata(monkeypatch):
    # usd-core records exact authored attribute names; no PhysX execution.
    def metadata(namespace):
        class API:
            def __init__(self,prim):self.prim=prim
            @classmethod
            def Apply(cls,prim):return cls(prim)
            def __getattr__(self,name):
                if name.startswith('Create'):
                    token=name[6:-4];token=token[:1].lower()+token[1:]
                    def create(value):
                        a=self.prim.CreateAttribute(namespace+':'+token,Sdf.ValueTypeNames.Float);a.Set(value);return a
                    return create
                if name.startswith('Get'):
                    token=name[3:-4];token=token[:1].lower()+token[1:]
                    return lambda:self.prim.GetAttribute(namespace+':'+token)
                raise AttributeError(name)
        return API
    monkeypatch.setattr(pxr,'PhysxSchema',SimpleNamespace(
        PhysxCollisionAPI=metadata('physxCollision'),PhysxContactReportAPI=metadata('physxContactReport')),raising=False)


def selected(difficulty='easy'):
    return spec_for_selected(difficulty,instance_id=config('selected')[difficulty]['instance_id'],station_asset='station',
        x5_asset='x5',station_proof_asset='proof',case_asset='case',station_relocation_asset='relocations',asset_version='test')


def stage():
    st=Usd.Stage.CreateInMemory()
    for side in ('Left','Right'):
        root=f'/World/Robot{side}/Asset';prim=UsdGeom.Xform.Define(st,root).GetPrim()
        for name,value in [('solverPositionIterationCount',8),('solverVelocityIterationCount',0)]:
            prim.CreateAttribute('physxArticulation:'+name,Sdf.ValueTypeNames.Int).Set(value)
        prim.CreateAttribute('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool).Set(True)
        for i in range(1,9):
            body=UsdGeom.Xform.Define(st,root+'/link'+str(i)).GetPrim();UsdPhysics.RigidBodyAPI.Apply(body)
            joint=(UsdPhysics.RevoluteJoint if i<=6 else UsdPhysics.PrismaticJoint).Define(st,root+'/joint'+str(i))
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular' if i<=6 else 'linear')
            props=PROFILE['arm' if i<=6 else 'finger']
            scale=math.pi/180 if i<=6 else 1.
            drive.CreateStiffnessAttr(props['stiffness']*scale);drive.CreateDampingAttr(props['damping']*scale);drive.CreateMaxForceAttr(100.)
    return st


def prepared(spec,tmp_path):
    robot=X5Robot();robot.bind_task_instance(spec,tmp_path)
    # Stage preparation itself is shared and tested separately; no SDK is used.
    robot._configured=True;robot._mounted=True
    return robot


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_cp09_shared_overlay_exact_receipt_and_persistence(tmp_path,difficulty):
    spec=selected(difficulty);st=stage();robot=prepared(spec,tmp_path)
    before=snapshot(st);desc=expected_descriptor(spec)
    receipt=robot.apply_task_collision_overlay(st,desc)
    assert receipt['source_sha256']==desc['source_sha256']
    assert len(receipt['added_robot_prims'])==4
    assert receipt['official_drive_solver_limits_unchanged']
    for name in ('before','after'):
        data=(tmp_path/'robot-collision-overlay'/(name+'.usda')).read_bytes()
        assert hashlib.sha256(data).hexdigest()==receipt[name+'_actual_usd_sha256']
    for _ in range(3):robot.verify_task_collision_overlay(st)
    with pytest.raises(RuntimeError):robot.apply_task_collision_overlay(st,desc)
    st.RemovePrim(desc['pads'][0]['path'])
    with pytest.raises(ValueError):robot.verify_task_collision_overlay(st)


def test_cp09_other_active_task_cannot_spoof_descriptor(tmp_path):
    spec=selected();other=replace(spec,task_id='CP08');robot=prepared(other,tmp_path)
    with pytest.raises(ValueError,match='active selected'):robot.apply_task_collision_overlay(stage(),expected_descriptor(spec))
    with pytest.raises(ValueError,match='cannot be replaced'):robot.bind_task_instance(spec,tmp_path)
    assert robot.active_task_spec is other
    with pytest.raises(AttributeError):robot.active_task_spec=spec


@pytest.mark.parametrize('mutation',['path','friction','source','extra'])
def test_cp09_arbitrary_descriptors_rejected_before_authoring(tmp_path,mutation):
    spec=selected();st=stage();robot=prepared(spec,tmp_path);desc=expected_descriptor(spec);before=snapshot(st)
    if mutation=='path':desc['pads'][0]['path']='/World/RobotLeft/Asset/link7/Other'
    if mutation=='friction':desc['pads'][0]['dynamic_friction']=4.
    if mutation=='source':desc['source_sha256']='0'*64
    if mutation=='extra':desc['pads'].append(deepcopy(desc['pads'][0]))
    with pytest.raises(ValueError,match='arbitrary'):robot.apply_task_collision_overlay(st,desc)
    assert snapshot(st)==before


def test_cp09_lifecycle_and_official_force_limits(tmp_path):
    spec=selected();st=stage();robot=X5Robot()
    with pytest.raises(RuntimeError):robot.apply_task_collision_overlay(st,expected_descriptor(spec))
    robot=prepared(spec,tmp_path);robot._initialization_started=True
    with pytest.raises(RuntimeError):robot.apply_task_collision_overlay(st,expected_descriptor(spec))
    robot=prepared(spec,tmp_path)
    force=st.GetAttributeAtPath('/World/RobotLeft/Asset/joint1.drive:angular:physics:maxForce');force.Set(101.)
    with pytest.raises(ValueError,match='nonofficial'):robot.apply_task_collision_overlay(st,expected_descriptor(spec))


def test_cp09_precise_guard_rejects_unrelated_robot_change(tmp_path):
    spec=selected();st=stage();robot=prepared(spec,tmp_path)
    robot.apply_task_collision_overlay(st,expected_descriptor(spec))
    st.GetAttributeAtPath('/World/RobotRight/Asset/joint3.drive:angular:physics:maxForce').Set(99.)
    with pytest.raises(ValueError,match='drive/solver/limits'):robot.verify_task_collision_overlay(st)
    st=stage();guard=SourceCollisionGuard(st,expected_descriptor(spec))
    UsdGeom.Xform.Define(st,'/World/RobotRight/Asset/Unauthorised')
    with pytest.raises(ValueError,match='unauthorized'):guard.authorize_after(st)


def test_cp09_runner_binds_actual_spec_before_task_build(tmp_path):
    from mind2act_eval.contracts import BuildContext,InstanceSpec,Health,Observation,NativeEvidence,Score,ConsumerKind
    from mind2act_eval.sim.demo import DemoRobot,DemoSimulator
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.tasks import TaskDefinition
    from mind2act_eval.runner import run_episode,EpisodeLimits
    spec=InstanceSpec('demo','binding-test','easy','public','v1','none','v1')
    authorizer=X5Robot()
    class Robot(DemoRobot):
        def bind_task_instance(self,actual,output):authorizer.bind_task_instance(actual,output)
    robot=Robot()
    bound_sim_specs=[]
    class Simulator(DemoSimulator):
        def bind_task_instance(self,actual,output):bound_sim_specs.append(actual)
    class Scene:
        def after_step(self,tick):pass
        def observe(self,tick):return Observation(tick,tick/120,**robot.observe())
        def evidence(self,tick):return NativeEvidence('demo','binding-test',tick,{})
        def is_complete(self):return False
        def health(self):return Health(True)
        def close(self):pass
    def build(actual,ctx):
        assert authorizer.active_task_spec is actual is spec
        assert len(bound_sim_specs)==1 and bound_sim_specs[0] is spec
        with pytest.raises(ValueError):authorizer.bind_task_instance(selected(),tmp_path)
        return Scene()
    result=run_episode(spec,TaskDefinition('demo',build,lambda evidence:Score(0.)),
        BuildContext(Simulator(),robot,None,tmp_path),DemoAdapter(),consumer_kind=ConsumerKind.DIAGNOSTIC,
        limits=EpisodeLimits(4,10))
    assert result.infra_valid,result.error


def test_cp09_deactivated_pad_is_not_physical_presence(tmp_path):
    spec=selected();st=stage();robot=prepared(spec,tmp_path)
    descriptor=expected_descriptor(spec);robot.apply_task_collision_overlay(st,descriptor)
    st.GetPrimAtPath(descriptor['pads'][0]['path']).SetActive(False)
    with pytest.raises(ValueError,match='pad missing'):robot.verify_task_collision_overlay(st)
