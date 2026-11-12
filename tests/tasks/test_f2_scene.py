"""CPU scene measurement hooks, binding and isolation; no Kit or GPU is used."""
import hashlib
import json
import builtins
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from test_f2_mechanism import source_case
from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.tasks.F2 import scene
from mind2act_eval.tasks.F2.provenance import SOURCE_ID,VERSION,effective_config
from mind2act_eval.tasks.F2.evidence import Manifolds,CAP,OBJECT_ROOT
from mind2act_eval.tasks.F2.scorer import score
from test_f2_scorer import Recorded


def origin_file(tmp_path):
    row=dict(public_task='F2',cell='F2_easy',instance_id='synthetic-input',rows=1000,
             actual_h5_sha256='a'*64,instance_sha256='b'*64,source_clock=dict(physics_hz=120,control_hz=30,rgb_hz=30),
             timeline=[dict(state_physics_step=600,action_physics_step=604,state_control_index=0,action_control_index=4)])
    path=tmp_path/'origin.json';raw=json.dumps(dict(cells=[row])).encode();path.write_bytes(raw)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),instance_id='synthetic-input')


def test_validation_uses_exact_source_origin_and_is_read_only(tmp_path):
    station=tmp_path/'scene.usda';station.write_text('#usda 1.0\n')
    origin=origin_file(tmp_path);source=source_case()
    cfg=dict(source_case=source,source_case_sha256=source['case_sha256'],scene_sha256=hashlib.sha256(station.read_bytes()).hexdigest(),
             instruction=scene.instruction(),input_protocol_version=VERSION,source_start_tick=600,
             source_receipt_sha256=origin['sha256'],source_instance_id='synthetic-input',source_instance_sha256='b'*64)
    spec=InstanceSpec('F2','synthetic','easy',SOURCE_ID,VERSION,'test','v1',seed=1,config=cfg,
                      assets=dict(station='scene.usda',source_receipt='origin.json'))
    ctx=NS(assets=NS(resolve=lambda key:station if key=='station' else Path(origin['path'])),output_dir=tmp_path/'out')
    assert scene.validate(spec,ctx)[2]==600
    assert scene.validate(spec,ctx)[2]==600
    assert not ctx.output_dir.exists()
    wrong=to_json(spec);wrong['config']['source_start_tick']=0
    with pytest.raises(ValueError,match='projection'):
        scene.validate(InstanceSpec.from_dict(wrong),ctx)
    wrong=to_json(spec);wrong['assets'].pop('source_receipt')
    with pytest.raises(ValueError,match='requires'):
        scene.validate(InstanceSpec.from_dict(wrong),ctx)


@pytest.mark.parametrize('level',['easy','medium','hard'])
@pytest.mark.parametrize('portable',[False,True])
def test_build_archives_verified_source_bytes_for_independent_scoring(tmp_path,monkeypatch,level,portable):
    from mind2act_eval.sim import LocalAssets, source_assets
    root=tmp_path/'fixture-release';root.mkdir()
    station=root/'scene.usda';station.write_text('#usda 1.0\n')
    origin=origin_file(root)
    value=json.loads(Path(origin['path']).read_bytes())
    value['cells'][0].update(cell='F2_'+level,source_path='/source/file')
    original=(json.dumps(value,indent=2)+'\n').encode()
    runtime=original.replace(b'/source/file',b'portable/file') if portable else original
    origin_path=Path(origin['path']);origin_path.write_bytes(runtime)
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    source=source_case(level)
    cfg=dict(source_case=source,source_case_sha256=source['case_sha256'],
             scene_sha256=digest(station.read_bytes()),instruction=scene.instruction(),
             input_protocol_version=VERSION,source_start_tick=600,
             source_receipt_sha256=digest(original),source_instance_id='synthetic-input',
             source_instance_sha256='b'*64)
    spec=InstanceSpec('F2','synthetic',level,SOURCE_ID,VERSION,'test','v1',seed=1,config=cfg,
                      assets=dict(station=station.name,source_receipt=origin_path.name))
    approvals=[]
    catalogue=dict(approvals=approvals)
    monkeypatch.setattr(source_assets,'_catalogue',lambda:catalogue)
    if portable:
        identity=source_assets._identity(spec)
        proof=dict(schema='mind2act_eval.source_projection_restoration.v1',identity=identity,
                   source_sha256=digest(original),runtime_sha256=digest(runtime),json=dict(
                       undo=[dict(pointer='/cells/0/source_path',source='/source/file',runtime='portable/file')],
                       serialization=dict(indent=2,trailing_newline=True)))
        raw=json.dumps(proof).encode();proof_root=tmp_path/'proofs';proof_root.mkdir()
        (proof_root/(digest(raw)+'.json')).write_bytes(raw)
        monkeypatch.setenv('MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT',str(proof_root))
        approvals.append(dict(asset_name=root.name,runtime_path=origin_path.name,
            source_sha256=digest(original),runtime_sha256=digest(runtime),identity=identity,
            proof_sha256=digest(raw),closure=[dict(path=origin_path.name,sha256=digest(runtime))]))
    ctx=NS(assets=LocalAssets(root,spec.assets,spec=spec),output_dir=tmp_path/'run')
    # Exercise real validation and build archival, without importing or faking the SDK.
    class NativeBoundary(Exception):pass
    real_import=builtins.__import__
    def import_guard(name,*args,**kwargs):
        if name=='isaacsim.core.prims':raise NativeBoundary()
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',import_guard)
    receipt=scene.validate(spec,ctx)[3]
    assert not ctx.output_dir.exists()
    with pytest.raises(NativeBoundary):scene.build(spec,ctx)
    archive=ctx.output_dir/'private/F2/source-origin.json'
    # This real scorer independently reads the build archive without source_bytes.
    measurements=tmp_path/'measurements';measurements.mkdir()
    f=Recorded(measurements,level,offset=600)
    try:
        f.place(f.c['target']);f.confirm()
        f.origin=dict(receipt,path=str(archive))
        assert score(f.evidence()).value==100
        assert archive.read_bytes()==original
        assert digest(archive.read_bytes())==receipt['sha256']==cfg['source_receipt_sha256']
        assert origin_path.read_bytes()==runtime
        # Rebuilding may reuse an identical archive, but never overwrite another origin.
        with pytest.raises(NativeBoundary):scene.build(spec,ctx)
        archive.write_bytes(runtime+b' ')
        with pytest.raises(ValueError,match='source-origin receipt SHA mismatch'):score(f.evidence())
        with pytest.raises(ValueError,match='private source receipt differs'):scene.build(spec,ctx)
        assert archive.read_bytes()==runtime+b' '
    finally:
        f.writer.close()


class Body:
    def __init__(self,position):self.p=np.asarray(position,dtype=float)
    def get_world_pose(self):return self.p.copy(),np.asarray([1.,0.,0.,0.])
    def get_linear_velocity(self):return np.zeros(3)
    def get_angular_velocity(self):return np.zeros(3)


@pytest.mark.parametrize('offset',[0,600])
def test_native_pose_and_contact_hooks_stream_to_independent_score(tmp_path,monkeypatch,offset):
    source=source_case();c=effective_config(source,input_version=VERSION)
    spec=InstanceSpec('F2','synthetic','easy',SOURCE_ID,VERSION,'test','v1',seed=1,
                      config=dict(instruction=scene.instruction(),input_protocol_version=VERSION))
    origin=origin_file(tmp_path) if offset else None
    bodies={name:Body(value['position']) for name,value in c['objects'].items()}
    cap=Body(c['button']['position']);joint=NS(q=.005)
    button=NS(get_joint_positions=lambda:np.asarray([joint.q]),get_joint_velocities=lambda:np.zeros(1))
    contacts=Manifolds();world=NS(current_time=.025)
    ctx=NS(output_dir=tmp_path,world=world,sim=NS(stage=None),robot=NS(observe=lambda:dict(
        left_arm=(0.,)*6,right_arm=(0.,)*6,left_gripper=1.,right_gripper=1.)))
    monkeypatch.setattr(scene,'FeedbackDisplay',lambda stage:NS(update=lambda public:None))
    native=scene.ExperimentScene(spec,ctx,source,c,offset,origin,bodies,button,cap,contacts)
    tick=0
    def step(count=1,pressed=False):
        nonlocal tick
        for _ in range(count):
            tick+=1;native.before_step(tick);world.current_time+=1/120
            joint.q=.001 if pressed else .005
            cap.p[2]=c['button']['position'][2]-(.004 if pressed else 0.)
            if pressed:
                contacts.events.append(dict(pair=[CAP+'/Collision','/World/RobotLeft/Asset/link7/Collision'],
                    active=True,contact_count=1,min_separation_m=0.,points=[dict(position=cap.p.tolist(),normal=[0.,0.,1.],separation=0.)]))
            native.after_step(tick)
    if offset==0:step(600)
    for region_id,name in zip(c['regions'],c['target']):
        r=c['regions'][region_id];p=np.asarray([*r['center'],r['floor_z']+c['objects'][name]['size'][2]/2])
        bodies[name].p=p
        pair=(OBJECT_ROOT+name+'/Collision',f'/World/Task/Regions/{region_id}/floor')
        contacts.pairs[pair]=True
        contacts.details[pair]=[dict(position=[*p[:2].tolist(),r['floor_z']],normal=[0.,0.,1.],separation=0.)]
    step(60);step(3,pressed=True)
    assert not native.is_complete() and score(native.evidence(tick)).value==0
    step()
    result=score(native.evidence(tick))
    assert native.is_complete() and result.value==100
    assert result.components['source_last_tick']==offset+tick
    assert result.components['native_physics_ticks']==tick
    assert native.observe(tick).instruction==scene.instruction()
    assert native.result_metadata()['native_physics_ticks']==tick
    native.close()


def test_task_contact_setup_does_not_write_robot_or_global_physics(monkeypatch):
    import pxr
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.CreateInMemory()
    for path in ('/World/Task/Objects/block1','/World/RobotLeft/Asset/link7','/World/CameraStand'):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage,path).GetPrim())
    touched=[]
    class Reports:
        @staticmethod
        def Apply(prim):
            touched.append(str(prim.GetPath()))
            return NS(CreateThresholdAttr=lambda value:None)
    monkeypatch.setattr(pxr,'PhysxSchema',NS(PhysxContactReportAPI=Reports),raising=False)
    scene.prepare_task_contacts(stage)
    assert touched==['/World/Task/Objects/block1']
