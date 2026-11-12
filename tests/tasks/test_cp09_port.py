"""Acquisition geometry, source identity, and independent audit regression checks."""
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
import json
import base64
import os
from functools import lru_cache
from pathlib import Path
import pxr
import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
from mind2act_eval.tasks.CP09.instances import spec_for_selected,validate,config,signature,DECISION
from mind2act_eval.tasks.CP09.geometry import descriptor,verify
from mind2act_eval.tasks.CP09.mechanism.props import build_props
from mind2act_eval.tasks.CP09.mechanism.layouts import make_case
from mind2act_eval.tasks.CP09.mechanism.environment import Environment
from mind2act_eval.tasks.CP09.replay import BoardFacts,Replay
from mind2act_eval.tasks.CP09.audit import Audit
from mind2act_eval.tasks.CP09.scorer import score
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP09 import CONTRACT


@lru_cache(maxsize=1)
def original_heldout():
    from mind2act_eval.original_test_inputs import digest,SNAPSHOT_SHA256
    path=os.environ.get('MIND2ACT_ORIGINAL_SNAPSHOT')
    if not path:pytest.skip('private original source snapshot not configured')
    raw=Path(path).read_bytes();assert digest(raw)==SNAPSHOT_SHA256
    return [r for r in json.loads(raw)['records'] if r['task_id']=='CP09']


@pytest.fixture(autouse=True)
def physx_metadata(monkeypatch):
    # usd-core has no Isaac PhysxSchema. This records authored metadata only;
    # CPU checks do not claim native contact/dynamics validation.
    class Metadata:
        def __init__(self,prim):self.prim=prim
        @classmethod
        def Apply(cls,prim):return cls(prim)
        def __getattr__(self,name):
            if name.startswith('Create'):
                key='cp09Test:'+name[6:-4]
                def create(value):
                    typ=Sdf.ValueTypeNames.Bool if isinstance(value,bool) else Sdf.ValueTypeNames.Float
                    attr=self.prim.CreateAttribute(key,typ);attr.Set(value);return attr
                return create
            if name.startswith('Get'):return lambda:self.prim.GetAttribute('cp09Test:'+name[3:-4])
            raise AttributeError(name)
    schema=SimpleNamespace(**{name:Metadata for name in ('PhysxCollisionAPI','PhysxRigidBodyAPI','PhysxContactReportAPI')})
    monkeypatch.setattr(pxr,'PhysxSchema',schema,raising=False)
    globals()['PhysxSchema']=schema


def selected(difficulty='easy'):
    return spec_for_selected(difficulty,instance_id=config('selected')[difficulty]['instance_id'],station_asset='station.usda',
        x5_asset='x5.usd',station_proof_asset='proof.json',case_asset='case.json',station_relocation_asset='relocations.json',asset_version='test')


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_cp09_exact_selected_and_instruction(difficulty):
    spec=selected(difficulty);validate(spec)
    case,_,_=make_case(difficulty,0)
    assert case['difficulty']==difficulty and case['seed']==spec.seed
    assert spec.config['geometry_decision']==DECISION
    for new in (replace(spec,seed=1),replace(spec,instance_id='other'),replace(spec,source_id='snapshot'),
                replace(spec,config={**spec.config,'instruction':'old external instruction'}),
                replace(spec,config={**spec.config,'geometry_decision':DECISION+' unreviewed'})):
        with pytest.raises(ValueError):validate(new)


@pytest.mark.parametrize('value',[None,False,{},'unreviewed',DECISION+' ',DECISION.replace('2026','2025')])
def test_cp09_source_decision_digest_is_not_a_general_metadata_alias(value):
    from mind2act_eval.robot.task_collision_overlay import is_geometry_decision
    assert not is_geometry_decision(value)


def pads(stage):
    material=UsdShade.Material.Define(stage,'/World/CP09/GripMaterial')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim());api.CreateStaticFrictionAttr(2.);api.CreateDynamicFrictionAttr(1.8);api.CreateRestitutionAttr(0.)
    for row in descriptor()['pads']:
        shape=UsdGeom.Cube.Define(stage,row['path']);shape.CreateSizeAttr(1.)
        shape.AddTranslateOp().Set(Gf.Vec3d(*row['translate']));shape.AddScaleOp().Set(Gf.Vec3f(*row['scale']))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        api=PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim());api.CreateContactOffsetAttr(.0001);api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material,materialPurpose='physics')


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_cp09_source_props_and_pads_persist_across_presentation(tmp_path,difficulty):
    stage=Usd.Stage.CreateInMemory();pads(stage)
    task=build_props(stage,tmp_path,difficulty,0,config('station'),make_case(difficulty,0)[0])
    tools=import_module('mind2act_eval.tasks.CP09.mechanism.'+difficulty+'_tools')
    support=import_module('mind2act_eval.tasks.CP09.mechanism.'+difficulty+'_board_support')
    env=Environment(task,{}, {},None,tools);tools.build_tools(env)
    env.board_support=support.BoardSupport(env)
    env.board_support.position=lambda:np.array([0,.310,.914])
    for side in ('Left','Right'):
        assert not stage.GetPrimAtPath('/World/CP09/Board/'+side+'FrameGrip')
        for suffix in ('HandleBridge','NearHandleMount','FarHandleMount'):
            p=stage.GetPrimAtPath('/World/CP09/Board/Visuals/'+side+suffix)
            assert UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            assert PhysxSchema.PhysxCollisionAPI(p).GetRestOffsetAttr().Get()==0
    env.board_support.retrieval_observation=lambda:dict(placed=True,machine_retracted=True)
    for tick in (0,359,360,432,480,528,600,2000,50400):
        env.tick=tick;env._boards();assert verify(stage)
    assert env.annotation_applied
    p=stage.GetPrimAtPath(descriptor()['pads'][0]['path'])
    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError,match='disabled'):verify(stage)


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
@pytest.mark.parametrize('seed',[50,99])
def test_original_heldout_props_reopen_without_regenerating_training_geometry(tmp_path,difficulty,seed):
    row=next(r for r in original_heldout() if r['difficulty']==difficulty and r['seed']==seed)
    original=json.loads(base64.b64decode(row['source_bytes_base64']))
    case=original['config']['source_case']
    stage=Usd.Stage.CreateInMemory();pads(stage)
    task=build_props(stage,tmp_path,difficulty,seed,config('station'),case)
    assert signature(task['case'])==signature(case)
    path=tmp_path/'original-heldout.usda';stage.GetRootLayer().Export(str(path))
    reopened=Usd.Stage.Open(str(path))
    assert reopened.GetPrimAtPath('/World/CP09/Work/BoardFace') and verify(reopened)
    mesh=UsdGeom.Mesh(reopened.GetPrimAtPath('/World/CP09/Work/ActualInk'))
    assert len(mesh.GetPointsAttr().Get())>0


def board(**kwargs):
    return dict(position_m=[0,.065,.784],rotation=np.eye(3).tolist(),machine_offset_m=.15,
        storage_clip_removed=True,foot_table_contact=True,latches={'Left':False,'Right':False},
        release_pending={'Left':False,'Right':False},**kwargs)


def test_cp09_board_release_and_source_boundaries():
    facts=BoardFacts();raw=board();raw['position_m']=[0,.065,.914];raw['latches']['Right']=True
    assert not facts.sample(raw)
    assert facts.extracted
    raw=board();raw['machine_offset_m']=0
    for _ in range(24):assert not facts.sample(raw)
    assert facts.placed
    # Exact original source minimum clearance is strict >12mm; no novel dock threshold.
    assert facts.sample(board())
    raw=board();raw['release_pending']['Right']=True
    assert not facts.sample(raw)


def row(tick):
    return dict(tick=tick,simulation_time_s=tick/120,annotation_applied=tick>=480,board=board(),
        eraser=dict(position_m=[0,0,0],initial_position_m=[0,0,0],latched=False,tray_contact=True),
        pickup_events=[],cap_events=[],ink_contacts=[])


def test_cp09_eof_audit_full_recompute_and_tamper(tmp_path):
    audit=Audit(tmp_path/'trace.jsonl');audit.append(dict(tick=0,contract=CONTRACT,source_case=make_case('medium',0)[0]))
    for tick in range(1,501):audit.append(row(tick))
    evidence=NativeEvidence('CP09',selected('medium').instance_id,500,dict(difficulty='medium',seed=0,audit=audit.receipt()))
    assert score(evidence).value==0
    with (tmp_path/'trace.jsonl').open('ab') as stream:stream.write(b'\n')
    with pytest.raises(ValueError,match='digest'):score(evidence)
    audit.close()


def test_cp09_production_scorer_rejects_unreceipted_progress():
    with pytest.raises(ValueError,match='append-only'):
        score(NativeEvidence('CP09','forged',0,dict(cp09_progress={})))


def test_cp09_self_consistent_station_proof_cannot_replace_trusted_anchors(tmp_path):
    from mind2act_eval.tasks.CP09.binding import validate_station
    import hashlib
    station=tmp_path/'station.usda';station.write_text('arbitrary station')
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(source_id='test',station_config_sha256='config',
        scene_sha256=hashlib.sha256(station.read_bytes()).hexdigest(),source_h5_sha256='h5',original_source_scene={'sha256':'source'},
        allowed_projection=dict(removed_prim_roots=['/World/CP09'],all_other_authored_prim_subtrees_equal=True))))
    relocation=tmp_path/'relocation.json';relocation.write_text('trusted source mapping')
    assets=SimpleNamespace(resolve=lambda key:{'station':station,'station_proof':proof,'station_relocation':relocation}[key])
    with pytest.raises(RuntimeError,match='authoritative'):
        validate_station(dict(source_id='test'),assets,'config')
    anchored=dict(source_id='test',h5_sha256='h5',source_scene_sha256='source',
        station_relocation_sha256=hashlib.sha256(relocation.read_bytes()).hexdigest(),station_sha256=hashlib.sha256(station.read_bytes()).hexdigest(),
        station_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    validate_station(anchored,assets,'config')
    relocation.write_text('self reported replacement target')
    with pytest.raises(ValueError,match='anchors'):validate_station(anchored,assets,'config')
    relocation.write_text('trusted source mapping')
    proof.write_text('{}')
    with pytest.raises(ValueError,match='anchors'):validate_station(anchored,assets,'config')
