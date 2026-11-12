"""Run against the private original source snapshot on Linux; never native proof.

Set MIND2ACT_ORIGINAL_SNAPSHOT and MIND2ACT_ORIGINAL_REQUEST to the frozen
source files. Raw decks, object truth and solutions remain test/environment data.
"""
import base64
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind2act_eval.contracts import InstanceSpec,NativeEvidence,to_json
from mind2act_eval import original_test_inputs as source

TASKS=('M2','CP06','CP08','CP09')


@pytest.fixture(scope='module')
def originals():
    paths=[os.environ.get(k) for k in ('MIND2ACT_ORIGINAL_SNAPSHOT','MIND2ACT_ORIGINAL_REQUEST')]
    if not all(paths):pytest.skip('private frozen original source paths not configured')
    raw=Path(paths[0]).read_bytes()
    assert source.digest(raw)==source.SNAPSHOT_SHA256
    records=json.loads(raw)['records']
    source.request_records(paths[1])
    return records,Path(paths[1])


def prepare(tmp_path,record,request):
    task=record['task_id'];tier=record['difficulty'];slot=record['logical_slot']
    raw=base64.b64decode(record['source_bytes_base64']);original=json.loads(raw)
    (tmp_path/'raw.json').write_bytes(raw)
    metadata=dict(schema=source.SCHEMA,instance_set='test50-r5',split='test',logical_slot=slot,
        native_case_id=record['native_case_id'],source_document_id=original.get('case_id',original.get('instance_id')),
        source_seed=record['seed'],source_configuration_sha256=record['source_configuration_sha256'],
        source_index_sha256=record['source_index']['sha256'],source_request_sha256=source.REQUEST_SHA256,
        snapshot_sha256=source.SNAPSHOT_SHA256)
    roles=dict(test_source_request=request,test_source_raw=tmp_path/'raw.json')
    assets={key:key for key in roles}
    module=import_module('mind2act_eval.tasks.'+task+'.instances') if task!='M2' else None
    rest=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044]
    config=dict(test_source=metadata)
    if task=='M2':
        config['m2_mode']=source.MODE;version='1';source_id='original-source-test'
    elif task=='CP06':
        core=module.original_core(original)
        config.update(source_task=core,source_spec_sha256=metadata['source_configuration_sha256'],
            declared_task_sha256=original['binding']['config_sha256'],instruction=module.instruction(tier),
            instruction_source_sha256=module.INSTRUCTION_SHA,public_parameters=module.public_parameters(core),robot_rest=rest)
        assets.update({k:k for k in ('station','task_source','source_geometry_proof','x5')})
        version=module.CONFIG_VERSION;source_id=original['binding']['declared_version']
    elif task=='CP08':
        config.update(source_case=original['config']['source_case'],source_spec_sha256=metadata['source_configuration_sha256'],
            physics_budget=original['config']['physics_budget'],scene_sha256=original['config']['scene_sha256'],
            instruction=module.instructions(tier)['en'],robot_rest=rest,
            source_projection=dict(source_physics_hz=120,source_state_tick=0,source_action_tick=4,
                source_cue_hide_tick=round(original['config']['source_case']['cue_seconds']*120)))
        version='cp08-source-projection-v1';source_id=original['binding']['declared_version']
    else:
        config.update(source_case_sha256=module.signature(original['config']['source_case']),
            instruction=module.instruction(tier,'en'),robot_rest=rest,geometry_decision=module.DECISION,
            dock_rule_status='provisional_source_retrieval_facts',
            station_config_sha256=module.signature(module.config('station')))
        version=module.CONFIG_VERSION;source_id=original['binding']['declared_version']
    spec=InstanceSpec(task,slot,tier,source_id,version,'cpu-source-input-check','v1',
        seed=record['seed'],config=config,assets=assets)
    resolver=SimpleNamespace(resolve=lambda key:roles[key])
    if task=='M2':
        # Use the physical control targets accepted by the source-bound runtime.
        # Separate USD tests verify the actual authored geometry.
        n=original['layout']['rows']*original['layout']['cols']
        binding=dict(schema='m2.source-binding.v1',mode=source.MODE,identity={k:getattr(spec,k) for k in
            ('task_id','instance_id','difficulty','source_id','seed','asset_version')},
            source_manifest_sha256=metadata['source_configuration_sha256'],scene_sha256='a'*64,
            mechanism_sha256='2d5a516d615e32beafb1fa839b62dd2ff722fae0fe5b54cf2747ced782dc8e04',
            card_identities=deepcopy(original['layout']['card_types']),source_layout=deepcopy(original['layout']),
            source_board_offset_xy=original['board_offset_xy'],button_rest_m=[.8055]*n,
            faceup_quaternions_wxyz=[[math.cos(math.radians(2)/2),0.,math.sin(math.radians(2)/2),0.]]*n,original_source=source.audit_source(spec,resolver))
    else:binding=None
    return spec,resolver,binding


def check(spec,assets,binding):
    if spec.task_id=='M2':
        from mind2act_eval.tasks.M2.binding import validate
        validate(spec,binding)
    else:
        module=import_module('mind2act_eval.tasks.'+spec.task_id+'.instances')
        module.validate(spec);module.original_case(spec,assets)


@pytest.mark.parametrize('task',TASKS)
def test_all150_original_inputs_keep_identity_seed_and_exact_task_parameters(originals,tmp_path,task):
    records,request=originals
    rows=[r for r in records if r['task_id']==task]
    assert len(rows)==150
    for row in rows:
        spec,assets,binding=prepare(tmp_path,row,request)
        check(spec,assets,binding)
        assert spec.seed==row['seed'] and spec.instance_id==row['logical_slot']
        assert source.read(spec,assets)==json.loads(base64.b64decode(row['source_bytes_base64']))


@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mutation',['split','native','seed','raw_self_hash','geometry','implicit_training'])
def test_original_branch_rejects_cross_source_geometry_and_training_aliases(originals,tmp_path,task,mutation):
    records,request=originals
    row=next(r for r in records if r['task_id']==task and r['difficulty']=='medium')
    spec,assets,binding=prepare(tmp_path,row,request)
    c=to_json(spec.config)
    if mutation=='split':c['test_source']['split']='train'
    elif mutation=='native':c['test_source']['native_case_id']='another-source'
    elif mutation=='seed':
        spec=replace(spec,seed=spec.seed+1);c['test_source']['source_seed']=spec.seed
    elif mutation=='raw_self_hash':
        assets.resolve('test_source_raw').write_bytes(b'{}');c['test_source']['source_configuration_sha256']=source.digest(b'{}')
    elif mutation=='implicit_training':
        c.pop('test_source')
        if task=='M2':c['m2_mode']='clean'
    elif mutation=='geometry':
        if task=='M2':binding['card_identities'][0]='another-picture'
        elif task=='CP06':c['source_task']['all_objects'][0]['initial_position_m'][0]+=.001
        elif task=='CP08':c['source_case']['blocks'][0]['initial'][0]+=.001
        else:c['source_case_sha256']='a'*64
    with pytest.raises((ValueError,KeyError)):
        check(replace(spec,config=c),assets,binding)


def test_original_cp08_projection_cannot_borrow_selected_training_h5(originals,tmp_path):
    from mind2act_eval.tasks.CP08.instances import validate
    from mind2act_eval.tasks.CP08.projection import H5_SHA256
    records,request=originals;row=next(r for r in records if r['task_id']=='CP08')
    spec,_,_=prepare(tmp_path,row,request);c=to_json(spec.config)
    c['source_projection']=dict(source_physics_hz=120,source_state_tick=72,source_action_tick=76,
        source_h5_row=18,source_cue_hide_tick=360,h5_sha256=H5_SHA256[spec.difficulty])
    with pytest.raises(ValueError):validate(replace(spec,config=c))


def test_original_m2_identity_cannot_select_diagnostic_training_binding(originals,tmp_path):
    records,request=originals;row=next(r for r in records if r['task_id']=='M2')
    spec,assets,binding=prepare(tmp_path,row,request);c=to_json(spec.config)
    c['m2_mode']='diagnostic_training';binding['mode']='diagnostic_training'
    with pytest.raises(ValueError,match='requires source_bound_test'):
        check(replace(spec,config=c),assets,binding)


def initial_evidence(spec,assets,tmp_path,*,tamper=False):
    task=spec.task_id;c=to_json(spec.config)
    module=import_module('mind2act_eval.tasks.'+task+'.audit')
    a=module.Audit(tmp_path/'audit.jsonl')
    values=source.audit_source(spec,assets)
    if task=='CP06':
        initial=dict(simulation_time_s=0.,objects={o['path']:dict(position=o['initial_position_m'],
            finger_contacts=[False,False],pan_supported=dict(Left=False,Right=False),
            max_pan_local_radius_m=.01,min_pan_local_z_m=.02) for o in c['source_task']['all_objects']})
        a.append(dict(tick=0,raw=initial))
        values.update(difficulty=spec.difficulty,source_spec_sha256=c['source_spec_sha256'],
            source_task_sha256=c['declared_task_sha256'],source_task=c['source_task'])
        if tamper:values['source_task']['all_objects'][0]['grams']+=1
    elif task=='CP08':
        from mind2act_eval.tasks.CP08.physical import CONTACT_SCHEMA
        a.append(dict(tick=0,blocks={b['id']:dict(position=b['initial'],orientation_wxyz=[1.,0.,0.,0.],
            linear_velocity_m_s=[0.,0.,0.]) for b in c['source_case']['blocks']+c['source_case']['distractors']},
            contacts=[],contact_measurement_schema=CONTACT_SCHEMA,cue_visible=True,
            tcps={s:[0.,0.,1.] for s in ('Left','Right')},
            flange_poses={s:dict(position=[-.145,0.,1.],orientation_wxyz=[1.,0.,0.,0.]) for s in ('Left','Right')},
            joints={s:[0.]*6 for s in ('Left','Right')},
            measured_finger_positions_m={s:[.044,.044] for s in ('Left','Right')}))
        values.update(source_case=c['source_case'],source_projection=c['source_projection'])
        if tamper:values['source_case']['blocks'][0]['initial'][0]+=.001
    else:
        from mind2act_eval.tasks.CP09 import CONTRACT
        case=source.read(spec,assets)['config']['source_case']
        if tamper:case['target_paths_m'][0][0][0]+=.001
        a.append(dict(tick=0,contract=CONTRACT,source_case=case))
        values.update(difficulty=spec.difficulty,seed=spec.seed)
    values['audit']=a.receipt();a.close()
    return NativeEvidence(task,spec.instance_id,0,values)


@pytest.mark.parametrize('task',['CP06','CP08','CP09'])
@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_independent_scorer_accepts_original_zero_tick_audit_without_training_alias(originals,tmp_path,task,tier):
    records,request=originals
    row=next(r for r in records if r['task_id']==task and r['difficulty']==tier)
    spec,assets,_=prepare(tmp_path,row,request)
    scorer=import_module('mind2act_eval.tasks.'+task+'.scorer')
    assert scorer.score(initial_evidence(spec,assets,tmp_path)).value==0


@pytest.mark.parametrize('task',['CP06','CP08','CP09'])
def test_independent_scorer_rechecks_original_geometry_against_authentic_audit(originals,tmp_path,task):
    records,request=originals
    row=next(r for r in records if r['task_id']==task and r['difficulty']=='medium')
    spec,assets,_=prepare(tmp_path,row,request)
    scorer=import_module('mind2act_eval.tasks.'+task+'.scorer')
    with pytest.raises(ValueError,match='original test'):
        scorer.score(initial_evidence(spec,assets,tmp_path,tamper=True))
