"""Real shared capability guards for the frozen 300 held-out CP09/PR4 inputs."""
from dataclasses import replace
from importlib import resources
import json

import pytest

from mind2act_eval.contracts import ClockSpec, InstanceSpec, VariationSpec, to_json
from mind2act_eval.robot.task_collision_overlay import expected_descriptor, DECISION
from mind2act_eval.sim.physics_profile import profile_for_instance
from mind2act_eval.tasks.CP09.instances import validate_case_source
from mind2act_eval.tasks.PR4.source_admission import validate_case_source as validate_pr4_case
from mind2act_eval.robot.cp09_pad_geometry import ASSET_SHA256


ROWS = json.loads(resources.files('mind2act_eval.sim').joinpath('new15_source_bindings.json').read_text())['bindings']


def spec_for(row, mode='clean'):
    return InstanceSpec(row['task_id'], row['instance_id'], row['difficulty'], row['source_id'],
        row['config_version'], 'test-assets', row['protocol_version'], seed=row['seed'],
        clock=ClockSpec(**row['clock']), config=row['config'],
        variation=VariationSpec() if mode=='clean' else VariationSpec('random',row['instance_id'],'1'*64,'2'*64,'3'*64))


@pytest.mark.parametrize('task', ['CP09','PR4'])
def test_actual_heldout_case_reaches_shared_source_capability(task):
    row=next(v for v in ROWS.values() if v['task_id']==task)
    spec=spec_for(row)
    if task=='CP09':
        validate_case_source(spec)
        descriptor=expected_descriptor(spec)
        assert len(descriptor['pads'])==4
        assert descriptor['source_sha256']==ASSET_SHA256
    else:
        validate_pr4_case(spec)
        assert profile_for_instance(spec).spec is spec


@pytest.mark.parametrize('mode',['clean','random'])
def test_all_exact_original_cases_preserve_existing_capability_values(mode):
    assert len(ROWS)==300
    for row in ROWS.values():
        spec=spec_for(row,mode)
        if spec.task_id=='PR4':
            validate_pr4_case(spec)
            receipt=profile_for_instance(spec).receipt()
            assert receipt['requested']=={'gpu_dynamics':True,'broadphase':'GPU'}
            assert receipt['readbacks']==[]  # Never constructs or runs a World.
        else:
            validate_case_source(spec)
            descriptor=expected_descriptor(spec)
            assert [p['scale'] for p in descriptor['pads']]==[[.018,.003,.022]]*4
            assert {p['static_friction'] for p in descriptor['pads']}=={2.}


@pytest.mark.parametrize('task',['CP09','PR4'])
@pytest.mark.parametrize('mutation',['instance','seed','source','clock','raw_hash','case_hash','source_identity'])
def test_nearby_or_tampered_heldout_declaration_is_rejected(task,mutation):
    row=next(v for v in ROWS.values() if v['task_id']==task);spec=spec_for(row)
    cfg=to_json(spec.config)
    if mutation=='instance':spec=replace(spec,instance_id=spec.instance_id+'-unselected')
    elif mutation=='seed':spec=replace(spec,seed=spec.seed+1)
    elif mutation=='source':spec=replace(spec,source_id='other-source')
    elif mutation=='clock':spec=replace(spec,clock=ClockSpec(240,30,30))
    elif mutation=='raw_hash':
        cfg['test_source']['source_configuration_sha256']='0'*64;spec=replace(spec,config=cfg)
    elif mutation=='case_hash':
        cfg['source_case_sha256' if task=='CP09' else 'scene_sha256']='0'*64;spec=replace(spec,config=cfg)
    elif mutation=='source_identity':
        cfg['test_source']['native_case_id']='other-case';spec=replace(spec,config=cfg)
    with pytest.raises(ValueError):
        (validate_case_source if task=='CP09' else validate_pr4_case)(spec)


@pytest.mark.parametrize('mode',['clean','random'])
def test_exact_english_acquisition_label_retains_closed_source_binding(mode):
    from mind2act_eval.robot.task_collision_overlay import english_geometry_decision
    for row in ROWS.values():
        if row['task_id']!='CP09':continue
        spec=spec_for(row,mode);cfg=to_json(spec.config)
        cfg['geometry_decision']=english_geometry_decision(cfg['geometry_decision'])
        validate_case_source(replace(spec,config=cfg))
        assert expected_descriptor(replace(spec,config=cfg))['source_sha256']==ASSET_SHA256
        cfg['geometry_decision']+=' altered'
        with pytest.raises(ValueError):validate_case_source(replace(spec,config=cfg))
