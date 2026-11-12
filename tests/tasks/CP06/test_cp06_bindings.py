from copy import deepcopy
import pytest
from mind2act_eval.tasks.CP06.instances import selected, instruction, validate_core, parse_original


@pytest.mark.parametrize('level,seed',[('easy',610000),('medium',620000),('hard',630000)])
def test_selected_tier_seed_and_delivered_instruction(level,seed):
    row=selected(level);validate_core(row['source_task'],level)
    assert row['seed']==seed and row['source_task']['candidate']['seed']==seed
    assert instruction(level) and instruction(level,'zh')
    assert row['source_spec_sha256']!=row['derived_spec_sha256']
    assert row['declared_task_sha256']!=row['actual_exported_task_sha256']
    assert row['original_task_layer_sha256']!=row['derived_task_layer_sha256']


def test_no_declared_hash_rewrite_or_wrong_spec_bypass():
    with pytest.raises(ValueError,match='H5 instance SHA'):parse_original(b'{}','easy')
    c=deepcopy(selected('medium')['source_task']);c['candidate']['seed']=610000
    with pytest.raises(ValueError):validate_core(c,'medium')
    with pytest.raises(ValueError):instruction('hard','old-external')


def runtime_spec(level='easy'):
    from math import radians
    from mind2act_eval.contracts import InstanceSpec
    from mind2act_eval.tasks.CP06.instances import SOURCE_ID,CONFIG_VERSION,INSTRUCTION_SHA,public_parameters
    row=selected(level)
    config={k:row[k] for k in ('source_spec_sha256','declared_task_sha256','original_task_layer_sha256',
        'derived_task_layer_sha256','prepared_scene_sha256','stand_sha256','composition_receipt_sha256',
        'resource_projection_proof_sha256')}
    config.update(source_task=row['source_task'],instruction=instruction(level),instruction_source_sha256=INSTRUCTION_SHA,
        public_parameters=public_parameters(row['source_task']),station_relocation_manifest='station_mapping',
        robot_rest=[radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044])
    return InstanceSpec(task_id='CP06',instance_id=row['instance_id'],difficulty=level,source_id=SOURCE_ID,
        config_version=CONFIG_VERSION,asset_version='approved-source',protocol_version='v1',seed=row['seed'],
        config=config,assets={k:'CP06/'+k for k in ('station','task_source','original_spec','station_mapping','projection_proof','x5')})


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_runtime_spec_keeps_original_and_resource_projection_bindings_separate(level):
    from mind2act_eval.tasks.CP06.instances import validate
    validate(runtime_spec(level))


@pytest.mark.parametrize('field,value',[('instruction','old external CP06 instruction'),
    ('stand_sha256','6c4b9ec4cd325f6e47d862b87e7dddbe9adc5e9a5aa0f04224c47d0f16a600a8'),
    ('source_spec_sha256','0'*64),('declared_task_sha256','0'*64),('station_relocation_manifest',None)])
def test_wrong_identity_instruction_or_projection_rejected_before_sdk_or_assets(field,value):
    from dataclasses import replace
    from types import SimpleNamespace
    from mind2act_eval.contracts import to_json
    from mind2act_eval.tasks.CP06.scene import build
    spec=runtime_spec();config=to_json(spec.config);config[field]=value
    def forbidden(*args):raise AssertionError('asset/SDK must not open for invalid spec')
    with pytest.raises(ValueError):build(replace(spec,config=config),SimpleNamespace(assets=SimpleNamespace(resolve=forbidden)))
