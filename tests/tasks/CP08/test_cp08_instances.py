from dataclasses import replace
import pytest
from mind2act_eval.tasks.CP08.instances import spec_for_selected, selected, validate, instructions
from mind2act_eval.contracts import to_json


def make(level='easy'):
    return spec_for_selected(level,instance_id=selected(level)['instance_id'],
        scene_asset='CP08/scene.usda',x5_asset='Robots/x5/ARX.usd',
        plan_asset='CP08/plan.png',idle_asset='CP08/idle.png',asset_version='source-pinned')


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_selected_training_instance_and_delivered_text(level):
    spec=make(level);validate(spec)
    assert spec.config['instruction']==instructions(level)['en']
    assert spec.config['source_case']['index']==2000
    assert spec.config['source_projection']['source_state_tick']==72


@pytest.mark.parametrize('field,value',[('instruction','old external CP08 instruction'),
    ('scene_sha256','0'*64),('source_spec_sha256','0'*64)])
def test_changed_binding_rejected(field,value):
    spec=make();config=to_json(spec.config);config[field]=value
    with pytest.raises(ValueError):validate(replace(spec,config=config))


def test_seed_and_asset_paths_cannot_be_bypassed():
    spec=make()
    with pytest.raises(ValueError):validate(replace(spec,seed=1))
    with pytest.raises(ValueError):validate(replace(spec,assets={'station':'/root/share/source.usda'}))
    config=to_json(spec.config);config['source_projection']['source_state_tick']=18
    with pytest.raises(ValueError):validate(replace(spec,config=config))
