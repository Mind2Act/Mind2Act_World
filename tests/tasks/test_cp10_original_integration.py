"""Actual original-layout inputs through the CPU scene and independent scorer."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.contracts import InstanceSpec,NativeEvidence,to_json
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.CP10.scene import ArenaScene,validate_binding
from mind2act_eval.tasks.CP10.scorer import score


@pytest.fixture(params=['easy','medium','hard'])
def actual(request):
    value=os.environ.get('CP10_ORIGINAL_PROBE_ROOT')
    if not value:pytest.skip('private original CP10 probe root not configured')
    root=Path(value);tier=request.param
    path=root/'test-cases'/'CP10'/tier/('CP10_'+tier+'_clean_000')/'spec.json'
    spec=InstanceSpec.from_dict(json.loads(path.read_text()))
    assets=LocalAssets(root,spec.assets,spec=spec)
    return spec,assets,validate_binding(spec,assets)


def scene(actual,tmp_path):
    spec,assets,config=actual
    names=list(config['objects']);positions=np.array([config['objects'][n]['position'] for n in names])
    objects=SimpleNamespace(get_world_poses=lambda:(positions,np.tile([1.,0.,0.,0.],(len(names),1))),
        get_velocities=lambda:np.zeros((len(names),6)))
    q=np.array([config['arena']['initial_slide_x_m'],math.radians(config['arena']['initial_yaw_deg']),0.])
    launcher=SimpleNamespace(get_joint_positions=lambda:q,get_joint_velocities=lambda:np.zeros(3))
    result=ArenaScene(spec,SimpleNamespace(output_dir=tmp_path,assets=assets,sim=SimpleNamespace(contacts=SimpleNamespace(supported=lambda *a,**kw:False))),None,objects,launcher,names,
        dict(Slide=0,Yaw=1,Pitch=2),config,'CPU fixture')
    result.prefix.observe(result._measure(0))
    return result


def test_real_original_layout_reaches_scene_and_scores_empty_physical_prefix(actual,tmp_path):
    spec,_,config=actual;s=scene(actual,tmp_path)
    try:
        e=s.evidence(0)
        assert e.values['input_binding_kind']=='original_test'
        assert to_json(e.values['source_config'])==config
        result=score(e)
        assert result.value==0 and result.components['completion']['n']=={'easy':4,'medium':5,'hard':7}[spec.difficulty]
    finally:s.close()


@pytest.mark.parametrize('mutation',['case','config','request_hash','scene_hash','missing_audit','both_markers','split'])
def test_original_scorer_rechecks_exact_source_independently_of_valid_prefix(actual,tmp_path,mutation):
    spec,_,_=actual;s=scene(actual,tmp_path)
    try:
        values=to_json(s.evidence(0).values)
        declaration=values['original_test_inputs']
        if mutation=='case':declaration['spec']['instance_id']='CP10_other_clean_000'
        elif mutation=='config':values['source_config']['seed']+=1
        elif mutation=='request_hash':declaration['audit']['test_source_request_sha256']='b'*64
        elif mutation=='scene_hash':declaration['audit']['scene_sha256']='b'*64
        elif mutation=='missing_audit':values.pop('original_test_inputs')
        elif mutation=='both_markers':
            values.pop('original_test_inputs')
            values.pop('input_binding_kind')
        else:declaration['spec']['config']['test_source']['split']='train'
        with pytest.raises(ValueError):score(NativeEvidence('CP10',spec.instance_id,0,values))
    finally:s.close()


def test_removing_original_source_declaration_does_not_open_training_cohort(actual):
    spec,assets,_=actual;config=to_json(spec.config);config.pop('test_source')
    with pytest.raises(ValueError,match='selected source configuration'):
        validate_binding(replace(spec,config=config),assets)
