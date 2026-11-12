"""Canonical build admission, actual CPU native-boundary probes and mode seams."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

from mind2act_eval.cli import read_spec
from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.CP07 import scene
from mind2act_eval.tasks.CP07.binding import ProfileBlocked


def delivered(tier='medium',mode='clean'):
    value=os.environ.get('CP07_TRAIN_LAYOUT_ROOT')
    if not value:pytest.skip('explicit frozen CP07 delivery required')
    root=Path(value);case=f'CP07_{tier}_clean_000'
    relative=(f'test-cases/CP07/{tier}/{case}/spec.json' if mode=='clean'
              else f'test50-random/CP07/specs/{tier}/{case}.json')
    return root,read_spec(root/relative)


@pytest.mark.parametrize('mode',['clean','random'])
@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_actual_canonical_build_reaches_pre_native_boundary(tier,mode,tmp_path):
    helper=os.environ.get('CP07_CPU_BUILD_HELPER')
    if not helper:pytest.skip('explicit read-only CPU native-boundary helper required')
    root,spec=delivered(tier,mode)
    sys.path.insert(0,helper)
    from cpu_runtime import construct
    manifest='test50-random/CP07/variation-manifest.json' if mode=='random' else None
    _,result=construct(spec,root,tmp_path/('build-'+tier+'-'+mode),Path(helper),manifest)
    report=os.environ.get('CP07_CANONICAL_REPORT_DIR')
    if report:
        out=Path(report);out.mkdir(parents=True,exist_ok=True)
        (out/(tier+'-'+mode+'.json')).write_text(json.dumps(dict(task_id=spec.task_id,instance_id=spec.instance_id,difficulty=tier,mode=mode,source_id=spec.source_id,result=result),indent=2)+'\n')
    assert result['status']=='NATIVE_BOUNDARY',result
    assert result['constructor_prefix_executed'] and result['physics_steps']==0
    assert result['variation_applied_in_constructor']==(mode=='random')


@pytest.mark.parametrize('mutation',['unknown_mode','missing_mode','source','seed','config',
                                    'assets','binding_sha','authority','legacy_training','legacy_test'])
def test_actual_build_rejects_unqualified_inputs_before_sensors(monkeypatch,mutation):
    root,spec=delivered();config=to_json(spec.config)
    if mutation=='unknown_mode':config['cp07_mode']='unknown'
    elif mutation=='missing_mode':config.pop('cp07_mode')
    elif mutation=='source':spec=replace(spec,source_id='unknown')
    elif mutation=='seed':spec=replace(spec,seed=spec.seed+1)
    elif mutation=='config':config['robot_rest'][0]+=.01
    elif mutation=='assets':spec=replace(spec,assets=dict(spec.assets,station='unknown.usda'))
    elif mutation=='binding_sha':config['cp07_binding_sha256']='0'*64
    elif mutation.startswith('legacy_'):
        config['cp07_mode']='diagnostic_training' if mutation=='legacy_training' else 'source_bound_test'
    else:
        from mind2act_eval import cp07_layout_identity as identity
        authority=Path(identity.__file__).parent/'tasks/CP07/config/train-layout-authority.json'
        original=Path.read_text
        def changed(path,*a,**kw):
            text=original(path,*a,**kw)
            if path==authority:
                doc=json.loads(text);doc['cases'][spec.instance_id]['config_sha256']='0'*64
                return json.dumps(doc)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    spec=replace(spec,config=config)
    def forbidden(*a,**kw):pytest.fail('unqualified source reached native sensors')
    monkeypatch.setattr(scene,'NativeSensors',forbidden)
    with pytest.raises((ValueError,ProfileBlocked,KeyError)):
        scene.build(spec,NS(assets=LocalAssets(root,spec.assets,spec=spec)))


@pytest.mark.parametrize('mode',['diagnostic_training','source_bound_test'])
def test_old_modes_still_reach_their_binding_guard(monkeypatch,mode):
    spec=InstanceSpec('CP07','old-case','easy','old-source','v1','v1','v1',config={'cp07_mode':mode})
    class ExistingGuard(RuntimeError):pass
    def existing(spec_arg,assets):
        assert spec_arg is spec
        raise ExistingGuard(mode)
    monkeypatch.setattr(scene,'load',existing)
    with pytest.raises(ExistingGuard,match=mode):scene.build(spec,NS(assets=object()))


@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_training_layout_facts_scorer_and_drive_use_training_protocol(tier):
    # Explicit fixture observations; no world, physics or native API is created.
    from test_physical import Trace
    from mind2act_eval.tasks.CP07.scorer import COUNTS
    t=Trace(tier);t.binding['mode']='train_layout_test'
    t.assemble()
    n=1
    assert t.result().components['completion']['n']==n
    t.press();t.state['hands_clear']=True;t.step()
    if tier!='easy':
        t.state['car'].update(p_m=[t.binding['car_start_x_m'],.010,.824],up=.90)
        t.step(6)
        assert t.scene.facts.route_valid
        assert any(len(d)==2 for d in t.drives)
    t.state['car'].update(p_m=[t.binding['finish_x_m'],0.,.824],up=1.)
    t.step()
    assert t.result().value==100
    assert t.scene.evidence(t.tick).values['binding_mode']=='train_layout_test'
