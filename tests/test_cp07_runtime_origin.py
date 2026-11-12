"""Source qualification at formal CPU admission, before asset/native admission."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mind2act_eval.cli import read_spec
from mind2act_eval.contracts import to_json
from mind2act_eval import runtime_release


@pytest.mark.parametrize('mode',['clean','random'])
@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_actual_training_layout_formal_source_guard(tier,mode,tmp_path,monkeypatch):
    value=os.environ.get('CP07_TRAIN_LAYOUT_ROOT')
    if not value:pytest.skip('explicit frozen CP07 delivery required')
    root=Path(value);rows=json.loads((root/'CP07-CLEAN-INDEX.json').read_text())['instances']
    row=deepcopy(next(r for r in rows if r['difficulty']==tier))
    spec=read_spec(root/row['spec' if mode=='clean' else 'random_spec']['path'])
    origin=spec.config['fixed50_origin']
    row.update(instance_id=spec.instance_id,logical_slot=spec.instance_id,native_case_id=origin['native_case_id'],source_configuration_sha256=origin['source_configuration_sha256'])
    formal=tmp_path/'new15-release-v1';formal.mkdir()
    wrapper=NS(asset_name='new15-release-v1',asset_root=root,instances=[{'task_id':t} for t in runtime_release.NEW15_SOURCE_CLEAN_TASKS],
               catalog=[row],code_commit='a'*40,code_identity_checked=True,resolve_case=lambda key:row)
    monkeypatch.setattr(runtime_release,'load_release',lambda *a,**kw:wrapper)
    from mind2act_eval import instance_catalog
    class AssetBoundary(RuntimeError):pass
    def assets(asset_root,selected,clean):
        assert asset_root==root and selected is row and clean.instance_id==spec.instance_id
        raise AssetBoundary('source/spec checks passed; asset closure/native checks not substituted')
    monkeypatch.setattr(instance_catalog,'verify_case_assets',assets)
    args=NS(asset_root=str(formal),case_id=spec.instance_id,physics_ticks=row['limits']['physics_ticks'],wall_s=row['limits']['wall_s'],retries=0)
    with pytest.raises(AssetBoundary):runtime_release.validate_release_inputs(spec,args,require_code_identity=True)
    for field in ('source','seed','mode','origin','injected_test_source','asset_map'):
        changed=spec;config=to_json(spec.config)
        if field=='source':changed=replace(spec,source_id='unknown')
        elif field=='seed':changed=replace(spec,seed=spec.seed+1);row['seed']=changed.seed
        elif field=='mode':config['cp07_mode']='unknown'
        elif field=='origin':config['fixed50_origin']['source_configuration_sha256']='0'*64
        elif field=='asset_map':changed=replace(spec,assets=dict(spec.assets,station='unknown.usda'))
        else:config['test_source']=dict(origin)
        changed=replace(changed,config=config)
        with pytest.raises(ValueError):runtime_release.validate_release_inputs(changed,args,require_code_identity=True)
        row['seed']=spec.seed
