"""Actual formal reader and consumed-cache integrity regressions for PR245."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path

import pytest

from mind2act_eval.cli import read_spec
from mind2act_eval.variations import file_digest
from mind2act_eval.variations.review import select_random






def test_actual_final150_formal_select_random_and_authority_negatives(monkeypatch,tmp_path):
    value=os.environ.get('CP07_TRAIN_LAYOUT_ROOT')
    if not value:pytest.skip('explicit cold CP07 delivery required')
    root=Path(value)
    import mind2act_eval.variations as variations
    monkeypatch.setattr(variations,'file_digest',lru_cache(maxsize=None)(variations.file_digest))
    rows=json.loads((root/'CP07-CLEAN-INDEX.json').read_text())['instances'];assert len(rows)==150
    selected=rows
    if os.environ.get('CP07_REVIEW_SAMPLE_ONLY'):
        selected=[next(r for r in rows if r['difficulty']==tier) for tier in ('easy','medium','hard')]
    for row in selected:
        spec=read_spec(root/row['spec']['path']);origin=spec.config['fixed50_origin']
        row=deepcopy(row);row.update(logical_slot=row['case_id'],native_case_id=origin['native_case_id'],source_configuration_sha256=origin['source_configuration_sha256'])
        manifest=root/row['variation_manifest']['path'];doc=json.loads(manifest.read_text());binding=doc['layouts'][row['case_id']]
        layout=manifest.parent/binding['path'];row['source_layout']=dict(path=str(layout.relative_to(root)),sha256=file_digest(layout))
        # The archived background's actual source slot remains mandatory.
        from mind2act_eval.sim import LocalAssets
        random=read_spec(root/row['random_spec']['path'])
        plan=variations.resolve(random,LocalAssets(root,dict(random.assets,variation_manifest=row['variation_manifest']['path'])))
        row['layout_slot']=plan.layout.slot
        result=select_random(root,{},row)
        assert result[0]==root/row['random_spec']['path'] and result[3]==plan.layout.slot
    # Exercise rejection through the actual formal reader with rehashed descriptors.
    from mind2act_eval.contracts import to_json
    from mind2act_eval.scene_probe import spec_digest
    for mutation in ('seed','config','assets','source','mode','test_source'):
        changed=to_json(spec)
        if mutation=='seed':changed['seed']+=1
        elif mutation=='assets':changed['assets']['station']='different.usda'
        elif mutation=='source':changed['source_id']='unknown'
        elif mutation=='mode':changed['config']['cp07_mode']='diagnostic_training'
        elif mutation=='test_source':changed['config']['test_source']=dict(origin)
        else:changed['config']['fixed50_origin']['source_configuration_sha256']='0'*64
        folder=tmp_path/mutation;folder.mkdir()
        (folder/'clean.json').write_text(json.dumps(changed))
        (folder/'random.json').write_text(json.dumps(to_json(spec)))
        for name in ('manifest.json','layout.json'):(folder/name).write_text('{}')
        altered=deepcopy(row)
        altered['seed']=changed['seed']
        for field,name in (('spec','clean.json'),('random_spec','random.json'),('variation_manifest','manifest.json'),('source_layout','layout.json')):
            altered[field]=dict(path=name,sha256=file_digest(folder/name))
        from mind2act_eval.contracts import InstanceSpec
        altered['spec']['semantic_sha256']=spec_digest(InstanceSpec.from_dict(changed))
        with pytest.raises(ValueError):select_random(folder,{},altered)
    legacy=os.environ.get('CP07_LEGACY_RANDOM_ROOT')
    if legacy:
        oldroot=Path(legacy);old=json.loads((oldroot/'catalog/CP07-published.json').read_text())
        oldrows=[r for r in old['instances'] if r['task_id']=='CP07'];assert len(oldrows)==150
        for tier in ('easy','medium','hard'):
            select_random(oldroot,{},next(r for r in oldrows if r['difficulty']==tier))
    # Index identity must be authenticated before fixed50_origin is trusted.
    from mind2act_eval.cp07_layout_identity import validate_identity
    for changed in (replace(spec,seed=spec.seed+1),replace(spec,source_id='unknown'),replace(spec,config=dict(spec.config,cp07_mode='diagnostic_training'))):
        with pytest.raises(ValueError):validate_identity(changed)
