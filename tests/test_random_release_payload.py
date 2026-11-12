"""Exercise compiler -> released selection with real adapter resolution on CPU."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from mind2act_eval.contracts import to_json
from mind2act_eval.variations import clean_spec,file_digest
from mind2act_eval.variations.release import TASKS,build
from mind2act_eval.variations.review import select_random
from test_variations import write_resolvable


def inputs(tmp_path):
    root=tmp_path/'clean';root.mkdir()
    random,_,pool=write_resolvable(root)
    base=clean_spec(random);rows=[]
    for task in TASKS:
        for tier in ('easy','medium','hard'):
            clean=replace(base,task_id=task,difficulty=tier,instance_id=task+'-'+tier,seed=17)
            name=task+'-'+tier+'.json';path=root/name;path.write_text(json.dumps(to_json(clean)))
            rows.append(dict(task_id=task,difficulty=tier,instance_id=clean.instance_id,seed=17,
                             spec=dict(path=name,sha256=file_digest(path))))
    (root/'RELEASE.json').write_text(json.dumps(dict(instances=rows)))
    pool['layouts']={};(root/'variation-pool.json').write_text(json.dumps(pool))
    raw=json.loads((root/'layout.json').read_text());records=[]
    raw.update(task='E2',card_sha256='0'*64,public_config_sha256='1'*64,
               settle=dict(thresholds=dict(window_translation_m=.04,window_angle_deg=10)))
    for tier in ('easy','medium','hard'):
        raw['difficulty']=tier;raw['Appearance'].update(task='E2',difficulty=tier)
        for i in range(50):
            slot=f'E2_{tier}_clean_{i:03d}';raw['logical_slot']=slot
            path=root/(slot+'.json');path.write_text(json.dumps(raw))
            records.append(dict(task_id='E2',difficulty=tier,logical_slot=slot,path=str(path),
                                sha256=file_digest(path),generator_version='frozen-source-v1'))
    return root,dict(layouts=records)


def test_all_fifty_slots_pair_with_exact_clean_without_synthesizing_seeds(tmp_path):
    source,inventory=inputs(tmp_path)
    receipt=build(source,inventory,source,tmp_path/'payload',allow_missing=True)
    assert receipt['parsed']==150 and receipt['resolved']==3
    assert receipt['tasks']['E2']=='READY'
    assert receipt['tasks']['CP09']==receipt['tasks']['DR4']=='PENDING_DELIVERY'
    assert receipt['tasks']['CP10']=='FINAL_INPUT_MISSING'
    assert receipt['native_completed']==0 and receipt['published'] is False
    candidate=tmp_path/'candidate';shutil.copytree(source,candidate)
    shutil.copytree(tmp_path/'payload/files',candidate,dirs_exist_ok=True)
    release=json.loads((source/'RELEASE.json').read_text())
    release.update(json.loads((tmp_path/'payload/release-additions.json').read_text()))
    for tier in ('easy','medium','hard'):
        clean=next(r for r in release['instances'] if (r['task_id'],r['difficulty'])==('E2',tier))
        for i in (0,49):
            path,_,_,slot=select_random(candidate,release,clean,layout_slot=f'E2_{tier}_clean_{i:03d}')
            random=json.loads(path.read_text())
            assert random['seed']==17 and random['instance_id']==clean['instance_id']
            assert random['variation']['layout_id']==slot
            assert (source/clean['spec']['path']).read_bytes()==(candidate/clean['spec']['path']).read_bytes()


def test_missing_delivery_and_duplicate_slot_do_not_shrink_or_replace(tmp_path):
    source,inventory=inputs(tmp_path)
    with pytest.raises(ValueError,match='final delivery input missing'):
        build(source,inventory,source,tmp_path/'strict')
    inventory['layouts'][1]=dict(inventory['layouts'][0])
    with pytest.raises(ValueError,match='duplicate/unsafe layout slot'):
        build(source,inventory,source,tmp_path/'duplicate',allow_missing=True)
