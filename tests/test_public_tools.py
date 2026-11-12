"""Public installer and scene selection keep exact identities on CPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.scene_probe import ProbeInputs, spec_digest
from mind2act_eval.tasks.catalog import TASK_MODULES

ROOT = Path(__file__).parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_lock_matches_runtime():
    from mind2act_eval.adapters.coding_agent_pin import LOCK_SHA256
    assert hashlib.sha256((ROOT/'tools/coding_agent_runtime/source-lock.json').read_bytes()).hexdigest() == LOCK_SHA256


def test_installed_runtime_checksums_and_tamper_detection(tmp_path):
    source = os.environ.get('MIND2ACT_CODING_AGENT_SOURCE_ROOT')
    if not source:
        pytest.skip('requires the installed pinned coding-agent runtime')
    installer = tool('install_coding_agent')
    assert installer.verify(Path(source))['checked'] == 143
    # Hard-link unchanged files; replace one link before writing so the SDK stays intact.
    import shutil
    target = tmp_path/'runtime'
    shutil.copytree(source, target, copy_function=os.link)
    altered = target/'runtime/env/global_configs.py'
    original = altered.read_bytes(); altered.unlink(); altered.write_bytes(original+b'\n# changed\n')
    with pytest.raises(ValueError, match='global_configs'):
        installer.verify(target)


@pytest.fixture
def selection(tmp_path):
    entries=[]
    for task in TASK_MODULES:
        if task == 'demo':continue
        for tier in ('easy','medium','hard'):
            spec=InstanceSpec(task,task+'-'+tier,tier,'fixed-source','v1','none','v1',seed=97)
            path=tmp_path/(spec.instance_id+'.json');path.write_text(json.dumps(to_json(spec)))
            entries.append({'spec':path.name,'asset_root':'.'})
    manifest=tmp_path/'scenes.json';manifest.write_text(json.dumps(entries))
    return manifest


def test_all_ninety_probe_configs_bind_exact_specs(selection,tmp_path):
    checker=tool('scene_check');rows=checker.plan(selection)
    assert len(rows)==90
    for row in rows:
        spec=InstanceSpec.from_dict(json.loads(Path(row['spec']).read_text()))
        assert spec.seed==97 and row['probe']['source_spec_sha256']==spec_digest(spec)
        path=tmp_path/'probe.json';raw=json.dumps(row['probe']).encode();path.write_bytes(raw)
        assert ProbeInputs.read(path,hashlib.sha256(raw).hexdigest(),spec).is_source_clean


def test_missing_or_duplicate_cell_is_an_error(selection):
    checker=tool('scene_check');entries=json.loads(selection.read_text())
    selection.write_text(json.dumps(entries[:-1]))
    with pytest.raises(ValueError,match='missing cells'):checker.plan(selection)
    selection.write_text(json.dumps(entries+[entries[0]]))
    with pytest.raises(ValueError,match='distinct'):checker.plan(selection)


def test_failed_cells_remain_in_contact_sheet(tmp_path):
    from PIL import Image
    checker=tool('scene_check')
    rows=[{'task':task,'tier':tier,'status':'failed'} for task in sorted(TASK_MODULES) if task!='demo' for tier in ('easy','medium','hard')]
    checker.contact_sheet(rows,tmp_path)
    with Image.open(tmp_path/'first-frames.jpg') as picture:
        assert picture.size==(960,7950)


def test_scene_probe_cli_reaches_supervisor_without_model(tmp_path,monkeypatch):
    from mind2act_eval import cli,batch
    spec=InstanceSpec('PR1','fixed','easy','fixed-source','v1','none','v1')
    path=tmp_path/'spec.json';path.write_text(json.dumps(to_json(spec)))
    probe={'schema':'mind2act_eval.scene_probe.v1','projection':'source-clean-preview-v1',
           'task_id':'PR1','difficulty':'easy','source_spec_sha256':spec_digest(spec)}
    raw=json.dumps(probe).encode();config=tmp_path/'probe.json';config.write_bytes(raw)
    seen=[]
    def run(jobs,output,**kwargs):
        seen.extend(jobs)
        return output,[]
    monkeypatch.setattr(batch,'run_batch',run)
    assert cli.main(['run','--spec',str(path),'--asset-root',str(tmp_path),'--consumer','scene_probe',
                     '--gpu','0','--probe-config',str(config),'--probe-config-sha256',hashlib.sha256(raw).hexdigest(),
                     '--physics-ticks','600','--retries','0'])==0
    assert len(seen)==1 and seen[0].consumer_kind.value=='scene_probe'


def test_static_gpu_cli_preserves_cases_and_groups_tasks(tmp_path,monkeypatch):
    from mind2act_eval import cli,batch
    rows=[]
    for i,task in enumerate(['PR1','PR2','PR1','PR3']):
        spec=InstanceSpec(task,'case-'+str(i),'easy','fixed-source','v1','none','v1',seed=100+i)
        path=tmp_path/(spec.instance_id+'.json');path.write_text(json.dumps(to_json(spec)))
        rows.append(dict(spec=str(path),asset_root=str(tmp_path),backend='demo',retries=0))
    manifest=tmp_path/'jobs.json';manifest.write_text(json.dumps(rows))
    def run(jobs,output,**kwargs):
        assert [job.gpu for job in jobs]==[2,5,2,2]
        assert [job.spec.seed for job in jobs]==[100,101,102,103]
        assert kwargs['max_parallel']==2
        assert all(job.argv[job.argv.index('--gpu')+1]==str(job.gpu) for job in jobs)
        return output,[]
    monkeypatch.setattr(batch,'run_batch',run)
    assert cli.main(['batch','--manifest',str(manifest),'--gpus','2,5'])==0

from test_runtime_release import release
from test_instance_catalog import catalog_release, save_catalog


def test_catalog_probe_uses_diagnostic_ticks_but_still_binds_spec(catalog_release,tmp_path):
    from mind2act_eval.runtime_release import validate_release_inputs
    from mind2act_eval.release_contract import ReleaseContractError
    spec,args,descriptor,catalog=catalog_release
    row=next(r for r in catalog['instances'] if r['case_id']==spec.instance_id)
    row['limits']=dict(physics_ticks=12000,wall_s=3600,retries=0)
    save_catalog(args,descriptor,catalog)
    args.consumer='scene_probe';args.physics_ticks=600;args.retries=0;args.gpu=0
    raw=json.dumps(dict(schema='mind2act_eval.scene_probe.v1',projection='source-clean-preview-v1',
                        task_id=spec.task_id,difficulty=spec.difficulty,source_spec_sha256=spec_digest(spec))).encode()
    config=tmp_path/'probe.json';config.write_bytes(raw)
    args.probe_config=str(config);args.probe_config_sha256=hashlib.sha256(raw).hexdigest()
    assert validate_release_inputs(spec,args) is not None
    args.physics_ticks=601
    with pytest.raises(ValueError,match='scene_probe requires'):validate_release_inputs(spec,args)
    args.physics_ticks=600
    from dataclasses import replace
    with pytest.raises((ValueError,ReleaseContractError)):
        validate_release_inputs(replace(spec,seed=spec.seed+1),args)
