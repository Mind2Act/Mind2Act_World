"""Ready case subsets retain the full campaign and exact execution budgets."""
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from mind2act_eval import cli
from mind2act_eval.contracts import to_json
from mind2act_eval.release_contract import ReleaseContractError, load_release
from mind2act_eval.scene_probe import spec_digest
from test_instance_catalog import catalog_release
from test_runtime_release import write, release


@pytest.fixture
def task_release(catalog_release):
    spec, args, release, catalog = catalog_release
    root = Path(args.asset_root)
    rows, source, counters = [], [], Counter()
    selected = None
    for original in catalog['instances']:
        row = dict(original)
        cell = row['task_id'], row['difficulty']
        seed = counters[cell] + 1000
        counters[cell] += 1
        row.update(seed=seed, logical_slot=row['case_id'], native_case_id=row['case_id']+'-native',
            source_configuration_sha256='a'*64, limits=dict(physics_ticks=1200,wall_s=300,retries=0))
        source.append({k:row[k] for k in ('case_id','task_id','difficulty','seed','logical_slot',
            'native_case_id','source_configuration_sha256','coding_agent_selected','limits')})
        if row['task_id'] == spec.task_id:
            config = to_json(spec.config)
            config['test_source'] = dict(instance_set='test50-r5',split='test',**{k:row[k]
                for k in ('logical_slot','native_case_id','source_configuration_sha256')})
            compiled = replace(spec, instance_id=row['case_id'], difficulty=row['difficulty'], seed=seed,config=config)
            relative = 'shard-' + str(len(rows)) + '.json'
            row['spec'] = dict(path=relative, sha256=write(root/relative,to_json(compiled)),
                               semantic_sha256=spec_digest(compiled))
            rows.append(row)
            if row['case_id'] == spec.instance_id:
                selected = compiled
    index = dict(schema='mind2act_eval.test_source_index.v1', instance_set='test50-r5',
        tasks=catalog['tasks'],cases_per_cell=50,coding_agent_per_cell=5,instances=source)
    shard = dict(schema='mind2act_eval.ready_case_catalog.v1',instance_set='test50-r5',
                 task_id=spec.task_id,expected_task_case_count=150,instances=rows)
    states = {t:dict(status='pending') for t in catalog['tasks']}
    states[spec.task_id] = dict(status='ready',catalog=dict(path='task.json',sha256=write(root/'task.json',shard)))
    manifest = dict(schema='mind2act_eval.task_shards.v1',instance_set='test50-r5',expected_case_count=2250,
        source_index=dict(path='source-index.json',sha256=write(root/'source-index.json',index)),tasks=states)
    release.pop('catalog')
    release['task_shards'] = dict(path='shards.json',sha256=write(root/'shards.json',manifest))
    write(root/'RELEASE.json',release)
    return selected,args,release,manifest,index,shard


def save(value):
    spec,args,release,manifest,index,shard = value
    root = Path(args.asset_root)
    manifest['source_index']['sha256'] = write(root/'source-index.json',index)
    manifest['tasks'][spec.task_id]['catalog']['sha256'] = write(root/'task.json',shard)
    release['task_shards']['sha256'] = write(root/'shards.json',manifest)
    write(root/'RELEASE.json',release)


def test_explicit_task_selection_keeps_full_campaign_denominator(task_release):
    spec,args,*_ = task_release
    release = load_release(args.asset_root)
    assert len(release.test_source_index) == 2250
    assert release.ready_case_tasks == (spec.task_id,)
    assert len(release.select_ready_cases(spec.task_id)) == 150
    assert len(release.select_ready_cases(spec.task_id,'coding-agent')) == 15
    for instance_set in ('test50-r5','coding-agent'):
        with pytest.raises(ReleaseContractError,match='incomplete'):
            release.select_cases(instance_set)


def test_ready_task_uses_standard_exact_case_admission(task_release):
    spec,args,*_ = task_release
    args.case_id,args.spec = spec.instance_id,None
    args.adapter = 'mind2act_eval.adapters.hold:create'
    args.gpu,args.wall_s,args.physics_ticks,args.retries = 0,300,1200,0
    assert cli._job(args).spec == spec


@pytest.mark.parametrize('status',['pending','hold'])
def test_unready_task_cannot_be_selected_or_resolved(task_release,status):
    spec,args,release,manifest,index,_ = task_release
    other = next(t for t in manifest['tasks'] if t != spec.task_id)
    manifest['tasks'][other]['status'] = status
    save(task_release)
    value = load_release(args.asset_root)
    assert value.select_ready_cases(other) == ()
    case = next(r['case_id'] for r in index['instances'] if r['task_id'] == other)
    args.case_id,args.spec = case,None
    with pytest.raises(ReleaseContractError,match='not ready'):
        cli._job(args)


@pytest.mark.parametrize('mutation',['missing_case','duplicate_seed','wrong_subset','missing_task_status',
                                    'empty_shard','source_mismatch','budget_mismatch','hold_with_catalog',
                                    'phase_mismatch','document_id_mismatch'])
def test_malformed_task_readiness_is_rejected(task_release,mutation):
    spec,args,release,manifest,index,shard = task_release
    if mutation == 'missing_case':
        index['instances'].pop()
    elif mutation == 'duplicate_seed':
        index['instances'][1]['seed'] = index['instances'][0]['seed']
    elif mutation == 'wrong_subset':
        index['instances'][0]['coding_agent_selected'] = not index['instances'][0]['coding_agent_selected']
    elif mutation == 'missing_task_status':
        manifest['tasks'].pop(next(t for t in manifest['tasks'] if t != spec.task_id))
    elif mutation == 'empty_shard':
        shard['instances'].clear()
    elif mutation == 'source_mismatch':
        shard['instances'][0]['source_configuration_sha256'] = 'b'*64
    elif mutation == 'budget_mismatch':
        shard['instances'][0]['limits'] = dict(physics_ticks=2400,wall_s=300,retries=0)
    elif mutation == 'hold_with_catalog':
        manifest['tasks'][spec.task_id]['status'] = 'hold'
    elif mutation == 'phase_mismatch':
        shard['instances'][0]['source_condition'] = dict(phase_ticks=13)
    elif mutation == 'document_id_mismatch':
        shard['instances'][0]['source_document_id'] = 'another-source'
    save(task_release)
    with pytest.raises(ReleaseContractError):
        load_release(args.asset_root)


def test_task_manifest_tampering_is_rejected(task_release):
    _,args,*_ = task_release
    (Path(args.asset_root)/'shards.json').write_text('{}')
    with pytest.raises(ReleaseContractError,match='SHA256'):
        load_release(args.asset_root)


def test_partial_ready_cases_do_not_require_a_complete_cell_or_task(task_release):
    spec,args,_,manifest,_,shard = task_release
    chosen = [r for r in shard['instances'] if r['case_id'] == spec.instance_id]
    chosen.extend(r for r in shard['instances'] if r['case_id'] != spec.instance_id)
    shard['instances'] = chosen[:42]
    manifest['tasks'][spec.task_id]['status'] = 'partial'
    save(task_release)
    value = load_release(args.asset_root)
    assert len(value.test_source_index) == 2250
    assert len(value.select_ready_cases(spec.task_id)) == 42
    assert len(value.select_ready_cases()) == 42
    with pytest.raises(ReleaseContractError,match='incomplete'):
        value.select_cases('test50-r5')
    args.case_id,args.spec = spec.instance_id,None
    args.adapter = 'mind2act_eval.adapters.hold:create'
    args.gpu,args.wall_s,args.physics_ticks,args.retries = 0,300,1200,0
    assert cli._job(args).spec == spec


@pytest.mark.parametrize('name,value',[('physics_ticks',600),('wall_s',60),('retries',1)])
def test_selected_case_requires_original_execution_budget(task_release,name,value):
    spec,args,*_ = task_release
    args.case_id,args.spec = spec.instance_id,None
    args.adapter = 'mind2act_eval.adapters.hold:create'
    args.gpu,args.wall_s,args.physics_ticks,args.retries = 0,300,1200,0
    setattr(args,name,value)
    with pytest.raises(ReleaseContractError,match='execution limits'):
        cli._job(args)


def test_selected_spec_seed_must_match_full_source_identity(task_release):
    spec,args,_,_,_,shard = task_release
    row = next(r for r in shard['instances'] if r['case_id'] == spec.instance_id)
    changed = replace(spec,seed=spec.seed+1)
    row['spec']['sha256'] = write(Path(args.asset_root)/row['spec']['path'],to_json(changed))
    row['spec']['semantic_sha256'] = spec_digest(changed)
    save(task_release)
    args.case_id,args.spec = spec.instance_id,None
    args.gpu,args.wall_s,args.physics_ticks,args.retries = 0,300,1200,0
    with pytest.raises(ReleaseContractError,match='seed differs'):
        cli._job(args)


@pytest.mark.parametrize('field',['native_case_id','source_configuration_sha256','logical_slot',
                                 'instance_set','split','missing'])
def test_actual_compiled_source_identity_must_match_selected_source_row(task_release,field):
    spec,args,_,_,_,shard = task_release
    config = to_json(spec.config)
    if field == 'missing':
        config.pop('test_source')
    else:
        config['test_source'][field] = 'b'*64 if field == 'source_configuration_sha256' else 'another-source'
    changed = replace(spec,config=config)
    row = next(r for r in shard['instances'] if r['case_id'] == spec.instance_id)
    row['spec']['sha256'] = write(Path(args.asset_root)/row['spec']['path'],to_json(changed))
    row['spec']['semantic_sha256'] = spec_digest(changed)
    save(task_release)
    args.case_id,args.spec = spec.instance_id,None
    args.gpu,args.wall_s,args.physics_ticks,args.retries = 0,300,1200,0
    with pytest.raises(ReleaseContractError,match='compiled test source identity'):
        cli._job(args)
