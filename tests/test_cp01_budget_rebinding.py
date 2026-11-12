"""Explicit CP01 fresh-output rebinding keeps immutable source inputs intact."""
import copy
import hashlib
import json
from fractions import Fraction
from math import ceil
from pathlib import Path

import pytest

from mind2act_eval.training_budgets import bind_case_budgets


def tables():
    root = Path(__file__).parents[1]/'manifests/budgets'
    current = json.loads((root/'budgets-e2e-p90x1.2.json').read_text())
    proof = json.loads((root/'cp01-budget-source-proof.json').read_text())
    previous = copy.deepcopy(current)
    previous.update(proof['previous_table_sources'])
    prior_cells = {c['difficulty']:c for c in proof['previous_cp01_cells']}
    previous['cells'] = [prior_cells[c['difficulty']] if c['task_id']=='CP01' else c
                         for c in previous['cells']]
    return previous, current


def document(previous):
    return bind_case_budgets(dict(instances=[dict(task_id=task, difficulty=tier,
        instance_id=f'{task}-{tier}', limits=dict(physics_ticks=50000, wall_s=43200, retries=0),
        instruction='Keep this exact text.', seed=7, spec=dict(path='same.json', sha256='a'*64))
        for task in ('CP01','PR1') for tier in ('easy','medium','hard')]), previous)


def test_rebind_changes_only_approved_cp01_limits_and_source_digest():
    from mind2act_eval.training_budgets import rebind_cp01_case_budgets
    previous, current = tables()
    original = document(previous)
    before = copy.deepcopy(original)
    output = rebind_cp01_case_budgets(original, previous_table=previous, table=current)
    assert original == before
    for old, new in zip(original['instances'], output['instances']):
        expected = copy.deepcopy(old)
        if old['task_id'] == 'CP01' and old['difficulty'] != 'easy':
            ticks = {'medium':15196, 'hard':22300}[old['difficulty']]
            expected['limits_e2e'] = dict(physics_ticks=ticks, wall_s=ticks, retries=0)
        assert new == expected
    expected_sha = hashlib.sha256(json.dumps(current,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert output['e2e_budget_source']['table_semantic_sha256'] == expected_sha
    with pytest.raises(ValueError, match='overwrite'):
        bind_case_budgets(original, current)


@pytest.mark.parametrize('tamper', ['other_cell','easy','old_row','source_pin','target'])
def test_rebind_rejects_any_unapproved_change(tamper):
    from mind2act_eval.training_budgets import rebind_cp01_case_budgets
    previous, current = tables()
    original = document(previous)
    if tamper == 'other_cell':
        next(c for c in current['cells'] if c['task_id']=='PR1')['p50_seconds'] += 1
    elif tamper == 'easy':
        next(c for c in current['cells'] if c['task_id']=='CP01' and c['difficulty']=='easy')['p50_seconds'] += 1
    elif tamper == 'old_row':
        original['instances'][1]['limits_e2e']['physics_ticks'] += 4
    elif tamper == 'source_pin':
        original['e2e_budget_source']['table_semantic_sha256'] = 'f'*64
    else:
        next(c for c in current['cells'] if c['task_id']=='CP01' and c['difficulty']=='medium')['physics_ticks'] = 15192
    with pytest.raises(ValueError):
        rebind_cp01_case_budgets(original, previous_table=previous, table=current)


def test_committed_cp01_budgets_pin_verified_restored_source_and_exact_rounding():
    table = json.loads((Path(__file__).parents[1]/'manifests/budgets/budgets-e2e-p90x1.2.json').read_text())
    assert table['training_index_sha256'] == '6036310c15f438e9c31c02009a039477eb432a0c625ba7603eaf805288cb5270'
    assert table['training_source']['finalization_sha256'] == '3b112ef89f526d37f4a91331dd89dec51571c181ad4095d91a48fc5f893773b3'
    cells = {c['difficulty']:c for c in table['cells'] if c['task_id']=='CP01'}
    for tier, exact, steps, ticks in [('easy','16081/300',1930,7720),
            ('medium','10551/100',3799,15196), ('hard','3097/20',5575,22300)]:
        cell = cells[tier]
        assert cell['p90_seconds_exact'] == exact
        assert cell['control_steps'] == steps
        assert cell['physics_ticks'] == cell['wall_s'] == ticks
        assert cell['sample_count'] == 50 and not cell['flags']
        assert cell['training_manifest_sha256'] == 'e9bf08fd2f4229da8d060c57e6e49f891d2c7fa56f2c27ab31fc039e9ac1d37b'


def test_archived_metadata_reproduces_all_90_cells_and_source_hashes(tmp_path):
    import gzip
    from mind2act_eval.training_budgets import summarize, write_budget_csv
    root = Path(__file__).parents[1]/'manifests/budgets'
    proof = json.loads((root/'cp01-budget-source-proof.json').read_text())
    compressed = (root/proof['inventory']['path']).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == proof['inventory']['gzip_sha256']
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == proof['inventory']['decompressed_sha256']
    inventory = json.loads(raw)
    previous, current = tables()
    old = {c['task_id']+'/'+c['difficulty']:c for c in previous['cells']}
    reproduced = summarize(inventory, old_budgets={k:c['published_limits'] for k,c in old.items()},
        base_budgets={k:c['old_base_limits'] for k,c in old.items()},wall_slowdown='120')
    assert reproduced['cells'] == current['cells']
    write_budget_csv(tmp_path/'reproduced.csv', current)
    assert (tmp_path/'reproduced.csv').read_bytes() == (root/'budgets-e2e-p90x1.2.csv').read_bytes()
    assert reproduced['training_index_sha256'] == current['training_index_sha256'] == proof['source_index_sha256']
    assert len(inventory['episodes']) == 4500
    assert sum('error' in r for r in inventory['episodes']) == 0
    assert sum(r['rows'] for r in inventory['episodes']) == proof['source_statistics']['action_frames']
    unchanged = [c for c in current['cells'] if c == old[c['task_id']+'/'+c['difficulty']]]
    assert len(unchanged) == proof['unchanged_complete_cells'] == 87
    for suffix, name in [('json','table_raw_sha256'),('csv','csv_sha256')]:
        content = (root/f'budgets-e2e-p90x1.2.{suffix}').read_bytes()
        assert hashlib.sha256(content).hexdigest() == proof[name]
        assert b'\r' not in content
    assert hashlib.sha256(json.dumps(current,sort_keys=True,separators=(',',':')).encode()).hexdigest() == proof['table_semantic_sha256']


def test_cp01_rebind_cli_requires_fresh_output_and_keeps_input_bytes(tmp_path):
    from mind2act_eval.training_budgets import main
    previous, current = tables()
    original = document(previous)
    source, old_table, new_table, target = [tmp_path/name for name in
        ('source.json','previous-table.json','table.json','fresh.json')]
    for path,value in [(source,original),(old_table,previous),(new_table,current)]:
        path.write_text(json.dumps(value))
    before = source.read_bytes()
    argv = ['--document',str(source),'--table',str(new_table),
            '--rebind-cp01-from',str(old_table),'--output',str(target)]
    main(argv)
    assert source.read_bytes() == before
    written = json.loads(target.read_text())
    assert written['instances'][1]['limits_e2e'] == dict(physics_ticks=15196,wall_s=15196,retries=0)
    with pytest.raises(SystemExit):
        main(argv)
