import copy
import json
from fractions import Fraction
from types import SimpleNamespace as NS

import pytest

from coding_payload_fixture import valid_input

from mind2act_eval import cli
from mind2act_eval.budgets import (AGENT_BUDGET_VERSION, E2E_BUDGET_VERSION,
    DecisionBudget, CallBudgetExhausted, selected_limits, validate_execution_limits)
from mind2act_eval.contracts import ConsumerKind, Termination
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.training_budgets import quantile, summarize, bind_case_budgets
from mind2act_eval.release_contract import ReleaseContractError, load_release
from test_runner import fixture
from test_task_shards import task_release, save
from test_instance_catalog import catalog_release
from test_runtime_release import release


def inventory():
    return dict(index_sha256='a'*64, tasks=[dict(task_id='PR1', manifest_sha256='b'*64)],
        episodes=[dict(task_id='PR1', difficulty='easy', episode_id=str(n),
                       expected_sha256=str(n)*64, rows=n*30, frequency_hz=30,
                       size_matches_manifest=True) for n in range(1, 11)])


def test_p90_interpolation_and_control_rounding_are_exact():
    assert quantile([Fraction(i) for i in range(1, 11)], 9, 10) == Fraction(91, 10)
    table = summarize(inventory(), wall_slowdown='60')
    row = table['cells'][0]
    assert row['p50_seconds'] == 5.5 and row['p90_seconds'] == 9.1
    assert row['control_steps'] == 328 and row['physics_ticks'] == 1312
    assert row['simulated_seconds_exact'] == '164/15' and row['wall_s'] == 1800
    # Already aligned values must not gain an extra tick through float noise.
    data = inventory()
    for episode in data['episodes']:
        episode['rows'] = 100
    assert summarize(data)['cells'][0]['physics_ticks'] == 480


def test_frequencies_change_duration_and_bad_statistics_cannot_bind():
    data = inventory()
    for episode in data['episodes']:
        episode['frequency_hz'] = 15
    assert summarize(data)['cells'][0]['p90_seconds'] == 18.2
    table = summarize(data)
    with pytest.raises(ValueError, match='unfrozen wall'):
        bind_case_budgets(dict(instances=[dict(task_id='PR1', difficulty='easy')]), table)
    data['episodes'] = data['episodes'][:2]
    assert 'fewer_than_10_experts' in summarize(data)['cells'][0]['flags']


def test_binding_preserves_published_limits_and_input_bytes():
    table = summarize(inventory(), wall_slowdown='60')
    # Ten samples are enough, but this synthetic population intentionally isn't 50.
    table['cells'][0]['flags'] = []
    original = dict(instances=[dict(task_id='PR1', difficulty='easy',
                                   limits=dict(physics_ticks=11520, wall_s=7200, retries=0))])
    before = copy.deepcopy(original)
    bound = bind_case_budgets(original, table)
    assert original == before
    row = bound['instances'][0]
    assert row['limits'] == original['instances'][0]['limits']
    assert row['limits_e2e'] == dict(physics_ticks=1312, wall_s=1800, retries=0)
    assert selected_limits(row, 'e2e', E2E_BUDGET_VERSION) == dict(physics_ticks=1312, wall_s=7200, retries=0)
    assert selected_limits(row, 'coding_agent', AGENT_BUDGET_VERSION) == dict(physics_ticks=5248, wall_s=21600, retries=0)
    assert selected_limits(row) == row['limits']
    with pytest.raises(ValueError, match='overwrite'):
        bind_case_budgets(bound, table)


def version_task(task_release):
    spec, args, _, _, source, shard = task_release
    for row in source['instances'] + shard['instances']:
        row.update(limits_e2e=dict(physics_ticks=400, wall_s=1800, retries=0),
                   budget_version_e2e=E2E_BUDGET_VERSION)
    save(task_release)
    return spec, args


def test_real_case_admission_selects_policy_and_child_keeps_version(task_release):
    spec, args = version_task(task_release)
    args.case_id, args.spec = spec.instance_id, None
    args.adapter, args.gpu = 'mind2act_eval.adapters.hold:create', 0
    args.policy_kind, args.budget_version = 'e2e', E2E_BUDGET_VERSION
    args.physics_ticks, args.wall_s, args.retries = 400, 7200, 0
    job = cli._job(args)
    assert job.budget_version == E2E_BUDGET_VERSION
    assert '--policy-kind' in job.argv and E2E_BUDGET_VERSION in job.argv
    args.physics_ticks = 1200
    with pytest.raises(ReleaseContractError, match='limits differ'):
        cli._job(args)
    args.policy_kind, args.budget_version = 'coding_agent', AGENT_BUDGET_VERSION
    args.physics_ticks, args.wall_s = 1600, 21600
    assert cli._job(args).physics_ticks == 1600
    args.wall_s = 1800
    with pytest.raises(ReleaseContractError):
        cli._job(args)


def test_source_and_ready_shard_e2e_budget_must_match(task_release):
    version_task(task_release)
    task_release[-1]['instances'][0]['limits_e2e']['physics_ticks'] = 404
    save(task_release)
    with pytest.raises(ReleaseContractError, match='source index'):
        load_release(task_release[1].asset_root)








def test_e2e_wall_can_grow_but_never_shrink_or_replace_physics():
    expected = dict(physics_ticks=400, wall_s=1800, retries=0)
    validate_execution_limits(NS(physics_ticks=400, wall_s=3600., retries=0), expected,
                              allow_longer_wall=True)
    for wall in (1799, float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            validate_execution_limits(NS(physics_ticks=400, wall_s=wall, retries=0), expected,
                                      allow_longer_wall=True)


def test_runner_stops_before_decision_151_and_retains_valid_score(tmp_path):
    from mind2act_eval.adapters.demo import DemoAdapter
    spec, task, ctx, _ = fixture(tmp_path)
    class Agent(DemoAdapter):
        def bind_decision_budget(self, budget):
            self.budget = budget
        def act(self, observation):
            self.budget.decision()
            return super().act(observation)
    result = run_episode(spec, task, ctx, Agent(), limits=EpisodeLimits(800, 30,
                         'coding_agent', AGENT_BUDGET_VERSION))
    assert result.infra_valid and result.score == 25
    assert result.termination == Termination.CALL_BUDGET and result.ticks == 600
    assert result.calls_used == result.calls_limit == 150
    assert result.decision_retry_limit == 12
    saved = json.loads((tmp_path/'result.json').read_text())
    assert saved['budget_version'] == AGENT_BUDGET_VERSION
    assert 'provider_attempt_limit' not in saved and 'provider_attempts_used' not in saved
    assert json.loads((tmp_path/'run-budget.json').read_text())['calls_used'] == 150


def test_e2e_can_exceed_150_action_requests_and_has_no_call_cap(tmp_path):
    from mind2act_eval.adapters.demo import DemoAdapter
    spec, task, ctx, _ = fixture(tmp_path)
    result = run_episode(spec, task, ctx, DemoAdapter(), limits=EpisodeLimits(604, 30,
                         'e2e', E2E_BUDGET_VERSION))
    assert result.infra_valid and result.ticks == 604
    assert result.calls_limit is None and result.budget_version == E2E_BUDGET_VERSION


def test_agent_without_framework_accounting_fails_before_actions(tmp_path):
    from mind2act_eval.adapters.demo import DemoAdapter
    spec, task, ctx, events = fixture(tmp_path)
    result = run_episode(spec, task, ctx, DemoAdapter(), limits=EpisodeLimits(800, 30,
                         'coding_agent', AGENT_BUDGET_VERSION))
    assert not result.infra_valid and result.calls_used == 0
    assert 'decision accounting' in result.error and not events


def test_failed_network_calls_do_not_spend_completed_decisions(tmp_path):
    from mind2act_eval.adapters.coding_agent_client import LoggedNativeClient
    class Native:
        model, reasoning_effort, api_style, max_retries = 'fixture', 'medium', 'responses', 1
        timeout_s, hard_timeout_s, api_key = 1., 2., ''
        def complete(self, messages, tools):
            for _ in range(2):
                try:
                    return self._complete_fn(messages, tools)
                except RuntimeError:
                    pass
            raise RuntimeError('logical request failed')
    path = tmp_path/'ledger.json'
    path.write_text(json.dumps(dict(schema_version=1, limit=2000, attempts=[])))
    native = Native()
    client = LoggedNativeClient(native, ledger=path, total_limit=2000, episode='fixture',
                                episode_limit=1, attempt_dir=tmp_path/'attempts')
    calls = DecisionBudget(2)
    client.bind_decision_budget(calls)
    client._request_once = lambda *args: (_ for _ in ()).throw(RuntimeError('failed provider'))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='logical request failed'):
            native.complete(*valid_input())
    assert calls.used == 0 and native.max_retries == 12
    client._request_once = lambda *args: {'id':'complete'}
    for _ in range(2):
        assert native.complete(*valid_input())['id'] == 'complete'
    assert calls.used == 2
    with pytest.raises(CallBudgetExhausted):
        native.complete(*valid_input())
    assert len(json.loads(path.read_text())['attempts']) == 6


def test_scanner_uses_active_manifest_instead_of_globbing_retired_h5(tmp_path):
    import hashlib
    import h5py
    from mind2act_eval.training_budgets import scan_training_root
    task = tmp_path/'FR1'
    task.mkdir()
    active, retired = task/'active.hdf5', task/'retired.hdf5'
    for path, count in ((active, 90), (retired, 9000)):
        with h5py.File(path, 'w') as h:
            for side in ('left', 'right'):
                h.create_dataset(f'action/{side}_arm_joint_states', shape=(count, 6), dtype='f4')
                h.create_dataset(f'action/{side}_ee_joint_states', shape=(count, 1), dtype='f4')
            h.create_dataset('additional_info/frequency', data=30)
    manifest = task/'manifest.json'
    manifest.write_text(json.dumps(dict(rows=[dict(id='expert', difficulty='easy',
        destination=str(active), sha256='b'*64, bytes=active.stat().st_size)])))
    (tmp_path/'INDEX.json').write_text(json.dumps(dict(tasks=[dict(task='FR1', public_task='DR1',
        current_manifest=str(manifest), current_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())])))
    scanned = scan_training_root(tmp_path)
    assert len(scanned['episodes']) == 1
    assert scanned['episodes'][0]['task_id'] == 'DR1'
    assert scanned['episodes'][0]['seconds'] == 3


def test_parent_keeps_budget_version_and_persisted_calls_after_worker_failure(tmp_path):
    from mind2act_eval.batch import EpisodeJob, _invalid, _attach_sidecar_artifacts
    from test_batch import demo_spec
    spec = demo_spec()
    job = EpisodeJob(spec, ('explicit-worker',), ConsumerKind.MODEL, 1800, 0,
                     policy_kind='coding_agent', budget_version=AGENT_BUDGET_VERSION, physics_ticks=800)
    result = _invalid(job, Termination.WALL_TIMEOUT, 'supervisor timeout')
    assert result.calls_used is None and result.budget_version == AGENT_BUDGET_VERSION
    (tmp_path/'call-budget.json').write_text(json.dumps(dict(schema='mind2act_eval.call_budget.v1',
        task_id=spec.task_id, instance_id=spec.instance_id, budget_version=AGENT_BUDGET_VERSION,
        calls_used=7, calls_limit=150, decision_retry_limit=12)))
    result = _attach_sidecar_artifacts(result, tmp_path)
    assert result.calls_used == 7 and result.decision_retry_limit == 12
    assert not result.infra_valid and result.score is None


def test_parent_rejects_child_result_from_another_protocol():
    from mind2act_eval.batch import EpisodeJob, _validate_result_budget
    from mind2act_eval.contracts import Result
    from test_batch import demo_spec
    spec = demo_spec()
    job = EpisodeJob(spec, ('explicit-worker',), ConsumerKind.MODEL, 1800, 0,
                     policy_kind='e2e', budget_version=E2E_BUDGET_VERSION, physics_ticks=400)
    legacy = Result(spec.task_id, spec.instance_id, 25., Termination.COMPLETED, True, ConsumerKind.MODEL, True)
    with pytest.raises(ValueError, match='version differs'):
        _validate_result_budget(job, legacy)


def test_only_completed_decisions_are_recorded():
    snapshots = []
    meter = DecisionBudget(recorder=lambda row: snapshots.append(row.copy()))
    meter.ensure_available()
    assert snapshots == [] and meter.used == 0
    meter.decision()
    assert snapshots == [dict(calls_used=1, calls_limit=150, decision_retry_limit=12)]


def test_new_protocol_cannot_drop_original_budget_or_relabel_formula():
    from mind2act_eval.budgets import validate_e2e_fields
    row = dict(limits_e2e=dict(physics_ticks=400, wall_s=1800, retries=0),
               budget_version_e2e=E2E_BUDGET_VERSION)
    with pytest.raises(ValueError, match='original published'):
        validate_e2e_fields(row)
    table = summarize(inventory(), wall_slowdown='120')
    table['cells'][0]['flags'] = []
    table['cells'][0]['physics_ticks'] += 4
    with pytest.raises(ValueError, match='declared P90'):
        bind_case_budgets(dict(instances=[dict(task_id='PR1', difficulty='easy')]), table)


def test_committed_table_has_all_90_cells_and_exact_physics_wall_formulas():
    from pathlib import Path
    from math import ceil
    table = json.loads((Path(__file__).parents[1]/'manifests/budgets/budgets-e2e-p90x1.2.json').read_text())
    assert len(table['cells']) == 90 and sum(c['sample_count'] for c in table['cells']) == 4500
    assert len({(c['task_id'], c['difficulty']) for c in table['cells']}) == 90
    for cell in table['cells']:
        assert not cell['flags'] and cell['sample_count'] == 50
        steps = ceil(Fraction(cell['p90_seconds_exact']) * Fraction(6, 5) * 30)
        assert cell['physics_ticks'] == steps * 4
        assert cell['wall_s'] == max(1800, cell['physics_ticks'])




def test_mixed_budget_versions_cannot_share_a_batch(tmp_path, monkeypatch):
    from mind2act_eval import batch
    from test_batch import demo_spec
    monkeypatch.setattr(batch, 'resolve_resources', lambda *args: (1000, 1, None))
    old = batch.EpisodeJob(demo_spec(), ('old-worker',), ConsumerKind.MODEL, 1800, 0)
    new = batch.EpisodeJob(demo_spec(), ('new-worker',), ConsumerKind.MODEL, 1800, 0,
        policy_kind='e2e', budget_version=E2E_BUDGET_VERSION, physics_ticks=400)
    with pytest.raises(ValueError, match='different policy budget versions'):
        batch.run_batch([old, new], tmp_path/'mixed', require_shared=False)
    assert not (tmp_path/'mixed').exists()


def test_scanner_accepts_final_collection_relative_path_schema(tmp_path):
    import hashlib
    import h5py
    from mind2act_eval.training_budgets import scan_training_root
    task = tmp_path / 'CP01'
    task.mkdir()
    active = task / 'active.hdf5'
    with h5py.File(active, 'w') as stream:
        stream.create_dataset('action/arm', shape=(3165, 6), dtype='f4')
        stream.create_dataset('additional_info/frequency', data=30)
    manifest = task / 'manifest.json'
    manifest.write_text(json.dumps(dict(rows=[dict(identity='CP01_medium_episode_0001',
        difficulty='medium', relative_path='CP01/active.hdf5', sha256='b'*64,
        bytes=active.stat().st_size)])))
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path/'INDEX.json').write_text(json.dumps(dict(tasks=[dict(task='CP01',
        manifest_relative_path='CP01/manifest.json', manifest_sha256=manifest_sha)])))
    scanned = scan_training_root(tmp_path)
    assert scanned['tasks'][0]['index_manifest_sha256'] == manifest_sha
    assert scanned['episodes'][0]['episode_id'] == 'CP01_medium_episode_0001'
    assert scanned['episodes'][0]['seconds'] == 105.5
    assert scanned['episodes'][0]['size_matches_manifest'] is True


def test_runtime_budgets_use_all_90_e2e_cells_without_mutating_legacy_rows():
    from pathlib import Path
    table = json.loads((Path(__file__).parents[1]/'manifests/budgets/budgets-e2e-p90x1.2.json').read_text())
    assert AGENT_BUDGET_VERSION == 'coding-agent-e2e-x4-v1'
    assert len(table['cells']) == 90
    for cell in table['cells']:
        row = dict(limits=dict(physics_ticks=60000, wall_s=999, retries=0),
            limits_e2e=dict(physics_ticks=cell['physics_ticks'], wall_s=cell['wall_s'], retries=0),
            budget_version_e2e=E2E_BUDGET_VERSION)
        before = copy.deepcopy(row)
        assert selected_limits(row, 'coding_agent', AGENT_BUDGET_VERSION) == dict(
            physics_ticks=cell['physics_ticks']*4, wall_s=21600, retries=0)
        assert selected_limits(row, 'e2e', E2E_BUDGET_VERSION) == dict(
            physics_ticks=cell['physics_ticks'], wall_s=7200, retries=0)
        assert selected_limits(row) == before['limits'] and row == before
    with pytest.raises(ValueError, match='frozen'):
        selected_limits(dict(limits=row['limits']), 'coding_agent', AGENT_BUDGET_VERSION)
    with pytest.raises(ValueError):
        selected_limits(row, 'coding_agent', 'coding-agent-published-v1')


@pytest.mark.parametrize('kind,version,ticks,wall', [
    ('e2e', E2E_BUDGET_VERSION, 400, 7199),
    ('e2e', E2E_BUDGET_VERSION, 400, 7201),
    ('coding_agent', AGENT_BUDGET_VERSION, 1600, 21599),
    ('coding_agent', AGENT_BUDGET_VERSION, 1600, 21601),
])
def test_explicit_protocol_rejects_any_other_wall(task_release, kind, version, ticks, wall):
    spec, args = version_task(task_release)
    args.case_id, args.spec = spec.instance_id, None
    args.adapter, args.gpu = 'mind2act_eval.adapters.hold:create', 0
    args.policy_kind, args.budget_version = kind, version
    args.physics_ticks, args.wall_s, args.retries = ticks, wall, 0
    with pytest.raises(ReleaseContractError):
        cli._job(args)
