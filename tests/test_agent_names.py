"""New agent interfaces retain the previous release's budgets and inputs."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mind2act_eval import cli
from mind2act_eval.agent_names import agent_field
from mind2act_eval.budgets import AGENT_BUDGET_VERSION, E2E_BUDGET_VERSION, selected_limits
from mind2act_eval.contracts import (BuildContext, ConsumerKind, Health, InstanceSpec,
    NativeEvidence, Observation, Result, Score, Termination, to_json)
from mind2act_eval.instance_catalog import load_catalog
from mind2act_eval.release_contract import ReleaseConfig, ReleaseContractError
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.tasks import TaskDefinition
from mind2act_eval.sim.demo import DemoSimulator, DemoRobot
from mind2act_eval.adapters.demo import DemoAdapter


def document(root, name, data):
    p = root / name
    p.write_text(json.dumps(data))
    return dict(path=name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def grid(flag='agent_selected'):
    return [dict(case_id=f'E2-{tier}-{n}', task_id='E2', difficulty=tier, seed=n,
        logical_slot=f'{tier}-{n}', native_case_id=f'native-{tier}-{n}',
        source_configuration_sha256='a'*64, **{flag: n < 5},
        limits=dict(physics_ticks=1200, wall_s=300, retries=0),
        spec=dict(path='spec.json', sha256='a'*64, semantic_sha256='b'*64),
        assets={}, asset_closure=dict(path='closure.json', sha256='c'*64))
        for tier in ('easy', 'medium', 'hard') for n in range(50)]


@pytest.mark.parametrize('flag', ['agent_selected', 'coding_agent_selected'])
@pytest.mark.parametrize('count', ['agent_per_cell', 'coding_agent_per_cell'])
def test_real_catalogue_readers_and_selectors_accept_both_names(tmp_path, flag, count):
    from mind2act_eval.task_shards import load_source_index, load_task_shards
    rows = grid(flag)
    data = dict(schema='mind2act_eval.instance_catalog.v1', instance_set='test50-r5',
                cases_per_cell=50, tasks=['E2'], instances=rows, **{count: 5})
    original = json.dumps(data)
    descriptor = document(tmp_path, 'catalog.json', data)
    result = load_catalog(tmp_path, descriptor, tasks=['E2'])
    assert json.dumps(data) == original
    assert sum(r['agent_selected'] for r in result) == 15
    assert all(r['agent_selected'] is r['coding_agent_selected'] for r in result)
    release = ReleaseConfig(tmp_path, tmp_path/'RELEASE.json', 'test', tmp_path/'manifest.json',
        'a'*64, (), 'fixture', None, False, {}, catalog=result, test_source_index=result)
    assert release.select_cases('agent') == release.select_cases('coding-agent')
    assert release.select_ready_cases('E2', 'agent') == release.select_ready_cases('E2', 'coding-agent')
    index = dict(data, schema='mind2act_eval.test_source_index.v1')
    index_ref = document(tmp_path, 'index.json', index)
    assert len(load_source_index(tmp_path, index_ref, tasks=['E2'])) == 150
    # Mix old and new names across independently pinned source and ready documents.
    other = 'coding_agent_selected' if flag == 'agent_selected' else 'agent_selected'
    ready = dict(schema='mind2act_eval.ready_case_catalog.v1', instance_set='test50-r5',
                 task_id='E2', expected_task_case_count=150, instances=grid(other))
    shard = dict(schema='mind2act_eval.task_shards.v1', instance_set='test50-r5',
        expected_case_count=150, source_index=index_ref,
        tasks={'E2': dict(status='ready', catalog=document(tmp_path, 'ready.json', ready))})
    assert len(load_task_shards(tmp_path, document(tmp_path, 'shards.json', shard), tasks=['E2'])[0]) == 150
    # A conflicting selection cannot pass either ordinary or ready readers.
    rows[0].update(agent_selected=True, coding_agent_selected=False)
    with pytest.raises(ReleaseContractError, match='conflicting'):
        load_catalog(tmp_path, document(tmp_path, 'bad.json', data), tasks=['E2'])
    index['agent_per_cell'], index['coding_agent_per_cell'] = 5, 4
    with pytest.raises(ReleaseContractError, match='conflicting'):
        load_source_index(tmp_path, document(tmp_path, 'bad-index.json', index), tasks=['E2'])


def test_alias_conflicts_do_not_coerce_bool_to_int():
    with pytest.raises(ReleaseContractError):
        agent_field(dict(agent_selected=True, coding_agent_selected=1), 'agent_selected')


@pytest.mark.parametrize('kind', ['agent', 'coding_agent'])
def test_policy_limits_and_real_cli_keep_frozen_protocol(tmp_path, monkeypatch, kind):
    row = dict(limits=dict(physics_ticks=1200, wall_s=300, retries=0),
        limits_e2e=dict(physics_ticks=400, wall_s=1800, retries=0), budget_version_e2e=E2E_BUDGET_VERSION)
    assert AGENT_BUDGET_VERSION == 'coding-agent-e2e-x4-v1'
    assert selected_limits(row, kind, AGENT_BUDGET_VERSION) == dict(physics_ticks=1600, wall_s=21600, retries=0)
    with pytest.raises(ValueError):
        EpisodeLimits(1600, 21600, kind, E2E_BUDGET_VERSION)
    seen = []
    def worker(args):
        seen.append(EpisodeLimits(args.physics_ticks, args.wall_s, args.policy_kind, args.budget_version))
        return 0
    monkeypatch.setattr(cli, '_worker', worker)
    assert cli.main(['_worker', '--output', str(tmp_path), '--policy-kind', kind,
        '--budget-version', AGENT_BUDGET_VERSION, '--physics-ticks', '1600', '--wall-s', '21600']) == 0
    assert seen[0].policy_kind == kind
    valid = Result('demo', 'one', 0., Termination.PHYSICS_BUDGET, True, ConsumerKind.MODEL, True,
        policy_kind=kind, budget_version=AGENT_BUDGET_VERSION, calls_used=3, calls_limit=150, decision_retry_limit=12)
    assert Result(**to_json(valid)).policy_kind == kind
    with pytest.raises(ValueError):
        replace(valid, calls_limit=151)
    with pytest.raises(ValueError):
        replace(valid, decision_retry_limit=13)


def fixture(root):
    robot = DemoRobot()
    class Scene:
        def before_step(self, tick): pass
        def after_step(self, tick): pass
        def observe(self, tick): return Observation(tick, tick/120, **robot.observe())
        def evidence(self, tick): return NativeEvidence('demo', 'one', tick, {'ticks': tick})
        def health(self): return Health(True)
        def is_complete(self): return False
        def close(self): pass
    spec = InstanceSpec('demo', 'one', 'easy', 'public', 'v1', 'none', 'v1')
    return spec, TaskDefinition('demo', lambda spec, ctx: Scene(), lambda e: Score(25.)), BuildContext(DemoSimulator(), robot, None, root)


@pytest.mark.parametrize('kind', ['agent', 'coding_agent'])
def test_runner_enforces_150_decisions_and_retains_timeout_accounting(tmp_path, kind):
    from mind2act_eval.batch import EpisodeJob, _invalid, _attach_sidecar_artifacts
    class Agent(DemoAdapter):
        def bind_decision_budget(self, budget): self.budget = budget
        def act(self, observation):
            self.budget.decision()
            return super().act(observation)
    spec, task, ctx = fixture(tmp_path)
    result = run_episode(spec, task, ctx, Agent(), limits=EpisodeLimits(800, 30, kind, AGENT_BUDGET_VERSION))
    assert result.infra_valid and result.score == 25
    assert result.termination == Termination.CALL_BUDGET and result.ticks == 600
    assert result.calls_used == result.calls_limit == 150 and result.decision_retry_limit == 12
    job = EpisodeJob(spec, ('explicit-worker',), ConsumerKind.MODEL, 30, 0,
        policy_kind=kind, budget_version=AGENT_BUDGET_VERSION, physics_ticks=800)
    timed_out = _attach_sidecar_artifacts(_invalid(job, Termination.WALL_TIMEOUT, 'fixture'), tmp_path)
    assert not timed_out.infra_valid and timed_out.calls_used == 150


def test_payload_module_is_an_alias_and_preserves_rejections():
    from mind2act_eval.adapters import agent_payload, coding_payload
    assert agent_payload.validate_input is coding_payload.validate_input
    assert agent_payload.require_english is coding_payload.require_english
    for module in (agent_payload, coding_payload):
        with pytest.raises(ValueError): module.require_english('非英文')
        with pytest.raises(ValueError): module.validate_input([], [])


def test_runtime_environment_names_and_explicit_override(tmp_path, monkeypatch):
    from mind2act_eval.adapters.roboprobe_pin import resolve_source_root
    monkeypatch.delenv('MIND2ACT_AGENT_SOURCE_ROOT', raising=False)
    monkeypatch.delenv('MIND2ACT_CODING_AGENT_SOURCE_ROOT', raising=False)
    with pytest.raises(ValueError): resolve_source_root()
    for name in ('MIND2ACT_AGENT_SOURCE_ROOT', 'MIND2ACT_CODING_AGENT_SOURCE_ROOT'):
        monkeypatch.setenv(name, str(tmp_path))
        assert resolve_source_root() == tmp_path.resolve()
        monkeypatch.delenv(name)
    other = tmp_path/'other'; other.mkdir()
    monkeypatch.setenv('MIND2ACT_AGENT_SOURCE_ROOT', str(tmp_path))
    monkeypatch.setenv('MIND2ACT_CODING_AGENT_SOURCE_ROOT', str(other))
    with pytest.raises(ValueError): resolve_source_root()
    assert resolve_source_root(other) == other.resolve()


def test_installer_pins_and_old_entry_point(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('installer', root/'tools/install_agent_runtime.py')
    installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
    raw, lock = installer.source_lock()
    assert installer.digest(raw) == 'a70f8c497901a6aee695657a54988a547b5940ec30b6bc0631edc89b952e9e49'
    rows = json.loads((installer.DATA/'downloads.json').read_text())
    targets = {r['target'] for r in rows}
    assert set(lock['files']) - targets == {'curobo-source-hashes.json', *(
        'runtime/Assets/Robots/x5/configuration/X5A_temp_'+n+'.usd' for n in ('base','physics','robot','sensor'))}
    assert all(r['installed_sha256'] == lock['files'].get(r['target'], r['sha256']) for r in rows)
    # Exercise download hashing, asset copy, patch hashing and verify without network or vendor assets.
    data = tmp_path/'data'; data.mkdir(); out = tmp_path/'out'; assets = tmp_path/'assets'
    payload = b'original'; patched = b'approved patch'; sha = installer.digest
    fixture_lock = dict(commit='fixture', files={'runtime/source.py': sha(patched), 'robot.bin': sha(payload), 'curobo-source-hashes.json': sha(b'{}')})
    fixture_raw = json.dumps(fixture_lock).encode()
    monkeypatch.setattr(installer, 'DATA', data)
    monkeypatch.setattr(installer, 'source_lock', lambda:(fixture_raw, fixture_lock))
    (data/'curobo-source-hashes.json').write_bytes(b'{}'); (data/'patch.py').write_bytes(patched)
    (data/'downloads.json').write_text(json.dumps([
        dict(target='runtime/source.py', url='unused', sha256=sha(payload), installed_sha256=sha(patched), patched_file='patch.py'),
        dict(target='robot.bin', asset='robot.bin', sha256=sha(payload), installed_sha256=sha(payload))]))
    import io
    monkeypatch.setattr(installer, 'urlopen', lambda *a,**k:io.BytesIO(payload))
    robot = assets/'shared/robots/x5/robot.bin';robot.parent.mkdir(parents=True);robot.write_bytes(payload)
    assert installer.install(out, assets)['checked'] == 3
    (out/'robot.bin').write_bytes(b'mutated')
    with pytest.raises(ValueError): installer.verify(out)
    with pytest.raises(ValueError): installer.install(out, assets)


@pytest.mark.parametrize('flag', ['agent_selected', 'coding_agent_selected'])
def test_catalogue_producer_writes_new_names_and_reads_old_inputs(tmp_path, monkeypatch, flag):
    from mind2act_eval import cli
    from mind2act_eval.catalog_builder import freeze_catalog
    from mind2act_eval.contracts import VariationSpec
    rows = grid(flag)
    specs = {}
    for row in rows:
        name = row['case_id']+'.json'
        spec = InstanceSpec('demo', row['case_id'], row['difficulty'], 'public', 'v1', 'none', 'v1',
                            variation=VariationSpec('clean'))
        (tmp_path/name).write_text(json.dumps(to_json(spec)))
        specs[name] = spec
        row['spec_path'] = name
    monkeypatch.setattr(cli, 'read_spec', lambda path:specs[path.name])
    ref = freeze_catalog(tmp_path, rows)
    produced = json.loads((tmp_path/ref['path']).read_text())
    assert produced['agent_per_cell'] == produced['coding_agent_per_cell'] == 5
    assert all(row['agent_selected'] is row['coding_agent_selected'] for row in produced['instances'])
    assert len(load_catalog(tmp_path, ref, tasks=['demo'])) == 150


def test_old_installer_cli_and_extra_remain_available():
    import subprocess,sys,tomllib
    root = Path(__file__).resolve().parents[1]
    extras = tomllib.loads((root/'pyproject.toml').read_text())['project']['optional-dependencies']
    assert extras['agent'] == extras['coding-agent'] == extras['roboprobe']
    for name in ('install_agent_runtime.py', 'install_coding_agent.py'):
        result = subprocess.run([sys.executable,str(root/'tools'/name),'--help'],capture_output=True,text=True)
        assert result.returncode == 0
        assert '--out' in result.stdout and '--asset-root' in result.stdout and '--verify' in result.stdout
