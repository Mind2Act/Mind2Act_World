"""CPU-entry selection must cover every task without launching external work."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).parents[1] / 'tools/check_final_scoring_cpu.py'
SPEC = importlib.util.spec_from_file_location('check_final_scoring_cpu', PATH)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


@pytest.fixture
def repository(tmp_path):
    paths = {path for group in entry.TASK_TESTS.values() for path in group}
    paths.update(entry.COMMON_TESTS)
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('def test_fixture(): pass\n')
    version = tmp_path / 'eval/scoring.py'
    version.parent.mkdir(parents=True)
    version.write_text('SCORING_VERSION = "mind2act-final-scoring-20261105"\n')
    tool = tmp_path / 'tools/check_final_scoring_cpu.py'
    tool.parent.mkdir()
    tool.write_text('# fixture entrypoint\n')
    authority = tmp_path / entry.RULES_PATH
    authority.parent.mkdir(parents=True, exist_ok=True)
    # The scoring implementation is the public rules authority.
    return tmp_path


def test_default_expands_all_thirty_tasks_and_every_task_tree_file(repository):
    extra = repository / 'tests/tasks/test_new_shared_case.py'
    extra.write_text('def test_shared(): pass\n')
    selection = entry.build_selection(repository)
    assert len(selection['by_task']) == 30
    assert set(selection['by_task']) == set(entry.TASK_IDS)
    expected = {path.relative_to(repository).as_posix()
                for path in (repository / 'tests/tasks').rglob('test_*.py')}
    assert expected <= set(selection['paths'])
    assert set(entry.COMMON_TESTS) <= set(selection['paths'])


def test_missing_task_or_declared_fixture_fails_instead_of_silent_partial_coverage(repository, monkeypatch):
    missing = entry.TASK_TESTS['PR1'][0]
    (repository / missing).unlink()
    with pytest.raises(ValueError, match='missing declared test'):
        entry.build_selection(repository)
    monkeypatch.delitem(entry.TASK_TESTS, 'DR4')
    with pytest.raises(ValueError, match='30-task manifest'):
        entry.build_selection(repository)


def test_subset_discovers_case_variants_without_pr1_pr10_prefix_overlap(repository):
    for name in ('test_PR1_new.py', 'test_pr10_not_selected.py'):
        (repository / 'tests' / name).write_text('def test_new(): pass\n')
    selection = entry.build_selection(repository, ['PR1'])
    assert selection['tasks'] == ['PR1']
    assert 'tests/test_PR1_new.py' in selection['paths']
    assert 'tests/test_pr10_not_selected.py' not in selection['paths']
    assert set(selection['by_task']) == {'PR1'}


def test_collect_only_uses_this_python_and_cannot_inherit_hidden_deselection(repository, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k never --ignore=tests/tasks')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(entry, 'git_state', lambda root: {'head': 'a'*40, 'dirty': True})
    def run(command, **kwargs):
        calls.append(command)
        assert command[:3] == [sys.executable, '-m', 'pytest']
        assert '--collect-only' in command
        assert 'PYTEST_ADDOPTS' not in kwargs['env']
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs.get('shell', False) is False
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess, 'run', run)
    receipt = tmp_path / 'receipt.json'
    assert entry.main(['--task','PR1','--collect-only','--receipt',str(receipt)], root=repository) == 0
    report = json.loads(receipt.read_text())
    assert len(calls) == 1 and report['mode'] == 'collect_only'
    assert report['git_dirty'] is True and report['fixed_head_verified'] is False
    assert report['tested_ref'].startswith('worktree:')
    assert report['selected_test_hashes'] and report['source_hashes']


def test_source_change_during_run_cannot_certify_the_head(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'git_state', lambda root: {'head': 'a'*40, 'dirty': False})
    def run(command, **kwargs):
        with (repository / 'eval/scoring.py').open('a') as stream:
            stream.write('# concurrent edit\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess, 'run', run)
    receipt = tmp_path / 'changed.json'
    assert entry.main(['--task','PR1','--receipt',str(receipt)], root=repository) == 2
    report = json.loads(receipt.read_text())
    assert report['pytest_exit_code'] == 0 and report['exit_code'] == 2
    assert report['worktree_changed_during_run'] and not report['fixed_head_verified']


def test_failing_pytest_exit_is_preserved(repository, monkeypatch):
    monkeypatch.setattr(entry, 'git_state', lambda root: {'head': 'a'*40, 'dirty': True})
    monkeypatch.setattr(entry.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1))
    assert entry.main(['--task','PR1'], root=repository) == 1


@pytest.mark.parametrize('name', ['original.py', 'native.py'])
def test_partial_credit_authority_change_invalidates_a_fixed_source_run(repository, monkeypatch, name):
    authority = repository / 'tests/fixtures/recipes' / name
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_text('# approved partial-credit rules\n')
    monkeypatch.setattr(entry, 'git_state', lambda root: {'head': 'a'*40, 'dirty': False})
    def run(command, **kwargs):
        authority.write_text('# changed during CPU execution\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess, 'run', run)
    assert entry.main(['--task', 'PR1'], root=repository) == 2
