#!/usr/bin/env python3
"""Run the explicitly enumerated final-scoring CPU fixtures, without a simulator or model launch.

Examples:
  python tools/check_final_scoring_cpu.py --collect-only
  python tools/check_final_scoring_cpu.py --task DR3 --receipt /tmp/dr3-cpu.json
  python tools/check_final_scoring_cpu.py --receipt /tmp/final-scoring-cpu.json

A subset or collection receipt is never a thirty-task completion certificate.
Dirty source is identified by its actual worktree hashes, never by HEAD alone.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCORING_VERSION = 'mind2act-final-scoring-20261105'
TASK_IDS = ('PR1',
 'PR2',
 'PR3',
 'PR4',
 'PR5',
 'PR6',
 'DR1',
 'DR2',
 'DR3',
 'DR4',
 'E1',
 'E2',
 'S1',
 'S2',
 'P1',
 'P2',
 'F1',
 'F2',
 'M1',
 'M2',
 'CP01',
 'CP02',
 'CP03',
 'CP04',
 'CP05',
 'CP06',
 'CP07',
 'CP08',
 'CP09',
 'CP10')

# Deliberate fixture removals/renames must update this explicit thirty-task manifest.
TASK_TESTS = {'PR1': ('tests/test_pr1_scene.py', 'tests/test_pr1_scorer.py'),
 'PR2': ('tests/test_pr2_diagnostic.py',
         'tests/test_pr2_scene.py',
         'tests/test_pr2_scorer.py',
         'tests/test_pr2_source_contacts.py'),
 'PR3': ('tests/test_pr3_scene.py', 'tests/test_pr3_scorer.py'),
 'PR4': ('tests/tasks/PR4/test_geometry.py',
         'tests/tasks/PR4/test_pr4_binding_audit.py',
         'tests/tasks/PR4/test_pr4_scorer.py',
         'tests/test_pr4_source_collision_binding.py'),
 'PR5': ('tests/tasks/PR5/test_evidence.py',
         'tests/tasks/PR5/test_scene.py',
         'tests/tasks/PR5/test_scorer.py',
         'tests/tasks/PR5/test_streaming.py'),
 'PR6': ('tests/tasks/PR6/test_pr6_rules.py',
         'tests/tasks/PR6/test_pr6_scene.py',
         'tests/test_pr6_common_input_hold.py'),
 'DR1': ('tests/test_dr1_completion.py', 'tests/test_dr1_scorer.py'),
 'DR2': ('tests/test_dr2_scorer.py',),
 'DR3': ('tests/tasks/DR3/test_capture_geometry.py',
         'tests/tasks/DR3/test_catalog.py',
         'tests/tasks/DR3/test_events.py',
         'tests/tasks/DR3/test_instances.py',
         'tests/tasks/DR3/test_measured_geometry.py',
         'tests/tasks/DR3/test_scene.py'),
 'DR4': ('tests/tasks/DR4/test_balance.py', 'tests/tasks/DR4/test_dr4_scene.py'),
 'E1': ('tests/tasks/E1/test_boundaries.py',
        'tests/tasks/E1/test_evidence.py',
        'tests/tasks/E1/test_scorer.py',
        'tests/tasks/E1/test_timeline.py',
        'tests/tasks/E1/test_transport_lifetime.py'),
 'E2': ('tests/tasks/test_e2_keys.py',
        'tests/tasks/test_e2_measurements.py',
        'tests/tasks/test_e2_scene.py',
        'tests/tasks/test_e2_scorer.py'),
 'S1': ('tests/tasks/S1/test_boundaries.py',
        'tests/tasks/S1/test_mechanism.py',
        'tests/tasks/S1/test_scorer.py',
        'tests/test_s1_formal_timeline.py'),
 'S2': ('tests/tasks/test_s2.py',),
 'P1': ('tests/tasks/P1/test_batched_rays.py',
        'tests/tasks/P1/test_boundaries.py',
        'tests/tasks/P1/test_mechanism.py',
        'tests/tasks/P1/test_native_occupancy.py',
        'tests/tasks/P1/test_rim_boundary.py',
        'tests/tasks/P1/test_sat_memory.py',
        'tests/tasks/P1/test_scorer.py'),
 'P2': ('tests/tasks/test_p2_build.py',
        'tests/tasks/test_p2_facts.py',
        'tests/tasks/test_p2_instances.py',
        'tests/tasks/test_p2_measurements.py',
        'tests/tasks/test_p2_scene.py',
        'tests/tasks/test_p2_scorer.py',
        'tests/tasks/test_p2_stages.py'),
 'F1': ('tests/test_f1_evidence_history.py', 'tests/test_f1_judge.py'),
 'F2': ('tests/tasks/test_f2_mechanism.py',
        'tests/tasks/test_f2_scene.py',
        'tests/tasks/test_f2_scorer.py'),
 'M1': ('tests/test_m1_evidence_history.py', 'tests/test_m1_policy_stop.py', 'tests/test_m1_scorer.py'),
 'M2': ('tests/tasks/M2/test_m2_native_construction.py',
        'tests/tasks/M2/test_m2_native_stage.py',
        'tests/tasks/M2/test_m2_physical.py',
        'tests/tasks/M2/test_m2_raw_replay.py'),
 'CP01': ('tests/tasks/test_cp01_port.py',
          'tests/tasks/test_cp01_scoring.py',
          'tests/tasks/test_cp01_tray_terminal.py',
          'tests/test_cp01_initial_layout.py'),
 'CP02': ('tests/tasks/test_cp02_port.py',),
 'CP03': ('tests/tasks/test_cp03_port.py',
          'tests/tasks/test_cp03_start_protocol.py',
          'tests/tasks/test_cp03_terminal_placement.py'),
 'CP04': ('tests/tasks/test_cp04_binding.py',
          'tests/tasks/test_cp04_build_binding.py',
          'tests/tasks/test_cp04_contact_ownership.py',
          'tests/tasks/test_cp04_diagnostic_clock.py',
          'tests/tasks/test_cp04_evaluation_clock.py',
          'tests/tasks/test_cp04_observer_order.py',
          'tests/tasks/test_cp04_recipe_screen.py',
          'tests/tasks/test_cp04_scene.py',
          'tests/tasks/test_cp04_scorer.py',
          'tests/tasks/test_cp04_shared_slew.py'),
 'CP05': ('tests/tasks/test_cp05_scene.py', 'tests/tasks/test_cp05_scorer.py'),
 'CP06': ('tests/tasks/CP06/test_cp06_bindings.py',
          'tests/tasks/CP06/test_cp06_mechanisms.py',
          'tests/tasks/CP06/test_cp06_physical.py',
          'tests/tasks/CP06/test_cp06_projection.py',
          'tests/tasks/CP06/test_cp06_scene.py',
          'tests/tasks/CP06/test_cp06_scorer.py'),
 'CP07': ('tests/tasks/CP07/test_canonical_build.py',
          'tests/tasks/CP07/test_native_construction.py',
          'tests/tasks/CP07/test_physical.py',
          'tests/tasks/CP07/test_source_rules.py',
          'tests/tasks/CP07/test_success_lock.py',
          'tests/tasks/test_cp07_original_geometry.py',
          'tests/tasks/test_cp07_original_test.py',
          'tests/test_cp07_public_recipe.py',
          'tests/test_cp07_recipe_release.py',
          'tests/test_cp07_review_followups.py',
          'tests/test_cp07_runtime_origin.py',
          'tests/test_cp07_train_layout.py'),
 'CP08': ('tests/tasks/CP08/test_cp08_cue_clock.py',
          'tests/tasks/CP08/test_cp08_instances.py',
          'tests/tasks/CP08/test_cp08_physical.py',
          'tests/tasks/CP08/test_cp08_projection_audit.py',
          'tests/tasks/CP08/test_cp08_scorer.py',
          'tests/tasks/CP08/test_cp08_tcp.py'),
 'CP09': ('tests/tasks/test_cp09_pickup.py',
          'tests/tasks/test_cp09_port.py',
          'tests/tasks/test_cp09_scorer.py',
          'tests/test_cp09_pad_geometry.py',
          'tests/test_cp09_robot_overlay.py',
          'tests/test_cp09_source_pad_restore.py'),
 'CP10': ('tests/tasks/test_cp10_original_geometry.py',
          'tests/tasks/test_cp10_original_inputs.py',
          'tests/tasks/test_cp10_original_integration.py',
          'tests/tasks/test_cp10_scene.py',
          'tests/tasks/test_cp10_scoring.py')}

COMMON_TESTS = (
    'tests/test_final_scoring.py',
    'tests/test_mind_act_caps.py',
    'tests/tasks/test_cp_caps.py',
    'tests/test_runner.py',
    'tests/test_task_completion.py',
    'tests/test_a_scene_completion.py',
    'tests/test_policy_budgets.py',
    'tests/test_cp01_budget_rebinding.py',
    'tests/test_coding_payload.py',
    'tests/test_coding_agent_transport.py',
    'tests/test_final_scoring_recipes.py',
    'tests/test_observation_closeout.py',
    'tests/test_contracts.py',
    'tests/test_sim.py',
    'tests/test_delivered_instructions.py',
    'tests/test_english_instruction_projection.py',
    'tests/test_common_input_protocol.py',
    'tests/test_final_scoring_cpu_entrypoint.py',
)
RULES_PATH = 'src/mind2act_eval/scoring.py'
HASH_SUFFIXES = {'.py', '.json', '.csv', '.toml', '.yaml', '.yml', '.usda', '.sh'}
SOURCE_DIRECTORIES = ('src', 'tests', 'tools', 'scripts', 'manifests',
                      'tests/fixtures/recipes/baseline')


def _is_test(path):
    name = path.name.lower()
    return name.endswith('.py') and (name.startswith('test_') or name.endswith('_test.py'))


def _belongs(path, task):
    relative = path.as_posix()
    directory_match = relative.startswith('tests/tasks/') and any(
        part.lower() == task.lower() for part in path.parts[2:-1])
    return directory_match or bool(re.match(r'test_'+task.lower()+r'(?:_|\.py$)', path.name.lower()))


def build_selection(root, tasks=None):
    """Validate the explicit roster, then include newly added matching fixtures."""
    root = Path(root)
    if set(TASK_TESTS) != set(TASK_IDS) or len(TASK_TESTS) != 30:
        raise ValueError('30-task manifest is incomplete or contains unknown tasks')
    selected = list(TASK_IDS) if tasks is None else list(dict.fromkeys(tasks))
    if not selected or set(selected)-set(TASK_IDS):
        raise ValueError('unknown or empty task selection')
    available = sorted(p.relative_to(root) for p in (root/'tests').rglob('*.py') if _is_test(p))
    by_task = {}
    for task in selected:
        declared = TASK_TESTS[task]
        if not declared:
            raise ValueError('empty declared task test list: '+task)
        for relative in declared:
            if not (root/relative).is_file():
                raise ValueError('missing declared test for '+task+': '+relative)
        discovered = [p.as_posix() for p in available if _belongs(p, task)]
        by_task[task] = sorted(set(declared)|set(discovered))
    for relative in COMMON_TESTS:
        if not (root/relative).is_file():
            raise ValueError('missing declared shared test: '+relative)
    paths = {relative for group in by_task.values() for relative in group}
    paths.update(COMMON_TESTS)
    # The default also includes generic, cross-task tests in tests/tasks/.
    # An unfamiliar filename can never silently disappear from full acceptance.
    extra = []
    if set(selected) == set(TASK_IDS):
        extra = [p.as_posix() for p in available if p.as_posix().startswith('tests/tasks/')]
        paths.update(extra)
    return dict(tasks=selected, by_task=by_task, shared=list(COMMON_TESTS),
                additional_task_tree_tests=sorted(set(extra)-set().union(*(set(v) for v in by_task.values()))),
                paths=sorted(paths))


def cpu_environment(environ):
    env = dict(environ)
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PXR_WORK_THREAD_LIMIT='1',
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return env


def git_state(root):
    """Read only local Git identity; never fetch, checkout or mutate the index."""
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                              capture_output=True, check=True, timeout=10).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain=v1', '--untracked-files=all'],
                                cwd=root, text=True, capture_output=True, check=True, timeout=10).stdout
        return dict(head=head, dirty=bool(status), status_sha256=hashlib.sha256(status.encode()).hexdigest())
    except (OSError, subprocess.SubprocessError):
        return dict(head=None, dirty=None, error='local Git identity could not be verified')


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_snapshot(root):
    """Hash actual worktree code/config and test helpers, including new files."""
    files = set()
    for directory in SOURCE_DIRECTORIES:
        for path in (root/directory).rglob('*'):
            if (path.is_file() and not {'__pycache__', '.pytest_cache'}.intersection(path.parts)
                    and (path.suffix.lower() in HASH_SUFFIXES or path.name == 'SOURCE_COMMIT')):
                files.add(path.relative_to(root).as_posix())
    for relative in ('pyproject.toml', 'pytest.ini', 'setup.cfg', RULES_PATH,
                     'tests/fixtures/recipes/original.py',
                     'tests/fixtures/recipes/native.py',
                     'manifests/budgets/training-lengths-final-collection.json.gz'):
        if (root/relative).is_file():
            files.add(relative)
    hashes = {relative: _file_sha(root/relative) for relative in sorted(files)}
    return dict(sha256=_mapping_digest(hashes), hashes=hashes)


def _scoring_version(root):
    tree = ast.parse((root/'src/mind2act_eval/scoring.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCORING_VERSION' for t in node.targets):
            value = ast.literal_eval(node.value)
            if value == SCORING_VERSION:
                return value
    raise ValueError('shared scoring_version differs from the final authority')


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=str.upper, choices=TASK_IDS, action='append',
                        help='Select one task; repeat for several. Default: all 30.')
    parser.add_argument('--collect-only', action='store_true', help='Collect fixtures without executing them.')
    parser.add_argument('--list', action='store_true', help='Print the exact selection without running pytest.')
    parser.add_argument('--receipt', type=Path, help='Write a local JSON receipt (prefer a path outside the checkout).')
    args = parser.parse_args(argv)
    root = Path(root).resolve()
    try:
        selection = build_selection(root, args.task)
        version = _scoring_version(root)
        if not (root/RULES_PATH).is_file():
            raise ValueError('final scoring authority is missing: '+RULES_PATH)
        before = source_snapshot(root)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    before_git = git_state(root)
    command = [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=']
    if args.collect_only:
        command.append('--collect-only')
    command.extend(selection['paths'])
    mode = 'manifest_only' if args.list else 'collect_only' if args.collect_only else 'cpu_fixtures'
    print('CPU fixture selection: '+str(len(selection['tasks']))+' tasks, '+str(len(selection['paths']))+' test files.', flush=True)
    print('This records fixture execution/collection, not native/GPU/model certification.', flush=True)
    started = datetime.now(timezone.utc).isoformat()
    pytest_exit = None
    error = None
    if args.list:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
    else:
        try:
            pytest_exit = subprocess.run(command, cwd=root, env=cpu_environment(os.environ)).returncode
        except KeyboardInterrupt:
            pytest_exit, error = 130, 'interrupted'
        except OSError as exc:
            pytest_exit, error = 126, str(exc)
    try:
        after = source_snapshot(root)
        after_git = git_state(root)
        after_selection = build_selection(root, args.task)
        changed = (before['sha256'] != after['sha256'] or before_git.get('head') != after_git.get('head')
                   or selection['paths'] != after_selection['paths'])
    except (OSError, ValueError) as exc:
        after, after_git, changed = {'sha256': None}, {'head': None, 'dirty': None}, True
        error = str(exc)
    exit_code = 0 if pytest_exit is None else pytest_exit if pytest_exit >= 0 else 128-pytest_exit
    if changed and exit_code == 0:
        exit_code = 2
    clean_head = (not changed and before_git.get('head') is not None
                  and before_git.get('dirty') is False and after_git.get('dirty') is False)
    report = dict(schema='mind2act-final-scoring-cpu-receipt', scoring_version=version,
                  mode=mode, tasks=selection['tasks'], full_30_task_selection=set(selection['tasks']) == set(TASK_IDS),
                  task_test_manifest=selection['by_task'], shared_tests=selection['shared'],
                  selected_test_paths=selection['paths'],
                  selected_test_hashes={p: before['hashes'][p] for p in selection['paths']},
                  source_hashes=before['hashes'], source_snapshot_sha256=before['sha256'],
                  source_snapshot_after_sha256=after['sha256'],
                  hash_scope='src/tests/tools/scripts/manifests code/config, test helpers, frozen public coding baseline, pytest config and final authority',
                  code_sha=before_git.get('head'), git_dirty=before_git.get('dirty'),
                  git_after=after_git, worktree_changed_during_run=changed, fixed_head_verified=bool(clean_head),
                  tested_ref=('head:'+before_git['head']) if clean_head else 'worktree:'+before['sha256'],
                  command=command, cwd=str(root), started_at=started,
                  finished_at=datetime.now(timezone.utc).isoformat(), pytest_exit_code=pytest_exit,
                  exit_code=exit_code, error=error,
                  scope='CPU fixture execution/collection only; skips and native/GPU/model acceptance require separate review')
    if args.receipt:
        receipt = args.receipt.expanduser().resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=receipt.parent,
                                         prefix=receipt.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(receipt)
        print('Receipt: '+str(receipt), flush=True)
    if changed:
        print('Worktree source changed during this run; fixed-source acceptance is invalid.', file=sys.stderr)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
