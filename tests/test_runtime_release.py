"""Real CPU entrypoints reject retired inputs before native/provider imports."""
from argparse import Namespace
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from mind2act_eval import cli
from mind2act_eval.contracts import InstanceSpec, VariationSpec, to_json
from mind2act_eval.release_contract import ReleaseContractError
from mind2act_eval.scene_probe import SOURCE_CLEAN_TASKS, spec_digest
from mind2act_eval.runtime_release import NEW15_SOURCE_CLEAN_TASKS


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('entrypoint', [cli._job, cli._worker])
def test_actual_entrypoints_reject_deprecated_input_before_launch(tmp_path, entrypoint, monkeypatch):
    if entrypoint is cli._worker:
        def late_gpu_check(args):
            raise RuntimeError("worker reached GPU selection before deprecated-input guard")
        monkeypatch.setattr(cli, "_gpu_selection_kwargs", late_gpu_check)
    spec = InstanceSpec('demo', 'demo', 'easy', 'demo', 'v1', 'v1', 'v1')
    write(tmp_path / 'spec.json', to_json(spec))
    (tmp_path / 'DEPRECATED').touch()
    values = cli._job_defaults()
    values.update(spec=str(tmp_path / 'spec.json'), asset_root=str(tmp_path), backend='demo', wall_s=10, physics_ticks=600)
    with pytest.raises(ReleaseContractError, match='DEPRECATED'):
        entrypoint(Namespace(**values))




@pytest.fixture(params=['old15', 'new15'])
def release(tmp_path, request):
    old = request.param == 'old15'
    root = tmp_path / (request.param + '-release-v1')
    root.mkdir()
    task = 'PR1' if old else 'E2'
    spec = InstanceSpec(task, 'source-easy', 'easy', 'actual-source', 'v1', 'v1', 'v1', seed=17)
    path = root / 'spec.json'
    digest = write(path, to_json(spec))
    metadata = root / 'metadata.json'
    metadata_sha = write(metadata, {'original': 'retained'})
    rows = [dict(task_id=task, difficulty=tier, instance_id=f'{task}-{tier}')
            for task in sorted(SOURCE_CLEAN_TASKS if old else NEW15_SOURCE_CLEAN_TASKS)
            for tier in ('easy', 'medium', 'hard')]
    row = next(r for r in rows if r['task_id'] == task and r['difficulty'] == 'easy')
    row.update(instance_id=spec.instance_id,
               spec=dict(path='spec.json', sha256=digest, semantic_sha256=spec_digest(spec)),
               replay_metadata=dict(path='metadata.json', sha256=metadata_sha))
    write(root / 'asset-manifest.json', {})
    data = dict(asset_name=root.name, manifest='asset-manifest.json', manifest_sha256='a' * 64,
                generated_at='2026-11-02T07:00:00Z', code_commit='d' * 40, instances=rows)
    write(root / 'RELEASE.json', data)
    values = cli._job_defaults()
    values.update(spec=str(path), asset_root=str(root), backend='isaac')
    return spec, Namespace(**values), data


def test_formal_cpu_and_verified_code_binding(release):
    from mind2act_eval.runtime_release import validate_release_inputs
    spec, args, _ = release
    assert not validate_release_inputs(spec, args).code_identity_checked
    with pytest.raises(ReleaseContractError, match='actual SOURCE_COMMIT'):
        validate_release_inputs(spec, args, require_code_identity=True)
    assert validate_release_inputs(spec, args, require_code_identity=True,
                                   verified_code_commit='d' * 40).code_identity_checked
    with pytest.raises(ReleaseContractError, match='verified code'):
        validate_release_inputs(spec, args, verified_code_commit='e' * 40)


def test_native_entrypoint_requires_actual_identity_before_sdk(release):
    _, args, _ = release
    with pytest.raises(ReleaseContractError, match='actual SOURCE_COMMIT'):
        cli._worker(args)


@pytest.mark.parametrize('field,value', [('seed', 99), ('source_id', 'wrong-source'),
                                        ('config', {'robot': 'changed'})])
def test_release_rejects_semantic_substitution(release, field, value):
    from mind2act_eval.runtime_release import validate_release_inputs
    spec, args, _ = release
    with pytest.raises(ReleaseContractError, match='source spec'):
        validate_release_inputs(replace(spec, **{field: value}), args)


def test_random_binds_same_clean_source_and_preserves_metadata(release):
    from mind2act_eval.runtime_release import validate_release_inputs
    from mind2act_eval.variations import base_spec_digest
    spec, args, _ = release
    args.replay_metadata = str(Path(args.asset_root) / 'metadata.json')
    before = Path(args.replay_metadata).read_bytes()
    variation = VariationSpec('random', 'fixed-layout', 'b' * 64, base_spec_digest(spec), 'c' * 64)
    random = replace(spec, variation=variation)
    validate_release_inputs(random, args)
    assert Path(args.replay_metadata).read_bytes() == before
    with pytest.raises(ReleaseContractError, match='random base digest'):
        validate_release_inputs(replace(random, variation=replace(variation, base_spec_sha256='f' * 64)), args)
    Path(args.replay_metadata).write_bytes(before + b' ')
    with pytest.raises(ReleaseContractError, match='metadata differs'):
        validate_release_inputs(random, args)


def test_retired_asset_alias_is_rejected_even_with_live_release(release, tmp_path):
    from mind2act_eval.runtime_release import validate_release_inputs
    spec, args, _ = release
    old = tmp_path / 'old-assets'
    old.mkdir()
    (old / 'DEPRECATED').touch()
    (old / 'alias').symlink_to(args.asset_root, target_is_directory=True)
    spec = replace(spec, assets={'station': str(old / 'alias' / 'spec.json')})
    with pytest.raises(ReleaseContractError, match='DEPRECATED'):
        validate_release_inputs(spec, args)


def test_actual_batch_entry_admits_each_exact_release(release):
    spec, args, _ = release
    values = vars(args).copy()
    values.update(backend='demo', wall_s=10, physics_ticks=600)
    jobs = cli._batch_jobs([values], cli._job_defaults())
    assert len(jobs) == 1 and jobs[0].spec == spec


def test_existing_replay_rows_keep_explicit_diagnostic_budget(release):
    spec,args,data = release
    row = next(r for r in data['instances'] if r['instance_id'] == spec.instance_id)
    row['limits'] = dict(physics_ticks=60000,wall_s=7200,retries=0)
    write(Path(args.asset_root)/'RELEASE.json',data)
    args.physics_ticks,args.wall_s,args.retries = 600,60,0
    args.adapter,args.gpu = 'mind2act_eval.adapters.hold:create',0
    assert cli._job(args).spec == spec


def test_release_rejects_a_valid_45_cell_grid_from_the_other_release(release):
    from mind2act_eval.runtime_release import validate_release_inputs
    spec, args, data = release
    source = 'PR1' if spec.task_id == 'PR1' else 'E2'
    replacement = 'E2' if source == 'PR1' else 'PR1'
    for row in data['instances']:
        if row['task_id'] == source:
            row['task_id'] = replacement
    write(Path(args.asset_root) / 'RELEASE.json', data)
    with pytest.raises(ReleaseContractError, match='exact task grid'):
        validate_release_inputs(spec, args)


@pytest.mark.parametrize('alias', [False, True])
def test_named_formal_release_cannot_fall_back_to_unmarked_legacy_inputs(release, tmp_path, alias):
    from mind2act_eval.runtime_release import validate_release_inputs
    spec, args, _ = release
    (Path(args.asset_root) / 'RELEASE.json').unlink()
    if alias:
        link = tmp_path / 'release-alias'
        link.symlink_to(args.asset_root, target_is_directory=True)
        args.asset_root = str(link)
    with pytest.raises(ReleaseContractError, match='cannot read release descriptor'):
        validate_release_inputs(spec, args)
