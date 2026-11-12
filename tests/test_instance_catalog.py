"""Test instances outside the replay45 grid reach standard CPU admission."""
from argparse import Namespace
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from mind2act_eval import cli
from mind2act_eval.contracts import to_json
from mind2act_eval.instance_catalog import SCHEMA, sha256, asset_descriptor
from mind2act_eval.release_contract import ReleaseContractError, load_release
from mind2act_eval.runtime_release import validate_release_inputs
from mind2act_eval.scene_probe import spec_digest
from test_runtime_release import release, write


@pytest.fixture
def catalog_release(release):
    source, args, descriptor = release
    root = Path(args.asset_root)
    (root / 'payload.bin').write_bytes(b'actual task asset')
    asset = dict(path='payload.bin', sha256=sha256(root / 'payload.bin'))
    closure_sha = write(root / 'closure.json', dict(schema='mind2act_eval.asset_closure.v1',
        root_assets={'payload': 'payload.bin'}, files=[asset], unresolved=[]))
    closure_ref = dict(path='closure.json', sha256=closure_sha)
    rows = []
    test_spec = replace(source, instance_id=('PR1_easy_0c7134bad55b7c6c' if source.task_id == 'PR1' else 'E2-easy-test-fixture'),
                        assets={'payload': 'payload.bin'}, config={'target': 17})
    test_digest = write(root / 'test.json', to_json(test_spec))
    for original in descriptor['instances']:
        for slot in range(50):
            case = f'{original["task_id"]}-{original["difficulty"]}-test-{slot:03}'
            rows.append(dict(case_id=case, task_id=original['task_id'], difficulty=original['difficulty'],
                coding_agent_selected=slot < 5,
                spec=dict(path='test.json', sha256=test_digest, semantic_sha256=spec_digest(test_spec)),
                assets={'payload': asset}, replay_metadata=None, asset_closure=closure_ref))
    selected = next(r for r in rows if r['task_id'] == source.task_id and r['difficulty'] == 'easy')
    selected['case_id'] = test_spec.instance_id
    catalog = dict(schema=SCHEMA, instance_set='test50-r5',
                   tasks=sorted({r['task_id'] for r in rows}),
                   cases_per_cell=50, coding_agent_per_cell=5, instances=rows)
    descriptor['catalog'] = dict(path='catalog.json', sha256=write(root / 'catalog.json', catalog))
    write(root / 'RELEASE.json', descriptor)
    return test_spec, args, descriptor, catalog


def save_catalog(args, descriptor, catalog):
    root = Path(args.asset_root)
    descriptor['catalog']['sha256'] = write(root / 'catalog.json', catalog)
    write(root / 'RELEASE.json', descriptor)


def test_test50_and_coding_subsets_are_distinct_from_replays(catalog_release):
    spec, args, _, _ = catalog_release
    value = load_release(args.asset_root)
    assert len(value.instances) == 45
    assert len(value.select_cases('test50-r5')) == 2250
    assert len(value.select_cases('coding-agent')) == 225
    row = value.resolve_case(spec.instance_id)
    assert row['replay_metadata'] is None
    assert row not in value.instances
    assert value.select_cases('replay')


def test_standard_model_admission_accepts_actual_catalog_case_without_h5(catalog_release):
    spec, args, _, _ = catalog_release
    value = validate_release_inputs(spec, args)
    assert value.resolve_case(spec.instance_id)['spec']['path'] == 'test.json'
    args.spec = None
    args.case_id = spec.instance_id
    args.adapter = 'mind2act_eval.adapters.hold:create'
    args.gpu, args.wall_s, args.physics_ticks = 0, 60, 600
    job = cli._job(args)
    assert job.spec == spec
    assert '--h5' not in job.argv
    assert str(Path(args.asset_root) / 'test.json') in job.argv


def test_batch_resolves_case_id_through_same_admission(catalog_release):
    spec, args, _, _ = catalog_release
    entry = dict(case_id=spec.instance_id, asset_root=args.asset_root, backend='isaac',
                 adapter='mind2act_eval.adapters.hold:create', gpu=0, wall_s=60, physics_ticks=600)
    assert cli._batch_jobs([entry], cli._job_defaults())[0].spec == spec


@pytest.mark.parametrize('entrypoint', ['single', 'batch', 'validate'])
@pytest.mark.parametrize('substitution', ['same_cell', 'other_task_tier'])
def test_requested_case_cannot_load_another_catalog_rows_spec(catalog_release, entrypoint, substitution):
    spec, args, _, catalog = catalog_release
    if substitution == 'same_cell':
        requested = next(r for r in catalog['instances'] if r['task_id'] == spec.task_id
                         and r['difficulty'] == spec.difficulty and r['case_id'] != spec.instance_id)
    else:
        requested = next(r for r in catalog['instances'] if r['task_id'] != spec.task_id
                         and r['difficulty'] == 'hard')
    # Both rows point to the real, hash-valid test.json in this fixture. Its
    # spec can independently pass admission through its own catalog row, but
    # it must never satisfy a request for a different case or task/tier.
    args.case_id, args.spec = requested['case_id'], None
    args.adapter = 'mind2act_eval.adapters.hold:create'
    args.gpu, args.wall_s, args.physics_ticks = 0, 60, 600
    with pytest.raises(ReleaseContractError, match='requested case_id'):
        if entrypoint == 'single':
            cli._job(args)
        elif entrypoint == 'batch':
            cli._batch_jobs([vars(args).copy()], cli._job_defaults())
        else:
            validate_release_inputs(spec, args)


def test_unknown_case_is_not_replaced_with_replay45(catalog_release):
    spec, args, _, _ = catalog_release
    args.case_id = 'missing-test-case'
    args.spec = None
    with pytest.raises(ReleaseContractError, match='unknown or ambiguous'):
        cli._job(args)
    with pytest.raises(ReleaseContractError, match='unknown or ambiguous'):
        validate_release_inputs(spec, args)
    args.case_id = None
    with pytest.raises(ReleaseContractError, match='absent'):
        validate_release_inputs(replace(spec, instance_id='not-in-catalog'), args)


def test_test_case_cannot_replay_an_unrelated_expert_file(catalog_release):
    spec, args, _, _ = catalog_release
    args.case_id, args.spec, args.h5 = spec.instance_id, None, 'unrelated-expert.h5'
    with pytest.raises(ReleaseContractError, match='requires replay_metadata'):
        cli._job(args)


def test_catalog_and_spec_and_asset_tampering_are_rejected(catalog_release):
    spec, args, _, _ = catalog_release
    root = Path(args.asset_root)
    with pytest.raises(ReleaseContractError, match='source spec'):
        validate_release_inputs(replace(spec, config={'target': 999}), args)
    (root / 'payload.bin').write_bytes(b'modified asset')
    with pytest.raises(ReleaseContractError, match='SHA256 differs'):
        validate_release_inputs(spec, args)
    (root / 'catalog.json').write_text('{}')
    with pytest.raises(ReleaseContractError, match='SHA256 differs'):
        load_release(root)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'subset', 'path_escape'])
def test_malformed_catalog_rejected_before_sdk(catalog_release, mutation):
    _, args, descriptor, catalog = catalog_release
    if mutation == 'missing':
        catalog['instances'].pop()
    elif mutation == 'duplicate':
        catalog['instances'][1]['case_id'] = catalog['instances'][0]['case_id']
    elif mutation == 'subset':
        catalog['instances'][0]['coding_agent_selected'] = False
    else:
        catalog['instances'][0]['spec']['path'] = '../outside.json'
    save_catalog(args, descriptor, catalog)
    with pytest.raises(ReleaseContractError):
        load_release(args.asset_root)


def test_directory_asset_requires_exact_file_set_and_hashes(tmp_path):
    root = tmp_path
    (root / 'cards').mkdir()
    (root / 'cards/a.png').write_bytes(b'card')
    descriptor = dict(path='cards', files=[dict(path='a.png', sha256=sha256(root / 'cards/a.png'))])
    asset_descriptor(root, descriptor, verify=True)
    (root / 'cards/unlisted.png').write_bytes(b'wrong')
    with pytest.raises(ReleaseContractError, match='file set differs'):
        asset_descriptor(root, descriptor, verify=True)


def test_directory_asset_can_share_a_pinned_manifest(tmp_path):
    (tmp_path / 'sdk').mkdir()
    (tmp_path / 'sdk/material.mdl').write_bytes(b'MDL')
    manifest = dict(schema='mind2act_eval.asset_directory.v1', files=[
        dict(path='material.mdl', sha256=sha256(tmp_path / 'sdk/material.mdl'))])
    digest = write(tmp_path / 'directory.json', manifest)
    descriptor = dict(path='sdk', manifest=dict(path='directory.json', sha256=digest))
    asset_descriptor(tmp_path, descriptor, verify=True)
    (tmp_path / 'sdk/material.mdl').write_bytes(b'changed')
    with pytest.raises(ReleaseContractError, match='SHA256 differs'):
        asset_descriptor(tmp_path, descriptor, verify=True)


def test_recursive_usd_layer_and_texture_are_bound_before_native_imports(tmp_path):
    from pxr import Sdf, Usd
    from mind2act_eval.asset_catalog_closure import build_asset_closure
    from mind2act_eval.instance_catalog import verify_case_assets
    from mind2act_eval.contracts import InstanceSpec
    texture = tmp_path / 'texture.png'
    texture.write_bytes(b'original texture')
    child = Usd.Stage.CreateNew(str(tmp_path / 'child.usda'))
    child.DefinePrim('/World', 'Xform').CreateAttribute('texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('texture.png'))
    child.GetRootLayer().Save()
    parent = Sdf.Layer.CreateNew(str(tmp_path / 'scene.usda'))
    parent.subLayerPaths = ['child.usda']
    parent.Save()
    assets = {'station': 'scene.usda'}
    closure = build_asset_closure(tmp_path, assets)
    assert {r['path'] for r in closure['files']} == {'scene.usda', 'child.usda', 'texture.png'}
    closure_sha = write(tmp_path / 'closure.json', closure)
    row = dict(assets={'station': dict(path='scene.usda', sha256=sha256(tmp_path / 'scene.usda'))},
               asset_closure=dict(path='closure.json', sha256=closure_sha))
    spec = InstanceSpec('PR1', 'real-test', 'easy', 'source', 'v1', 'v1', 'v1', assets=assets)
    verify_case_assets(tmp_path, row, spec)
    texture.write_bytes(b'modified indirect texture')
    with pytest.raises(ReleaseContractError, match='SHA256 differs'):
        verify_case_assets(tmp_path, row, spec)


def test_usd_closure_rejects_an_indirect_dependency_outside_release(tmp_path):
    from pxr import Sdf
    from mind2act_eval.asset_catalog_closure import build_asset_closure
    outside = Sdf.Layer.CreateNew(str(tmp_path / 'outside.usda'))
    outside.Save()
    root = tmp_path / 'release'
    root.mkdir()
    layer = Sdf.Layer.CreateNew(str(root / 'scene.usda'))
    layer.subLayerPaths = ['../outside.usda']
    layer.Save()
    with pytest.raises(ReleaseContractError, match='escapes'):
        build_asset_closure(root, {'station': 'scene.usda'})


def test_cpu_hash_does_not_accept_changed_bytes_with_unchanged_stat(tmp_path, monkeypatch):
    path = tmp_path / 'asset.bin'
    path.write_bytes(b'first')
    stat = path.stat()
    original_stat = Path.stat
    monkeypatch.setattr(Path, 'stat', lambda self, **kw: stat if self == path else original_stat(self, **kw))
    first = sha256(path)
    assert sha256(path) == first
    path.write_bytes(b'other')
    assert sha256(path) != first


def test_real_pr1_test_source_keeps_identity_and_parameters():
    from mind2act_eval.tasks.PR1.instances import spec_from_row
    source = json.loads((Path(__file__).parent / 'fixtures/test50_pr1_easy.json').read_text())
    spec = spec_from_row(dict(task='PR1', difficulty=source['difficulty'], source_case=source,
                             source_protocol=dict(fixture_y_offset_m=0.)), asset_version='test50-r5')
    assert spec.instance_id == 'PR1_easy_0c7134bad55b7c6c'
    assert spec.config['initial_ohm'] == source['parameters']['initial']
    assert spec.config['target_ohm'] == source['parameters']['target']


