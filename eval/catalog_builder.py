"""Freeze already compiled, real test specs into a digest-bound fixed50 catalog.

This producer does not rename replay instances into test cases or generate task
parameters. Task-specific compilers must provide the actual source-bound specs.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .contracts import to_json
from .agent_names import agent_field
from .instance_catalog import (SCHEMA, INSTANCE_SET, asset_descriptor,
                               load_catalog, relative_path, sha256)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def repair_cp03_asset_hashes(spec, asset_root):
    """Bind CP03 closure keys to its current relative station/texture paths."""
    if spec.task_id != 'CP03':
        return spec
    config = to_json(spec.config)
    config['asset_sha256'] = {spec.assets[key]: sha256(relative_path(asset_root, spec.assets[key]))
                              for key in ('station', 'texture_map')}
    return replace(spec, config=config)


def freeze_catalog(asset_root, records, *, output='catalog.json'):
    """Write and fully validate a catalog in a new producer-owned release tree.

    Each record must reference a spec already compiled from a frozen test source,
    carrying its case_id, source_configuration_sha256 and subset membership.
    Existing specs and RELEASE.json are never overwritten by this function.
    """
    from .cli import read_spec
    from .spec_identity import spec_digest
    root = Path(asset_root).resolve()
    target = relative_path(root, output)
    if target.exists():
        raise FileExistsError('refusing to overwrite an existing catalog')
    assets_cache, closure_cache, rows = {}, {}, []
    for record in records:
        source_sha = record['source_configuration_sha256']
        from .instance_catalog import digest_field
        digest_field(source_sha)
        path = relative_path(root, record['spec_path'])
        spec = read_spec(path)
        if spec.instance_id != record['case_id'] or spec.variation.mode != 'clean':
            raise ValueError('compiled test spec must retain the actual clean case identity')
        assets = {}
        for key, relative in spec.assets.items():
            if relative not in assets_cache:
                asset = relative_path(root, relative)
                if asset.is_dir():
                    files = [dict(path=str(p.relative_to(asset)), sha256=sha256(p))
                             for p in sorted(asset.rglob('*')) if p.is_file()]
                    content = dict(schema='mind2act_eval.asset_directory.v1', files=files)
                    identity = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
                    manifest = root / 'catalog-assets' / (identity + '.json')
                    if not manifest.exists():
                        write_json(manifest, content)
                    descriptor = dict(path=relative, manifest=dict(
                        path=str(manifest.relative_to(root)), sha256=sha256(manifest)))
                else:
                    descriptor = dict(path=relative, sha256=sha256(asset))
                asset_descriptor(root, descriptor, verify=True)
                assets_cache[relative] = descriptor
            assets[key] = assets_cache[relative]
        key = tuple(sorted(spec.assets.items()))
        if key not in closure_cache:
            from .asset_catalog_closure import build_asset_closure
            closure = build_asset_closure(root, spec.assets)
            identity = hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest()
            closure_file = root / 'catalog-assets' / ('closure-' + identity + '.json')
            if not closure_file.exists():
                write_json(closure_file, closure)
            closure_cache[key] = dict(path=str(closure_file.relative_to(root)), sha256=sha256(closure_file))
        metadata = record.get('replay_metadata')
        row = dict(case_id=spec.instance_id, instance_id=spec.instance_id,
                   task_id=spec.task_id, difficulty=spec.difficulty,
                   spec=dict(path=str(path.relative_to(root)), sha256=sha256(path),
                             semantic_sha256=spec_digest(spec)), assets=assets, asset_closure=closure_cache[key],
                   agent_selected=agent_field(record, 'agent_selected'),
                   coding_agent_selected=agent_field(record, 'agent_selected'),
                   replay_metadata=metadata, source_configuration_sha256=source_sha,
                   logical_slot=record['logical_slot'], native_case_id=record['native_case_id'],
                   limits=record['limits'])
        for field in ('limits_e2e', 'budget_version_e2e'):
            if field in record:
                row[field] = record[field]
        rows.append(row)
    tasks = sorted({row['task_id'] for row in rows})
    document = dict(schema=SCHEMA, instance_set=INSTANCE_SET, tasks=tasks,
                    cases_per_cell=50, agent_per_cell=5, coding_agent_per_cell=5, instances=rows)
    # Check the complete grid before publishing the catalog's final filename.
    temporary = target.with_name(target.name + '.pending')
    if temporary.exists():
        raise FileExistsError('catalog staging file already exists')
    write_json(temporary, document)
    descriptor = dict(path=str(temporary.relative_to(root)), sha256=sha256(temporary))
    try:
        load_catalog(root, descriptor, tasks=tasks)
    except BaseException:
        temporary.unlink()
        raise
    temporary.rename(target)
    return dict(path=output, sha256=sha256(target))
