"""Optional offline audits for historical source projections.

Callers continue to supply their original source digest. An approval binds a
particular portable file and its complete closure; caller config cannot add an
approval. JSON readers recover the exact original bytes before existing source
validation. USD projection readers recover only the original typed asset tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from .source_assets import source_file, asset_bytes, asset_layer

_hashes = {}


@lru_cache(maxsize=1)
def _catalogue():
    raw = Path(__file__).with_name('source_asset_approvals.json').read_bytes()
    value = json.loads(raw)
    # Immutable training extension generated from reviewed per-instance proofs.
    training_raw = Path(__file__).with_name('training_source_asset_approvals.json').read_bytes()
    training = json.loads(training_raw)
    if training.get('schema') != 'mind2act_eval.source_asset_approvals.v1':
        raise ValueError('unsupported training source asset approval catalogue')
    for key in ('approvals', 'deployments', 'public_recipe_rebindings'):
        value.setdefault(key, []).extend(training.get(key, []))
    for key, closure in training.get('closures', {}).items():
        if key in value.setdefault('closures', {}) and value['closures'][key] != closure:
            raise ValueError('training source closure conflicts with base catalogue')
        value['closures'][key] = closure
    # Independently approved clean validation identities; never train/test.
    raw = Path(__file__).with_name('validation_source_asset_approvals.json').read_bytes()
    validation = json.loads(raw)
    if validation.get('schema') != 'mind2act_eval.source_asset_approvals.v1' or len(validation['approvals']) != 190:
        raise ValueError('public validation inventory differs')
    if len(training['approvals']) != 1380 or len(value['approvals']) != 1380:
        raise ValueError('public training inventory differs')
    value['approvals'].extend(validation['approvals'])
    for key, closure in validation['closures'].items():
        if key in value['closures'] and value['closures'][key] != closure:
            raise ValueError('public validation closure conflicts')
        value['closures'][key] = closure
    if value.get('schema') != 'mind2act_eval.source_asset_approvals.v1':
        raise ValueError('unsupported source asset approval catalogue')
    if any(not isinstance(r.get('identity'), dict) for r in value['approvals']):
        raise ValueError('source asset approvals require their frozen source identity')
    if any('json' in r or 'usd' in r for r in value['approvals']):
        raise ValueError('source URI restoration tables belong in hash-bound external proofs')
    approved = [row['identity'] for row in value['approvals'] + value.get('deployments', [])]
    seen = set()
    for row in value.get('public_recipe_rebindings', []):
        identity = row['identity']
        previous = dict(identity, config_sha256=row['approved_config_sha256'])
        key = json.dumps(identity, sort_keys=True)
        if (row['changed_fields'] != ['goal_recipe'] or previous not in approved
                or identity == previous or key in seen
                or any(re.fullmatch('[0-9a-f]{64}', item) is None
                       for item in (identity['config_sha256'], row['approved_config_sha256']))):
            raise ValueError('public recipe rebind requires one exact reviewed configuration and prior approval')
        seen.add(key)
    return value


def _approvals():
    return _catalogue()['approvals']


def _closure(row):
    """Keep private closure paths in a code-pinned external proof, when needed."""
    if 'closure' in row:
        return row['closure']
    closure_id = row['closure_id']
    reference = _catalogue()['closures'][closure_id]
    if isinstance(reference, list):
        return reference
    if (not isinstance(reference, dict) or set(reference) != {'schema', 'proof_sha256'}
            or reference['schema'] != 'mind2act_eval.source_asset_closure_reference.v1'):
        raise ValueError('invalid code-owned source closure reference')
    proof = _private_proof(reference['proof_sha256'])
    if (not isinstance(proof, dict) or set(proof) != {'schema', 'closure_id', 'members'}
            or proof['schema'] != 'mind2act_eval.source_asset_closure.v1'
            or proof['closure_id'] != closure_id
            or not isinstance(proof['members'], list) or not proof['members']):
        raise ValueError('source closure proof identity differs')
    members = proof['members']
    paths = set()
    for ref in members:
        if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
                or not isinstance(ref['path'], str) or not ref['path']
                or not isinstance(ref['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', ref['sha256']) is None
                or ref['path'] in paths):
            raise ValueError('invalid source closure proof member')
        paths.add(ref['path'])
    if hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest() != closure_id:
        raise ValueError('source closure proof members differ from their code-owned identity')
    return members


def _private_proof(digest):
    root = os.environ.get('MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT')
    if not root or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('explicit source projection proof root and code-owned proof SHA are required')
    path = _contained(Path(root).resolve(strict=True), digest + '.json')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('source projection restoration proof SHA differs')
    return json.loads(raw)


def _restoration(row):
    proof = _private_proof(row.get('proof_sha256'))
    if (proof.get('schema') != 'mind2act_eval.source_projection_restoration.v1'
            or proof.get('source_sha256') != row['source_sha256']
            or proof.get('runtime_sha256') != row['runtime_sha256']
            or proof.get('identity') != row.get('identity')):
        raise ValueError('source projection restoration proof identity differs')
    return proof


def authorize_station_deployment(spec, assets, manifest_asset_id):
    """Authorize the actual opening source, selected manifest and every input.

    The manifest is an input, never an authority. Only an immutable catalogue
    entry can select its hash-bound external closure proof.
    """
    identity = _identity(spec)
    source = assets.resolve('station')
    manifest = assets.resolve(manifest_asset_id)
    source_sha = _digest(source)
    manifest_sha = _digest(manifest)
    matches = [r for r in _catalogue().get('deployments', [])
               if r['identity'] == identity
               and r['source_station_sha256'] == source_sha
               and r['manifest_sha256'] == manifest_sha
               and r['manifest_asset_id'] == manifest_asset_id]
    if len(matches) != 1:
        raise ValueError('station material deployment has no code-owned identity/source/manifest approval')
    row = matches[0]
    proof = _private_proof(row['proof_sha256'])
    if (proof.get('schema') != 'mind2act_eval.station_deployment_approval.v1'
            or any(proof.get(key) != row[key] for key in (
                'identity', 'source_station_sha256', 'manifest_sha256', 'manifest_asset_id'))):
        raise ValueError('station material deployment proof differs from its code-owned approval')
    raw = json.loads(manifest.read_bytes())
    required = {'station', manifest_asset_id} | {r['target_asset_id'] for r in raw['mappings']}
    source_manifest = spec.config.get('station_relocation_manifest')
    if source_manifest:
        required.add(source_manifest)
    roles = {r['asset_id']: r for r in proof['assets']}
    if not required <= set(roles):
        raise ValueError('approved station deployment omits an actual opening dependency')
    for key, ref in roles.items():
        actual = assets.resolve(key)
        expected = (assets.root / ref['path']).resolve(strict=True)
        if not expected.is_relative_to(assets.root) or actual != expected:
            raise ValueError('approved station deployment asset role/path changed: ' + key)
        if ref.get('kind') == 'directory':
            if not actual.is_dir():
                raise ValueError('approved station deployment search directory changed: ' + key)
        elif _digest(actual) != ref['sha256']:
            raise ValueError('approved station deployment target bytes changed: ' + key)
    for ref in proof['closure']:
        if _digest(_contained(assets.root, ref['path'])) != ref['sha256']:
            raise ValueError('approved station deployment resource closure changed: ' + ref['path'])
    return dict(identity=identity, source_station_actual_sha256=source_sha,
                material_manifest_actual_sha256=manifest_sha,
                code_owned_proof_sha256=row['proof_sha256'],
                verified_asset_roles=len(roles), verified_closure_files=len(proof['closure']))


def _digest(path):
    st = path.stat()
    identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if identity in _hashes:
        return _hashes[identity]
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    result = h.hexdigest()
    _hashes[identity] = result
    return result


def _contained(root, relative):
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('approval closure requires a contained relative path')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('approval closure escapes its asset root')
    return path


def _identity(spec):
    if spec is None:
        return None
    from mind2act_eval.contracts import to_json
    value = {key: getattr(spec, key) for key in (
        'task_id', 'instance_id', 'difficulty', 'source_id', 'seed',
        'config_version', 'asset_version', 'protocol_version')}
    value['clock'] = to_json(spec.clock)
    raw = json.dumps(to_json(spec.config), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    value['config_sha256'] = hashlib.sha256(raw).hexdigest()
    from mind2act_eval.english_specs import approved_source_config_sha256
    projected_source = approved_source_config_sha256(spec)
    if projected_source is not None:
        value['config_sha256'] = projected_source
    # Only complete, code-owned identity matches may reuse the prior resource
    # approval. A different recipe, private config, instance or seed still fails.
    for row in _catalogue().get('public_recipe_rebindings', []):
        if value == row['identity']:
            value['config_sha256'] = row['approved_config_sha256']
            break
    return value


def _projection(path, expected, actual, assets):
    matches = []
    for row in _approvals():
        if row['runtime_sha256'] != actual or expected is not None and row['source_sha256'] != expected:
            continue
        if row.get('identity') is not None and row['identity'] != _identity(getattr(assets, 'spec', None)):
            continue
        roots = [p for p in path.parents if p.name == row['asset_name']]
        for root in roots:
            # A caller may itself store the package below another 'assets'
            # directory. Only its exact approved relative path is a candidate.
            if path.relative_to(root) != Path(row['runtime_path']):
                continue
            if path == _contained(root, row['runtime_path']):
                matches.append((row, root))
    if len(matches) != 1:
        raise ValueError('resource differs from its pinned source without one code-approved projection')
    row, root = matches[0]
    for ref in _closure(row):
        if _digest(_contained(root, ref['path'])) != ref['sha256']:
            raise ValueError('code-approved source projection closure changed: ' + ref['path'])
    return row


def _remember(assets, path, row, actual):
    if not hasattr(assets, '__dict__'):
        return
    if not hasattr(assets, '_source_projection_records'):
        assets._source_projection_records = {}
    assets._source_projection_records[str(path)] = dict(
        runtime_path=str(path), runtime_actual_sha256=actual,
        declared_source_sha256=row['source_sha256'],
        approval_runtime_sha256=row['runtime_sha256'],
        closure_id=row.get('closure_id'),
        projection_kind=row.get('restoration_kind', 'usd-relative-export'),
        restoration_proof_sha256=row.get('proof_sha256'))


def write_source_projection_receipt(assets, output):
    rows = getattr(assets, '_source_projection_records', {})
    if not rows:
        return
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'source-asset-projections.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(
        schema='mind2act_eval.source_asset_projections.v1',
        code_approval_catalogue_sha256=_digest(Path(__file__).with_name('source_asset_approvals.json')),
        resources=list(rows.values())), indent=2) + '\n')
    temporary.replace(path)








def verify_source_file(assets, asset_id, expected_sha256, *, error=None):
    try:
        return _verify_source_file(assets, asset_id, expected_sha256)
    except ValueError as exc:
        if error is not None:
            raise ValueError(error + ': ' + str(exc)) from exc
        raise


def _verify_source_file(assets, asset_id, expected_sha256):
    """Verify original bytes or one fully pinned portable projection."""
    path = assets.resolve(asset_id)
    actual = _digest(path)
    if actual != expected_sha256:
        row = _projection(path, expected_sha256, actual, assets)
        _remember(assets, path, row, actual)
    return path


def read_source_bytes(assets, asset_id, expected_sha256=None, *, error=None):
    try:
        return _read_source_bytes(assets, asset_id, expected_sha256)
    except ValueError as exc:
        if error is not None:
            raise ValueError(error + ': ' + str(exc)) from exc
        raise


def _read_source_bytes(assets, asset_id, expected_sha256=None):
    """Offline: recover historical source JSON from reviewed path edits."""
    path = assets.resolve(asset_id)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual == expected_sha256:
        return raw
    candidates = [r for r in _approvals() if r['runtime_sha256'] == actual
                  and (expected_sha256 is None or r['source_sha256'] == expected_sha256)]
    if not candidates:
        if expected_sha256 is None:
            return raw
        raise ValueError('JSON differs from its pinned source without a code-approved projection')
    row = _projection(path, expected_sha256, actual, assets)
    _remember(assets, path, row, actual)
    proof = _restoration(row)
    value = json.loads(raw)
    for edit in proof['json']['undo']:
        keys = [s.replace('~1', '/').replace('~0', '~') for s in edit['pointer'].split('/')[1:]]
        parent = value
        for key in keys[:-1]:
            parent = parent[int(key)] if isinstance(parent, list) else parent[key]
        key = int(keys[-1]) if isinstance(parent, list) else keys[-1]
        if parent[key] != edit['runtime']:
            raise ValueError('approved JSON path edit differs at ' + edit['pointer'])
        parent[key] = edit['source']
    settings = dict(proof['json']['serialization'])
    suffix = settings.pop('trailing_newline')
    if settings.get('separators') is not None:
        settings['separators'] = tuple(settings['separators'])
    restored = (json.dumps(value, **settings) + ('\n' if suffix else '')).encode()
    if hashlib.sha256(restored).hexdigest() != row['source_sha256']:
        raise ValueError('approved JSON restoration does not recover the exact pinned source')
    return restored


def source_layer(assets, asset_id, expected_sha256):
    """Offline: recover historical reference opinions from a typed export."""
    from pxr import Sdf
    path = assets.resolve(asset_id)
    actual = _digest(path)
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    if layer is None:
        raise ValueError('source projection is not a readable USD layer')
    if actual == expected_sha256:
        return layer
    row = _projection(path, expected_sha256, actual, assets)
    _remember(assets, path, row, actual)
    proof = _restoration(row)
    changes = {r['location']: r for r in proof['usd']['tokens']}
    seen = set()

    def token(raw, location):
        edit = changes.get(location)
        if edit is None or edit['runtime'] != raw:
            raise ValueError('approved typed asset token differs at ' + location)
        seen.add(location)
        return edit['source']

    def restore(value, location):
        if isinstance(value, Sdf.AssetPath):
            raw = token(value.path, location)
            return Sdf.AssetPath(raw), raw != value.path
        if isinstance(value, Sdf.Reference):
            raw = token(value.assetPath, location)
            return Sdf.Reference(raw, value.primPath, value.layerOffset, value.customData), raw != value.assetPath
        if isinstance(value, Sdf.Payload):
            raw = token(value.assetPath, location)
            return Sdf.Payload(raw, value.primPath, value.layerOffset), raw != value.assetPath
        if isinstance(value, dict):
            result, changed = {}, False
            for key, item in value.items():
                result[key], change = restore(item, location + '/' + str(key))
                changed |= change
            return result, changed
        if isinstance(value, (list, tuple)) or type(value).__name__ == 'AssetPathArray':
            result, changed = [], False
            for i, item in enumerate(value):
                item, change = restore(item, location + '/' + str(i))
                result.append(item)
                changed |= change
            return type(value)(result) if changed else value, changed
        if type(value).__name__.endswith('ListOp'):
            result, changed = type(value)(), False
            names = ('explicitItems',) if value.isExplicit else (
                'prependedItems', 'appendedItems', 'addedItems', 'deletedItems', 'orderedItems')
            for name in names:
                items, change = restore(list(getattr(value, name)), location + '/' + name)
                setattr(result, name, items)
                changed |= change
            return result if changed else value, changed
        return value, False

    def visit(path):
        obj = layer.GetObjectAtPath(path)
        if obj is None:
            return
        for field in obj.ListInfoKeys():
            if field in ('subLayers', 'subLayerOffsets'):
                continue
            value, changed = restore(obj.GetInfo(field), str(path) + '#' + field)
            if changed:
                obj.SetInfo(field, value)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    offsets = list(layer.subLayerOffsets)
    layer.subLayerPaths = [token(raw, '/#subLayers/' + str(i))
                           for i, raw in enumerate(layer.subLayerPaths)]
    for i, offset in enumerate(offsets):
        layer.subLayerOffsets[i] = offset
    if seen != set(changes):
        raise ValueError('approved USD token locations are incomplete')
    return layer


def prepared_projection(source, assets, expected_sha256):
    """Return a verified existing export, or None for the original writer path."""
    actual = _digest(source)
    if actual == expected_sha256:
        return None
    row = _projection(source, expected_sha256, actual, assets)
    _remember(assets, source, row, actual)
    return source
