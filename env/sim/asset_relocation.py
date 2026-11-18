"""Opt-in, hash-pinned USD asset-path relocation before a station is opened.

This does not qualify materials, appearance, physics, or native task success.
Source files and the original LocalAssets station identity are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

SCHEMA = 'mind2act_eval.station_relocation.v1'
USD_SUFFIXES = {'.usd', '.usda', '.usdc'}
GEOMETRY_SUFFIXES = USD_SUFFIXES | {'.usdz', '.abc', '.obj', '.ply', '.stl'}
# POSIX ERE boundaries exclude relative path suffixes and USD prim targets.
DELIVERY_FORBIDDEN_PATTERN = (
    r'(^|[^[:alnum:]_.<\\/-])'
    r'([Ff][Ii][Ll][Ee]:(//[^/[:space:]]*)?)?'
    r'/(home|root|shared|Users)(/|$|[^[:alnum:]_.-])'
)


def _relative_file_path(layer, target, *, must_exist):
    from pxr import Ar
    from urllib.parse import urlsplit
    from urllib.request import url2pathname
    if not layer.realPath or Ar.IsPackageRelativePath(layer.realPath):
        raise ValueError('relative asset authoring requires a file-backed loose USD layer')
    raw = str(target)
    parsed = urlsplit(raw)
    if parsed.scheme == 'file' and parsed.netloc in ('', 'localhost'):
        raw = url2pathname(parsed.path)
    elif parsed.scheme:
        raise ValueError('relative asset target must be a resolved local file')
    outer, member = Ar.SplitPackageRelativePathOuter(raw) if Ar.IsPackageRelativePath(raw) else (raw, None)
    path = Path(outer).resolve(strict=must_exist)
    if must_exist and (not path.is_file() or member and not Ar.GetResolver().Resolve(raw)):
        raise FileNotFoundError(raw)
    token = os.path.relpath(path, Path(layer.realPath).parent)
    return Ar.JoinPackageRelativePath(token, member) if member else token


def relative_asset_path(layer, target):
    """Return a verified local target relative to a file-backed authoring layer.

    Callers wrap this string in Sdf.AssetPath for shader/texture inputs, or use
    it directly for references. Anonymous overlays must receive a file anchor
    before authoring; no current-directory or machine-cache fallback is used.
    """
    return _relative_file_path(layer, target, must_exist=True)


def export_relative_layer(source, destination):
    """Export a fresh relative-path copy without editing the live/source layer."""
    from pxr import Ar, Sdf, UsdUtils
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = Sdf.Layer.CreateNew(str(destination))
    output.TransferContent(source)

    def relocate(raw):
        if not raw:
            return raw
        uri = re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', raw) is not None
        if uri:
            # Local SDK file URIs receive the same normalization as PR100.
            return _relative_file_path(output, raw, must_exist=False)
        anchored = Sdf.ComputeAssetPathRelativeToLayer(source, raw)
        if (raw.endswith('.mdl') and not Path(raw).is_absolute()
                and not raw.startswith(('./', '../')) and not Ar.GetResolver().Resolve(anchored)):
            return raw  # SDK symbolic module, not an existing layer-local MDL.
        outer = Ar.SplitPackageRelativePathOuter(anchored)[0] if Ar.IsPackageRelativePath(anchored) else anchored
        if not Path(outer).is_absolute():
            if not source.realPath or Ar.IsPackageRelativePath(source.realPath):
                raise ValueError('relative source asset needs a file-backed source anchor')
            anchored = str((Path(source.realPath).parent / raw).resolve())
        return _relative_file_path(output, anchored, must_exist=False)

    def non_asset_text(layer):
        masked = Sdf.Layer.CreateAnonymous('non-asset.usda')
        masked.TransferContent(layer)
        UsdUtils.ModifyAssetPaths(masked, lambda raw: '__asset_path__' if raw else raw)
        return masked.ExportToString()

    before = non_asset_text(source)
    UsdUtils.ModifyAssetPaths(output, relocate)
    if non_asset_text(output) != before:
        raise ValueError('runtime export changed non-asset opinions')
    output.Save()
    report = audit_generated_layers([destination])
    if not report['passed']:
        raise ValueError('nonportable runtime export: ' + json.dumps(report['issues']))
    return destination


def audit_delivery_tree(root):
    """Apply the delivery acceptance grep to every file, not only active USD."""
    import subprocess
    root = Path(root)
    if not root.is_dir():
        raise ValueError('delivery tree must be a directory')
    command = ['grep', '-rEl', DELIVERY_FORBIDDEN_PATTERN, str(root)]
    result = subprocess.run(command, capture_output=True, text=True, errors='replace')
    return dict(passed=result.returncode == 1 and not result.stdout and not result.stderr, command=command,
        returncode=result.returncode, matching_files=result.stdout.splitlines(),
        error=result.stderr or None)


def audit_generated_layers(paths):
    """Read generated USD layers on CPU; reject machine or network asset tokens.

    Scan authored opinions, including unused variants and time samples, rather
    than only a composed stage. This is a publication check, not the runtime
    visible-material validity gate. Plain strings containing provenance paths
    are not asset references. Source USDZ packages are never rewritten here.
    """
    from pxr import UsdUtils, Sdf

    files, issues, tokens = set(), [], 0
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.update(p for p in path.rglob('*') if p.suffix.lower() in USD_SUFFIXES and p.is_file())
        elif path.is_file() and path.suffix.lower() in USD_SUFFIXES:
            files.add(path)
        else:
            issues.append(dict(layer=str(path), reason='missing_or_unsupported_layer'))
    for path in sorted(files):
        try:
            # OpenAsAnonymous reads the file afresh and keeps source files and
            # the process-wide layer registry unchanged.
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            if layer is None:
                raise ValueError('cannot read layer')
            found = set()
            def inspect(raw):
                if raw:
                    found.add(raw)
                return raw
            UsdUtils.ModifyAssetPaths(layer, inspect)
            tokens += len(found)
            for raw in sorted(found):
                if raw.startswith(('/', '\\', '~', '$')) or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', raw):
                    issues.append(dict(layer=str(path), asset_path=raw,
                        reason='non_relative_asset_path'))
        except Exception as exc:
            issues.append(dict(layer=str(path), reason='unreadable_layer', error=str(exc)))
    if not files and not issues:
        issues.append(dict(reason='no_generated_layers'))
    return dict(schema='mind2act_eval.generated_layer_portability.v1',
        passed=not issues, layers_checked=len(files), asset_paths_checked=tokens, issues=issues)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _hash_value(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('relocation requires a full SHA256')
    return value


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _manifest_rules(manifest):
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported station relocation manifest')
    if manifest.get('asset_path_mode', 'relative') != 'relative':
        raise ValueError('generated layers require relative asset_path_mode; regenerate the manifest')
    expected_source = _hash_value(manifest['source_sha256'])
    rules = {}
    for row in manifest['mappings']:
        key = (_hash_value(row['source_layer_sha256']), row['authored_path'])
        if not isinstance(key[1], str) or not key[1] or key in rules:
            raise ValueError('duplicate or invalid exact asset-path mapping')
        if not isinstance(row['target_asset_id'], str) or not row['target_asset_id']:
            raise ValueError('relocation target must be an asset ID')
        _hash_value(row['target_sha256'])
        rules[key] = row
    return expected_source, rules


def prepare_station(original_station, assets, manifest_asset_id, output_dir, *, strict=False):
    """Relocate current assets into this episode.

    Explicit strict mode additionally verifies provenance and uses the immutable
    node cache. Normal runs rebuild the projection to avoid stale cached bytes.
    """
    if not strict:
        return _prepare_station(original_station, assets, manifest_asset_id, output_dir)
    if manifest_asset_id is None:
        return original_station
    if not isinstance(manifest_asset_id, str) or not manifest_asset_id:
        raise ValueError('station_relocation_manifest must be an asset ID')
    from pxr import Sdf
    from .source_assets import asset_bytes
    from .asset_provenance import prepared_projection
    from mind2act_eval.episode_storage import node_cache, durable_json, _physical
    import fcntl

    source = Path(original_station).resolve(strict=True)
    manifest_bytes = asset_bytes(assets, manifest_asset_id)
    manifest = json.loads(manifest_bytes)
    expected_source, _ = _manifest_rules(manifest)
    prepared = prepared_projection(source, assets, expected_source) if strict else None
    if prepared is not None:
        return prepared
    output = _physical(output_dir)
    if source.is_relative_to(output):
        raise ValueError('relocation output must not contain the source station')
    cache = node_cache(output) / 'station-layers'
    key = hashlib.sha256(manifest_bytes + str(source).encode() + str(assets.root).encode()
                         + Path(__file__).read_bytes()).hexdigest()
    entry = cache / key
    cache_receipt = entry / 'station-relocation.json'
    lock = os.open(cache / (key + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not entry.exists():
            entry.mkdir()
            _prepare_station(original_station, assets, manifest_asset_id, entry, strict=True)
            # Cache publication is durable and immutable; failed partial builds
            # remain diagnostic evidence and cannot be used on a later attempt.
            built = json.loads(cache_receipt.read_text())
            from mind2act_eval.sealing import _fsync_file, _fsync_directories
            for row in built['layers']:
                path = Path(row['runtime_path'])
                if path.is_relative_to(entry):
                    path.chmod(0o444)
                    _fsync_file(path)
            _fsync_directories(entry)
            durable_json(cache_receipt, built)
            cache_receipt.chmod(0o444)
        # Incomplete/failed entries are never adopted or silently removed.
        _physical(entry)
        receipt = json.loads(cache_receipt.read_text())
        if (receipt.get('status') != 'prepared'
                or receipt['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest()
                or receipt['source_path'] != str(source)):
            raise ValueError('immutable station cache is incomplete or mismatched')
        for row in receipt['layers']:
            for prefix in ('source', 'runtime'):
                path = _physical(row[prefix + '_path'])
                if _sha(path) != row[prefix + '_sha256']:
                    raise ValueError('immutable station cache/input SHA mismatch: ' + str(path))
        for row in receipt['mappings']:
            target = assets.resolve(row['target_asset_id'])
            if str(target) != row['target_path'] or _sha(target) != row['target_sha256']:
                raise ValueError('immutable station mapping target SHA mismatch')
        immutable = _physical(receipt['runtime_path'])
        if _sha(immutable) != receipt['runtime_sha256']:
            raise ValueError('immutable station runtime SHA mismatch')
        directory = output / 'station-relocation'
        directory.mkdir(parents=True, exist_ok=False)
        thin = directory / 'station.usda'
        layer = Sdf.Layer.CreateNew(str(thin))
        layer.subLayerPaths = [relative_asset_path(layer, immutable)]
        immutable_layer = Sdf.Layer.FindOrOpen(str(immutable))
        for field in immutable_layer.pseudoRoot.ListInfoKeys():
            if field not in {'subLayers', 'subLayerOffsets'}:
                layer.pseudoRoot.SetInfo(field, immutable_layer.pseudoRoot.GetInfo(field))
        layer.Save()
        receipt.update(immutable_runtime_path=str(immutable),
            immutable_runtime_sha256=receipt['runtime_sha256'],
            runtime_path=str(thin), runtime_sha256=_sha(thin), storage='thin_node_cached_projection')
        durable_json(output / 'station-relocation.json', receipt)
        return thin
    except Exception:
        # Keep the original composition/mapping failure visible in this episode,
        # even though heavy generated data lives in the persistent node cache.
        if cache_receipt.is_file() and not (output / 'station-relocation.json').exists():
            output.mkdir(parents=True, exist_ok=True)
            durable_json(output / 'station-relocation.json', json.loads(cache_receipt.read_text()))
        raise
    finally:
        os.close(lock)


def _prepare_station(original_station, assets, manifest_asset_id, output_dir, *, strict=False):
    """Return the original path without I/O when no manifest is configured."""
    if manifest_asset_id is None:
        return original_station
    if not isinstance(manifest_asset_id, str) or not manifest_asset_id:
        raise ValueError('station_relocation_manifest must be an asset ID')
    from pxr import Ar, Sdf, Usd, UsdUtils

    source = Path(original_station).resolve(strict=True)
    manifest_path = assets.resolve(manifest_asset_id)
    from .source_assets import asset_bytes
    from .asset_provenance import prepared_projection
    manifest_bytes = asset_bytes(assets, manifest_asset_id)
    manifest = json.loads(manifest_bytes)
    expected_source, rules = _manifest_rules(manifest)
    prepared = prepared_projection(source, assets, expected_source) if strict else None
    if prepared is not None:
        return prepared
    output = Path(output_dir).resolve()
    if source.is_relative_to(output):
        raise ValueError('relocation output must not contain the source station')
    directory = output / 'station-relocation'
    directory.mkdir(parents=True, exist_ok=False)
    receipt_path = output / 'station-relocation.json'
    receipt = dict(schema=SCHEMA, status='preparing', source_path=str(source),
        source_sha256=expected_source, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        layers=[], mappings=[], unverified_assets=[], composition_errors=[])
    cache, pins, used_rules = {}, {}, set()

    def checked(path, expected):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        if strict and _sha(path) != expected:
            raise ValueError('relocation source/target SHA mismatch: ' + str(path))
        pins[path] = expected
        return path

    def masked_digest(layer):
        copy = Sdf.Layer.CreateAnonymous('masked.usda')
        copy.TransferContent(layer)
        UsdUtils.ModifyAssetPaths(copy, lambda raw: '__asset_path__' if raw else raw)
        return hashlib.sha256(copy.ExportToString().encode()).hexdigest()

    def derive(path, expected):
        path = checked(path, expected)
        if path in cache:
            return cache[path]
        if path.suffix.lower() == '.usdz':
            # Preserve the archive and all member geometry byte-for-byte. USD's
            # package resolver handles internal members; composition errors are
            # still checked on the final stage below. Never unpack/rewrite it.
            cache[path] = path
            receipt['layers'].append(dict(source_path=str(path), source_sha256=expected,
                runtime_path=str(path), runtime_sha256=expected, kind='immutable_package'))
            return path
        original = Sdf.Layer.FindOrOpen(str(path))
        if original is None:
            raise ValueError('cannot parse relocation source layer: ' + str(path))
        runtime = Sdf.Layer.CreateAnonymous('relocated.usda')
        runtime.TransferContent(original)
        destination = directory / (str(len(cache)).zfill(4) + '.usda')
        cache[path] = destination
        composition = set(original.GetExternalReferences()) | set(original.subLayerPaths)

        def relocate(raw):
            if not raw:
                return raw
            rule = rules.get((expected, raw))
            outer, member = Ar.SplitPackageRelativePathOuter(raw) if Ar.IsPackageRelativePath(raw) else (raw, None)
            suffix = Path(outer).suffix.lower()
            if rule is not None:
                used_rules.add((expected, raw))
                target = checked(assets.resolve(rule['target_asset_id']), rule['target_sha256'])
                runtime_target = derive(target, rule['target_sha256']) if suffix in USD_SUFFIXES | {'.usdz'} else target
                target_token = os.path.relpath(runtime_target, destination.parent)
                replacement = Ar.JoinPackageRelativePath(target_token, member) if member else target_token
                receipt['mappings'].append(dict(source_layer_sha256=expected, authored_path=raw,
                    target_asset_id=rule['target_asset_id'], target_path=str(target),
                    target_sha256=rule['target_sha256'], runtime_asset_path=replacement))
                return replacement
            if raw in composition or suffix in GEOMETRY_SUFFIXES:
                if not strict:
                    resolved = Sdf.ComputeAssetPathRelativeToLayer(original, outer)
                    target = Path(resolved)
                    if not target.is_absolute():
                        target = path.parent / target
                    target = target.resolve(strict=True)
                    runtime_target = derive(target, _sha(target)) if suffix in USD_SUFFIXES | {'.usdz'} else checked(target, None)
                    token = os.path.relpath(runtime_target, destination.parent)
                    return Ar.JoinPackageRelativePath(token, member) if member else token
                raise ValueError('missing exact geometry/composition mapping: ' + str(path) + ' -> ' + raw)
            # SDK symbolic MDLs must keep their tokens. Other relative soft
            # assets keep the ORIGINAL layer anchor when the derived file moves.
            # An unknown texture remains unverified/missing, never substituted.
            uri = re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', raw) is not None
            symbolic_mdl = suffix == '.mdl' and not Path(raw).is_absolute() and not uri
            # Filesystem anchoring can collapse a URI's double slash in Sdf.
            # Unmapped URI tokens belong to their resolver, unchanged.
            replacement = raw if symbolic_mdl or uri else Sdf.ComputeAssetPathRelativeToLayer(original, raw)
            if (not symbolic_mdl and not uri and not Path(raw).is_absolute()
                    and not Ar.IsPackageRelativePath(raw) and not Path(replacement).is_absolute()):
                # Sdf leaves a missing search-path asset unchanged; retain its
                # original filesystem anchor rather than the derived directory.
                replacement = str((path.parent / raw).resolve())
            if Path(replacement).is_absolute():
                replacement = os.path.relpath(replacement, destination.parent)
            receipt['unverified_assets'].append(dict(source_layer_sha256=expected,
                authored_path=raw, runtime_asset_path=replacement,
                status='deferred_sdk_mdl' if symbolic_mdl else 'unverified_shader_or_texture'))
            return replacement

        UsdUtils.ModifyAssetPaths(runtime, relocate)
        source_digest = masked_digest(original)
        if masked_digest(runtime) != source_digest:
            raise ValueError('relocation changed non-asset opinions')
        runtime.Export(str(destination))
        reopened = Sdf.Layer.FindOrOpen(str(destination))
        runtime_digest = masked_digest(reopened)
        if runtime_digest != source_digest:
            raise ValueError('relocation export changed non-asset opinions')
        receipt['layers'].append(dict(source_path=str(path), source_sha256=expected,
            runtime_path=str(destination), runtime_sha256=_sha(destination),
            source_non_asset_sha256=source_digest, runtime_non_asset_sha256=runtime_digest,
            kind='derived_layer'))
        return destination

    try:
        derived = derive(source, expected_source)
        if strict and set(rules) != used_rules:
            raise ValueError('manifest contains unused layer/path mappings')
        receipt['generated_layer_portability'] = audit_generated_layers([directory])
        if not receipt['generated_layer_portability']['passed']:
            raise ValueError('nonportable generated layer asset paths: ' +
                json.dumps(receipt['generated_layer_portability']['issues']))
        stage = Usd.Stage.Open(str(derived), load=Usd.Stage.LoadAll)
        if stage is None:
            raise ValueError('cannot compose relocated station')
        # Isaac's USD build predates Stage.GetCompositionErrors. Its public
        # prim indices expose the same composition failures without Kit APIs.
        if hasattr(stage, 'GetCompositionErrors'):
            errors = stage.GetCompositionErrors()
        else:
            prims = [stage.GetPseudoRoot(), *stage.TraverseAll()]
            for prototype in stage.GetPrototypes():
                prims.extend(Usd.PrimRange(prototype))
            errors = [error for prim in prims for error in prim.GetPrimIndex().localErrors]
        receipt['composition_errors'] = list(dict.fromkeys(str(error) for error in errors))
        if receipt['composition_errors']:
            raise ValueError('relocated USD composition errors: ' + '; '.join(receipt['composition_errors']))
        for layer in stage.GetUsedLayers():
            if not layer.realPath:
                continue
            outer, _ = Ar.SplitPackageRelativePathOuter(layer.realPath) if Ar.IsPackageRelativePath(layer.realPath) else (layer.realPath, None)
            physical = Path(outer).resolve(strict=True)
            if physical not in pins and not physical.is_relative_to(directory):
                raise ValueError('unmapped external composition layer: ' + layer.identifier)
            if physical.suffix.lower() == '.usdz':
                # Inspect package material tokens without rewriting or unpacking
                # the archive. These observations never become an offline gate.
                copy = Sdf.Layer.CreateAnonymous('inspect-package.usda')
                copy.TransferContent(layer)
                composition = set(layer.GetExternalReferences()) | set(layer.subLayerPaths)
                def inspect(raw):
                    if raw and raw not in composition and Path(raw).suffix.lower() not in GEOMETRY_SUFFIXES:
                        receipt['unverified_assets'].append(dict(source_package_sha256=pins[physical],
                            authored_layer=layer.identifier, authored_path=raw, runtime_asset_path=raw,
                            status='package_material_deferred_to_native'))
                    return raw
                UsdUtils.ModifyAssetPaths(copy, inspect)
        for path, expected in pins.items():
            checked(path, expected)
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError('relocation manifest changed during preparation')
        receipt.update(status='prepared', runtime_path=str(derived), runtime_sha256=_sha(derived))
        _write(receipt_path, receipt)
        return derived
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _write(receipt_path, receipt)
        raise


def main(argv=None):
    """Scan only explicitly selected generated-layer trees; no Kit or GPU."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--delivery-tree', action='append', default=[], type=Path,
        help='also run the literal delivery grep against this entire tree')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.paths and not args.delivery_tree:
        parser.error('select generated layer paths or a delivery tree')
    report = audit_generated_layers(args.paths + args.delivery_tree)
    report['delivery_trees'] = [audit_delivery_tree(root) for root in args.delivery_tree]
    report['passed'] = report['passed'] and all(r['passed'] for r in report['delivery_trees'])
    if args.output:
        _write(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
