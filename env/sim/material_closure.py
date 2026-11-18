"""Offline USD/material dependency audit. No Kit, renderer, network, or fallback.

The inventory names selected scene assets and exact relocation manifests through
LocalAssets. Every authored layer (including inactive variants and USDZ members)
is inspected, rather than just the currently visible materials. Unresolved SDK
MDL tokens fail too: callers must supply explicit, portable MDL search assets.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from . import LocalAssets

USD = {'.usd', '.usda', '.usdc', '.usdz'}
MDL_BUILTINS = {'anno', 'base', 'builtins', 'df', 'math', 'state', 'tex', 'debug', 'limits', 'scene'}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _mdl_text(text):
    # Preserve quoted resource names while removing comments before inspection.
    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else ' ', text, flags=re.S)


def configure_mdl_search(assets, asset_ids, environ=None):
    """Configure standard MDL paths before SimulationApp initializes its SDK.

    Omniverse documents MDL_SYSTEM_PATH and MDL_USER_PATH as startup inputs;
    changing the material-preferences custom setting requires an app restart.
    No persistent user configuration or preexisting machine paths are reused.
    """
    if not isinstance(asset_ids, (list, tuple)) or not asset_ids or any(not isinstance(k, str) or not k for k in asset_ids):
        raise ValueError('material_mdl_search_asset_ids must be nonempty asset IDs')
    paths = [assets.resolve(key) for key in asset_ids]
    if any(not p.is_dir() for p in paths):
        raise ValueError('material MDL search asset must be a directory')
    environ = os.environ if environ is None else environ
    value = os.pathsep.join(str(p) for p in paths)
    for key in ('MDL_SYSTEM_PATH', 'MDL_USER_PATH'):
        environ[key] = value
    return dict(schema='mind2act_eval.material_resolver.v1', asset_ids=list(asset_ids),
                resolved_paths=[str(p) for p in paths], environment_keys=['MDL_SYSTEM_PATH', 'MDL_USER_PATH'],
                configured_before_simulation_app=True)


def verify_native_mdl_search(receipt, query=None):
    """Register resolver directories in the real SDK, then verify consumption.

    The SDK attachment matches Isaac 5.1 omni.kit.material.library 2.0.7.
    SDK paths can be local file URIs. Normalize those before comparison;
    explicitly register only missing directories before any scene is opened.
    """
    from urllib.parse import urlsplit
    from urllib.request import url2pathname

    def filesystem_path(value):
        parsed = urlsplit(value)
        if parsed.scheme == 'file' and parsed.netloc in ('', 'localhost'):
            return str(Path(url2pathname(parsed.path)).resolve())
        return value if parsed.scheme else str(Path(value).resolve())

    registrations = []
    if query is None:
        from omni.mdl import pymdlsdk
        import omni.mdl.neuraylib
        interface = omni.mdl.neuraylib.get_neuraylib().getNeurayAPI()
        neuray = pymdlsdk.attach_ineuray(interface)
        with neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
            def active_paths():
                paths = []
                for i in range(cfg.get_mdl_paths_length()):
                    with cfg.get_mdl_path(i) as path:
                        paths.append(path.get_c_str())
                return paths
            original_paths = active_paths()
            if receipt.get('prioritize_declared_paths'):
                # add_mdl_path appends. Random's byte-pinned release roots must
                # precede Kit's installation roots for transitive MDL imports.
                for path in original_paths:
                    code = cfg.remove_mdl_path(path)
                    if code != 0:
                        raise RuntimeError(f'native MDL path removal failed ({code}): {path}')
                ordered = list(dict.fromkeys([*receipt['resolved_paths'],
                    *(filesystem_path(p) for p in original_paths)]))
                for path in ordered:
                    code = cfg.add_mdl_path(path)
                    registrations.append(dict(path=path, return_code=code))
                    if code != 0:
                        raise RuntimeError(f'native MDL path registration failed ({code}): {path}')
            before = {filesystem_path(p) for p in active_paths()}
            for path in receipt['resolved_paths']:
                if path not in before:
                    code = cfg.add_mdl_path(path)
                    registrations.append(dict(path=path, return_code=code))
                    if code != 0:
                        raise RuntimeError(f'native MDL path registration failed ({code}): {path}')
            actual = dict(system=[cfg.get_mdl_system_path(i) for i in range(cfg.get_mdl_system_paths_length())],
                          user=[cfg.get_mdl_user_path(i) for i in range(cfg.get_mdl_user_paths_length())],
                          active=active_paths())
    else:
        actual = query()
    registered = {filesystem_path(p) for p in actual['active']}
    missing = sorted(set(receipt['resolved_paths']) - registered)
    if missing:
        raise RuntimeError('native MDL compiler is missing resolver search paths: ' + ', '.join(missing))
    if receipt.get('prioritize_declared_paths'):
        expected = list(dict.fromkeys(map(filesystem_path, receipt['resolved_paths'])))
        active = [filesystem_path(p) for p in actual['active']]
        if active[:len(expected)] != expected:
            raise RuntimeError('native MDL compiler did not prioritize pinned release paths')
    return dict(receipt, native_compiler_paths=actual, native_path_registrations=registrations,
                native_compiler_paths_verified=True)


def audit_material_graph(stage):
    from pxr import Usd, UsdShade
    from .label_materials import audit_visible_labels
    issues = []
    prims = list(stage.TraverseAll())
    for prototype in stage.GetPrototypes():
        prims.extend(Usd.PrimRange(prototype))
    for prim in prims:
        for rel in prim.GetRelationships():
            if not rel.GetName().startswith('material:binding'):
                continue
            targets = rel.GetTargets()
            material_targets = targets[1:] if rel.GetName().startswith('material:binding:collection:') else targets
            for target in material_targets:
                material = stage.GetPrimAtPath(target)
                if not material or not material.IsA(UsdShade.Material):
                    issues.append(dict(kind='unresolved_material_binding', prim=str(prim.GetPath()), target=str(target)))
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph):
            for attr in prim.GetAttributes():
                for target in attr.GetConnections():
                    if not stage.GetPropertyAtPath(target):
                        issues.append(dict(kind='unresolved_shader_connection', attribute=str(attr.GetPath()), target=str(target)))
    issues.extend(audit_visible_labels(stage))
    return issues


def audit_scene(assets, scene_asset_ids, *, relocation_asset_id=None, mdl_search_asset_ids=(), strict=False):
    from pxr import Ar, Sdf, UsdUtils

    mappings = {}
    if relocation_asset_id:
        manifest = json.loads(assets.resolve(relocation_asset_id).read_text())
        if manifest['schema'] != 'mind2act_eval.station_relocation.v1':
            raise ValueError('unsupported relocation manifest')
        for row in manifest['mappings']:
            key = (row['source_layer_sha256'], row['authored_path'])
            if key in mappings:
                raise ValueError('duplicate relocation mapping')
            mappings[key] = row
    search = [assets.resolve(key) for key in mdl_search_asset_ids]
    if any(not p.is_dir() for p in search):
        raise ValueError('MDL search asset must be a directory')
    refs, issues, visited, checked_hashes, repairs = [], [], set(), {}, []

    def issue(kind, layer, raw, **extra):
        issues.append(dict(kind=kind, layer=layer, authored_path=raw, **extra))

    def checked(path, expected):
        if path not in checked_hashes:
            checked_hashes[path] = _sha(path)
        if strict and checked_hashes[path] != expected:
            raise ValueError('material closure target SHA mismatch: ' + str(path))

    def visit_mdl(path):
        if str(path) in visited:
            return
        visited.add(str(path))
        # USD 24.5 (Isaac 5.1) has no Python Resolver.OpenAsset. Read the
        # already-resolved package members without extracting or rewriting it.
        def read(identifier):
            if not Ar.IsPackageRelativePath(identifier):
                return Path(identifier).read_bytes()
            outer, member = Ar.SplitPackageRelativePathOuter(identifier)
            def read_member(archive, name):
                if not Ar.IsPackageRelativePath(name):
                    return archive.read(name)
                nested, child = Ar.SplitPackageRelativePathOuter(name)
                with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as subarchive:
                    return read_member(subarchive, child)
            with zipfile.ZipFile(outer) as archive:
                return read_member(archive, member)
        text = _mdl_text(read(str(path)).decode('utf-8'))

        def relative_resource(name):
            if Ar.IsPackageRelativePath(str(path)):
                outer, member = Ar.SplitPackageRelativePathInner(str(path))
                from posixpath import normpath, dirname, join
                return Ar.JoinPackageRelativePath(outer, normpath(join(dirname(member), name)))
            return str(Path(path).parent / name)
        imports = re.findall(r'\bimport\s+([^;]+);|\busing\s+(\S+)\s+import\b', text)
        for direct, using in imports:
            for token in (using or direct).split(','):
                token = token.strip()
                parts = token.strip(':').split('::')
                if parts[0] in MDL_BUILTINS:
                    continue
                resolved = None
                for count in range(len(parts), 0, -1):
                    name = '/'.join(parts[:count]) + '.mdl'
                    candidates = [str(p/name) for p in search]
                    if not token.startswith('::'):
                        candidates.insert(0, relative_resource(name))
                    resolved = next((str(Ar.GetResolver().Resolve(p)) for p in candidates
                                     if Ar.GetResolver().Resolve(p)), None)
                    if resolved:
                        break
                refs.append(dict(layer=str(path), authored_path=token, kind='mdl_import', resolved=str(resolved or '')))
                if resolved:
                    visit_mdl(resolved)
                else:
                    issue('unresolved_mdl_import', str(path), token)
        for name in re.findall(r'\b(?:texture_2d|texture_3d|light_profile|bsdf_measurement)\s*\(\s*"([^"\n]+)"', text):
            target = str(Ar.GetResolver().Resolve(relative_resource(name)))
            refs.append(dict(layer=str(path), authored_path=name, kind='mdl_resource', resolved=target))
            if Path(name).is_absolute() or re.match(r'\w+://', name):
                issue('nonportable_mdl_resource', str(path), name)
            if not target:
                issue('unresolved_mdl_resource', str(path), name)

    def visit(identifier):
        identifier = str(identifier)
        if identifier in visited:
            return
        if Path(identifier).suffix.lower() == '.mdl':
            visit_mdl(identifier)
            return
        visited.add(identifier)
        layer = Sdf.Layer.FindOrOpen(identifier)
        if layer is None:
            issue('unreadable_layer', identifier, identifier)
            return
        physical = Ar.SplitPackageRelativePathOuter(layer.realPath)[0] if Ar.IsPackageRelativePath(layer.realPath) else layer.realPath
        digest = _sha(physical)
        copy = Sdf.Layer.CreateAnonymous('material-audit.usda')
        copy.TransferContent(layer)
        raw_paths = set()
        UsdUtils.ModifyAssetPaths(copy, lambda raw: (raw_paths.add(raw) or raw) if raw else raw)
        for raw in sorted(raw_paths):
            mapping = mappings.get((digest, raw))
            if mapping:
                target = assets.resolve(mapping['target_asset_id'])
                checked(target, mapping['target_sha256'])
                anchored = str(target)
                # Match prepare_station's package member preservation.
                if Ar.IsPackageRelativePath(raw):
                    _, member = Ar.SplitPackageRelativePathOuter(raw)
                    anchored = Ar.JoinPackageRelativePath(anchored, member)
            else:
                anchored = Sdf.ComputeAssetPathRelativeToLayer(layer, raw)
            resolved = str(Ar.GetResolver().Resolve(anchored))
            if not resolved and not mapping and raw.endswith('.mdl'):
                resolved = str(next((p / raw for p in search if (p / raw).is_file()), ''))
            row = dict(layer=identifier, source_layer_sha256=digest, authored_path=raw,
                       resolved=resolved, mapped=bool(mapping))
            refs.append(row)
            if not mapping and (Path(raw).is_absolute() or re.match(r'^[A-Za-z][\w+.-]*:', raw)):
                issue('nonportable_reference', identifier, raw)
            if not resolved:
                issue('unresolved_reference', identifier, raw)
                continue
            if not Ar.IsPackageRelativePath(resolved) and not Path(resolved).resolve().is_relative_to(assets.root):
                issue('reference_outside_asset_root', identifier, raw, resolved=resolved)
            member = Ar.SplitPackageRelativePathInner(resolved)[1] if Ar.IsPackageRelativePath(resolved) else resolved
            suffix = Path(member).suffix.lower()
            if suffix in USD:
                visit(resolved)
            elif suffix == '.mdl':
                visit_mdl(resolved)

    for asset_id in scene_asset_ids:
        visit(assets.resolve(asset_id))
    if not issues:
        # Once every external dependency is available, validate the actual
        # composed binding/connection graph too. No task build or physics runs.
        from pxr import Usd
        from .asset_relocation import prepare_station
        with tempfile.TemporaryDirectory(prefix='mind2act-material-audit-') as temporary:
            for index, asset_id in enumerate(scene_asset_ids):
                source = assets.resolve(asset_id)
                runtime = prepare_station(source, assets, relocation_asset_id if index == 0 else None,
                                          Path(temporary).resolve()/str(index))
                stage = Usd.Stage.Open(str(runtime), load=Usd.Stage.LoadAll)
                if stage is None:
                    issue('unreadable_stage', str(runtime), str(runtime))
                else:
                    from mind2act_eval.robot import PROFILE
                    from mind2act_eval.robot.appearance import repair_camera_material_binding
                    if _sha(source) == PROFILE['asset_sha256'] and stage.GetDefaultPrim():
                        repairs.extend(repair_camera_material_binding(stage, [stage.GetDefaultPrim().GetPath()]))
                    issues.extend(audit_material_graph(stage))
    return dict(passed=not issues, references=refs, issues=issues, layers=len(visited), render_repairs=repairs)


def audit_inventory(document, asset_root):
    results = []
    for row in document['scenes']:
        identity = {k: row[k] for k in ('task_id', 'difficulty')}
        try:
            entries = row['assets']
            if any(Path(p).is_absolute() or re.match(r'^[A-Za-z][\w+.-]*:', p) for p in entries.values()):
                raise ValueError('inventory asset entries must be relative paths')
            assets = LocalAssets(asset_root, entries)
            # Runtime-created labels/screens can consume declared fonts/images
            # that are not yet referenced by the authored station layer.
            for asset_id in entries:
                assets.resolve(asset_id)
            result = audit_scene(assets, row['scene_asset_ids'],
                relocation_asset_id=row.get('relocation_asset_id'),
                mdl_search_asset_ids=row.get('mdl_search_asset_ids', ()))
        except Exception as exc:
            result = dict(passed=False, references=[], issues=[dict(kind='audit_error', error=f'{type(exc).__name__}: {exc}')])
        results.append(dict(**identity, **result))
    tasks = sorted({row['task_id'] for row in results})
    required = set(document.get('required_tasks', tasks))
    uncovered = sorted(required - set(tasks))
    return dict(schema='mind2act_eval.material_closure_report.v1', passed=not uncovered and all(r['passed'] for r in results),
                task_count=len(tasks), scene_count=len(results), uncovered_tasks=uncovered,
                failed_tasks=sorted({r['task_id'] for r in results if not r['passed']}), scenes=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_inventory(json.loads(args.inventory.read_text()), args.asset_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'scenes'}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
