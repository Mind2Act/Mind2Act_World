"""Read-only resolution gate for the composed stage, not an authored-file audit.

No hashes, stage edits, network clients or cluster-specific exemptions. Resolve
under the stage's actual Ar context; only live geometry/physics and reachable
bound material resources can invalidate an episode.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
import weakref
from urllib.parse import urlsplit
from urllib.request import url2pathname

REPORT_NAME = 'asset-resolution-report.json'


class AssetUnresolved(RuntimeError):
    invalid_reason = 'asset_unresolved'

    def __init__(self, report):
        self.report = report
        self.missing_paths = report['missing_paths']
        super().__init__('asset_unresolved: ' + ', '.join(self.missing_paths))


def _file_path(value):
    parsed = urlsplit(value)
    if parsed.scheme == 'file' and parsed.netloc in ('', 'localhost'):
        return url2pathname(parsed.path)
    return value


def native_mdl_paths():
    # Reuse PR100's attachment to the *active* SDK. An empty registration list
    # makes this a read-only query; do not register/configure any extra paths.
    from .material_closure import verify_native_mdl_search
    receipt = verify_native_mdl_search({'resolved_paths': []})
    return tuple(_file_path(p) for p in receipt['native_compiler_paths']['active'])


def _native_mdl_module(raw):
    """Ask the compiler about symbolic/archived modules, never exempt by name."""
    from omni.mdl import pymdlsdk
    import omni.mdl.neuraylib
    interface = omni.mdl.neuraylib.get_neuraylib().getNeurayAPI()
    neuray = pymdlsdk.attach_ineuray(interface)
    name = '::' + raw[:-4].replace('/', '::')
    with neuray.get_api_component(pymdlsdk.IMdl_configuration) as config:
        with config.get_entity_resolver() as resolver:
            with resolver.resolve_module(name, None, None, 0, 0, None) as module:
                if module.is_valid_interface():
                    return module.get_filename() or ('mdl:' + name)
    return ''


def _native_mdl_compile(identifier):
    """Load through Omniverse's tracked compiler, not just its entity resolver."""
    import omni.mdl.neuraylib
    backend = omni.mdl.neuraylib.get_neuraylib()
    module = None
    try:
        module = backend.createMdlModule(identifier)
        return dict(passed=bool(module and module.valid()), compiler='active_omni_mdl',
                    identifier=identifier)
    except Exception as exc:
        return dict(passed=False, compiler='active_omni_mdl', identifier=identifier,
                    detail=f'{type(exc).__name__}: {exc}')
    finally:
        if module is not None:
            backend.destroyMdlModule(module)


def inspect_stage(stage, *, mdl_search_paths=None, mdl_cache=None):
    """Inspect used resources, transitive MDL dependencies and native compilation.

    mdl_search_paths is a CPU-fixture seam. Native callers omit it and query the
    current Kit SDK configuration and compiler. CPU fixtures report compilation
    as not_run rather than silently claiming a compiler pass.
    """
    from pxr import Ar, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdVol
    started = time.monotonic()
    used, errors, warnings, seen = [], [], [], set()
    mdl_checks = {}
    mdl_cache = {} if mdl_cache is None else mdl_cache
    mdl_paths = mdl_search_paths
    resolver = Ar.GetResolver()

    def add(bucket, kind, path, prim='', **extra):
        key = (id(bucket), kind, path, prim)
        if key not in seen:
            seen.add(key)
            bucket.append(dict(kind=kind, path=path, prim=prim, **extra))

    def visible(prim):
        current = prim
        while current and not current.IsPseudoRoot():
            if current.IsA(UsdGeom.Imageable):
                image = UsdGeom.Imageable(current)
                if image.ComputeVisibility(Usd.TimeCode(0)) == UsdGeom.Tokens.invisible:
                    return False
                if image.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                    return False
                return True
            current = current.GetParent()
        return False

    def physical(prim):
        for api, field in ((UsdPhysics.CollisionAPI, 'physics:collisionEnabled'),
                           (UsdPhysics.RigidBodyAPI, 'physics:rigidBodyEnabled')):
            if prim.HasAPI(api) and prim.GetAttribute(field).Get(Usd.TimeCode(0)) is not False:
                return True
        return False

    def geometry(prim):
        if physical(prim):
            return True
        # A missing reference can leave an otherwise empty visible Xform. That
        # selected render anchor is still used; an unused typed Scope is not.
        if prim.IsA(UsdGeom.Imageable) and not prim.IsA(UsdGeom.Scope):
            return visible(prim)
        if not prim.GetTypeName() and (prim.HasAuthoredReferences() or prim.HasAuthoredPayloads()):
            # Missing referenced geometry can also remove its inherited type.
            # Scope can group real scene objects: continue through it to the
            # render anchor instead of hiding a missing inherited geometry type.
            # A standalone library Scope has no such anchor.
            parent = prim.GetParent()
            while parent and not parent.IsPseudoRoot():
                if parent.IsA(UsdShade.Material) or parent.IsA(UsdShade.NodeGraph):
                    return False
                if parent.IsA(UsdGeom.Imageable) and (not parent.IsA(UsdGeom.Scope)
                                                     or parent == stage.GetDefaultPrim()):
                    return visible(prim)
                parent = parent.GetParent()
            return prim == stage.GetDefaultPrim()
        return False

    def resolve_asset(raw, layer, resolved_hint=''):
        nonlocal mdl_paths
        candidates = [resolved_hint] if resolved_hint else []
        candidates += [Sdf.ComputeAssetPathRelativeToLayer(layer, raw) if layer else raw, raw]
        symbolic = (raw.endswith('.mdl') and not urlsplit(raw).scheme and not Path(raw).is_absolute()
                    and not raw.startswith(('./', '../')))
        if symbolic:
            if mdl_paths is None:
                mdl_paths = native_mdl_paths()
            candidates.extend(str(Path(_file_path(p)) / raw) for p in mdl_paths)
        for candidate in dict.fromkeys(candidates):
            candidate = _file_path(candidate)
            if not candidate:
                continue
            resolved = str(resolver.Resolve(candidate))
            if resolved:
                physical = _file_path(resolved)
                outer = Ar.SplitPackageRelativePathOuter(physical)[0] if Ar.IsPackageRelativePath(physical) else physical
                if urlsplit(outer).scheme or Path(outer).is_file():
                    return resolved
            # UDIM tokens describe a set, not a literal absent filename.
            if '<UDIM>' in candidate and not Ar.IsPackageRelativePath(candidate):
                path = Path(candidate)
                pattern = path.name.replace('<UDIM>', '[0-9][0-9][0-9][0-9]')
                if any(p.is_file() for p in path.parent.glob(pattern)):
                    return candidate
        if symbolic and mdl_search_paths is None:
            return _native_mdl_module(raw)
        return ''

    with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
        # Instance proxies include the actual uses of prototype geometry. Never
        # traverse authored inactive variants or uninstantiated class libraries.
        prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
        consumers = [p for p in prims if geometry(p) and not p.IsA(UsdGeom.Xform)]
        consumers += [p for p in prims if p.IsA(UsdGeom.Subset) and geometry(p.GetParent())]
        materials, graph_paths = {}, set()
        drawn = [p for p in consumers if visible(p)]
        # Cameras/lights are Imageable and may carry arbitrary material bindings,
        # but do not draw those surfaces. Their schema file inputs are checked
        # separately below (for example a dome light's environment texture).
        surfaces = [p for p in drawn if any(p.IsA(schema) for schema in
                    (UsdGeom.Gprim, UsdGeom.PointInstancer, UsdGeom.Subset, UsdVol.Volume))]
        if surfaces:
            bound, _ = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(surfaces, materialPurpose='full')
            for prim, material in zip(surfaces, bound):
                if material:
                    materials[str(material.GetPath())] = material
                elif any(r.GetName().startswith('material:binding') for r in prim.GetRelationships()):
                    add(warnings, 'dangling_or_default_material', str(prim.GetPath()), str(prim.GetPath()))
        physical_consumers = [p for p in prims if physical(p)]
        if physical_consumers:
            bound, _ = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(physical_consumers, materialPurpose='physics')
            graph_paths.update(str(mat.GetPath()) for mat in bound if mat)

        def asset_attribute(attr, kind='shader_asset'):
            nonlocal mdl_paths
            if attr.GetConnections():
                return  # The connected value, not an authored fallback, wins.
            value = attr.Get(Usd.TimeCode(0))
            values = value if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray else [value]
            stack = attr.GetPropertyStack()
            layer = stack[0].layer if stack else stage.GetRootLayer()
            for value in values or []:
                if not isinstance(value, Sdf.AssetPath) or not value.path:
                    continue
                resolved = resolve_asset(value.path, layer, value.resolvedPath)
                add(used, kind, value.path, str(attr.GetPrim().GetPath()),
                    attribute=str(attr.GetPath()), resolved=resolved)
                if not resolved:
                    add(errors, 'missing_used_asset', value.path, str(attr.GetPrim().GetPath()),
                        attribute=str(attr.GetPath()), layer=layer.identifier)
                elif value.path.endswith('.mdl') and resolved not in mdl_checks:
                    from .mdl_dependencies import audit_mdl
                    if mdl_paths is None:
                        mdl_paths = native_mdl_paths()
                    paths = tuple(_file_path(p) for p in mdl_paths)
                    key = (resolved, paths, mdl_search_paths is None)
                    check = mdl_cache.get(key)
                    if check is None:
                        check = audit_mdl(_file_path(resolved), paths)
                        if check['passed'] and mdl_search_paths is None:
                            check['compilation'] = _native_mdl_compile(resolved)
                        if check['passed'] and (mdl_search_paths is not None or check['compilation']['passed']):
                            mdl_cache[key] = check
                    mdl_checks[resolved] = check
                    for issue in check['issues']:
                        add(errors, issue['kind'], issue['path'], str(attr.GetPrim().GetPath()),
                            mdl_module=resolved, dependency_layer=issue.get('layer', resolved))
                    if check['passed'] and mdl_search_paths is None:
                        compiled = check['compilation']
                        if not compiled['passed']:
                            add(errors, 'mdl_compilation_failed', value.path,
                                str(attr.GetPrim().GetPath()), mdl_module=resolved)

        # Dome-light/volume and other schema-declared file resources are used
        # directly, without a material binding. Custom unused asset metadata is
        # not a renderer/physics consumer merely because its type is "asset".
        for prim in drawn:
            definition = prim.GetPrimDefinition()
            for attr in prim.GetAttributes():
                if (attr.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)
                        and definition.GetAttributeDefinition(attr.GetName())):
                    asset_attribute(attr, kind='schema_asset')

        visited_properties, visited_shaders = set(), set()
        def visit_property(prop):
            if not prop or not isinstance(prop, Usd.Attribute) or prop.GetPath() in visited_properties:
                return
            visited_properties.add(prop.GetPath())
            graph_paths.add(str(prop.GetPrim().GetPath()))
            connections = prop.GetConnections()
            if connections:
                for target in connections:
                    source = stage.GetPropertyAtPath(target)
                    if source:
                        visit_property(source)
                    else:
                        add(warnings, 'dangling_shader_connection', str(target), str(prop.GetPrim().GetPath()))
                return
            if prop.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                asset_attribute(prop)
            prim = prop.GetPrim()
            if prim.IsA(UsdShade.Shader) and prim.GetPath() not in visited_shaders:
                visited_shaders.add(prim.GetPath())
                for attr in prim.GetAttributes():
                    name = attr.GetName()
                    if name.startswith('inputs:') or name in ('info:mdl:sourceAsset', 'info:sourceAsset'):
                        visit_property(attr)

        for path, material in materials.items():
            graph_paths.add(path)
            for terminal in ('surface', 'displacement', 'volume'):
                outputs = [material.GetPrim().GetAttribute('outputs:mdl:' + terminal),
                           material.GetPrim().GetAttribute('outputs:' + terminal)]
                selected = next((out for out in outputs if out and any(stage.GetPropertyAtPath(p)
                                    for p in out.GetConnections())), None)
                if selected:
                    visit_property(selected)
                elif terminal == 'surface':
                    add(warnings, 'builtin_or_default_material', path, path)

        relevant = {str(p.GetPath()) for p in prims if geometry(p)} | graph_paths
        used_layers = {spec.layer.identifier for prim in prims if str(prim.GetPath()) in relevant
                       for spec in prim.GetPrimStack()}
        for layer in stage.GetUsedLayers():
            if layer.identifier in used_layers:
                add(used, 'composition_layer', layer.identifier, resolved=layer.realPath or layer.identifier)
        if consumers:
            used_layers.add(stage.GetRootLayer().identifier)
        # Missing sublayers are layer-stack errors, not necessarily prim-index
        # errors (including in Isaac's USD 24.5). Inspect only composed layers;
        # inactive variants and muted layers never enter this check.
        muted = set(stage.GetMutedLayers())
        for layer in stage.GetUsedLayers():
            for raw in layer.subLayerPaths:
                anchored = Sdf.ComputeAssetPathRelativeToLayer(layer, raw)
                if raw in muted or anchored in muted:
                    continue
                resolved = str(resolver.Resolve(_file_path(anchored)))
                try:
                    loaded = Sdf.Layer.FindOrOpen(anchored) if resolved else None
                except Exception:
                    loaded = None
                if loaded is None:
                    affects = layer.identifier in used_layers
                    add(errors if affects else warnings, 'missing_sublayer' if affects else 'unused_sublayer',
                        raw, layer=layer.identifier)
        seen_errors = set()
        for prim in [stage.GetPseudoRoot(), *prims]:
            for error in prim.GetPrimIndex().localErrors:
                text = str(error)
                if text in seen_errors:
                    continue
                seen_errors.add(text)
                kind = type(error).__name__
                if kind not in ('ErrorInvalidAssetPath', 'ErrorInvalidSublayerPath', 'ErrorUnresolvedPrimPath'):
                    continue
                paths = re.findall(r'@([^@]+)@', text)
                missing = paths[0] if paths else text
                if kind == 'ErrorUnresolvedPrimPath':
                    target = re.search(r'@[^@]+@(<[^>]+>)', text)
                    if target:
                        missing += target[1]
                path = str(prim.GetPath())
                affects = ((prim.IsPseudoRoot() and bool(consumers)) or path in relevant
                           or any(p.startswith(path.rstrip('/') + '/') for p in relevant))
                add(errors if affects else warnings, 'unresolved_composition' if affects else 'unused_composition',
                    missing, path, detail=text)

        # Dangling bindings and unconnected material assets are observations,
        # including shared X5 viewport light-rig references. Never promote them.
        for prim in prims:
            for rel in prim.GetRelationships():
                if rel.GetName().startswith('material:binding'):
                    for target in rel.GetTargets():
                        if not stage.GetPrimAtPath(target):
                            add(warnings, 'dangling_material_binding', str(target), str(prim.GetPath()))
            if (prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.Material)) and str(prim.GetPath()) not in graph_paths:
                for attr in prim.GetAttributes():
                    if attr.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                        value = attr.Get(Usd.TimeCode(0))
                        values = value if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray else [value]
                        for asset in values or []:
                            if isinstance(asset, Sdf.AssetPath) and asset.path:
                                add(warnings, 'unused_shader_asset', asset.path, str(prim.GetPath()))

    return dict(schema='mind2act_eval.asset_resolution.v1', status='invalid' if errors else 'passed',
        invalid_reason='asset_unresolved' if errors else None,
        stage=stage.GetRootLayer().identifier, used=used, errors=errors, warnings=warnings,
        missing_paths=sorted({row['path'] for row in errors}), mdl_checks=mdl_checks,
        elapsed_s=time.monotonic()-started,
        qualification='used resource resolution and native MDL module compilation; no appearance, physics or score comparison')


def check_stage(stage, output_dir, *, mdl_search_paths=None, mdl_cache=None):
    report = inspect_stage(stage, mdl_search_paths=mdl_search_paths, mdl_cache=mdl_cache)
    path = Path(output_dir) / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    if report['errors']:
        raise AssetUnresolved(report)
    from pxr import Ar
    from ..release_contract import reject_deprecated_paths
    paths = set()
    for row in report['used']:
        path = _file_path(row.get('resolved', ''))
        if Ar.IsPackageRelativePath(path):
            path = Ar.SplitPackageRelativePathOuter(path)[0]
        # Anonymous layers and virtual SDK modules are not filesystem paths.
        # Warnings and unused authored resources never enter this list.
        if path and not urlsplit(path).scheme:
            paths.add(path)
    reject_deprecated_paths(paths)  # Keep ReleaseContractError distinct.
    return report


class WorldAssetGate:
    """Guard reset (which can step physics) and direct World.step call sites."""
    def __init__(self, world, stage, output_dir, *, before_physics=None):
        from pxr import Tf, Usd
        self.stage, self.output_dir = stage, output_dir
        self.dirty, self.report = True, None
        self.mdl_cache = {}
        self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.changed, stage)
        for name in ('reset', 'step'):
            original = getattr(world, name)
            method = (weakref.WeakMethod(original) if getattr(original, '__self__', None) is world
                      else lambda _function=original: _function)
            def guarded(*args, _method=method, _name=name, **kwargs):
                self.check(force=_name == 'reset')
                if before_physics is not None:
                    before_physics(_name)
                return _method()(*args, **kwargs)
            setattr(world, name, guarded)

    def changed(self, notice, _sender):
        from pxr import Sdf, Usd
        if notice.GetResyncedPaths():
            self.dirty = True
            self.mdl_cache.clear()
        for path in notice.GetChangedInfoOnlyPaths():
            if path.name in ('visibility', 'purpose', 'physics:collisionEnabled', 'physics:rigidBodyEnabled'):
                self.dirty = True
                continue
            if path.name.startswith(('xformOp:', 'physics:', 'physx', 'drive:')) or path.name in (
                    'points', 'velocities', 'accelerations', 'normals', 'extent', 'xformOpOrder'):
                continue  # Ordinary physics updates do not re-run an asset audit.
            prop = self.stage.GetPropertyAtPath(path)
            if (isinstance(prop, Usd.Attribute) and prop.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)
                    or 'material:binding' in str(path) or path.name.startswith(('collection:', 'info:'))
                    or {'connectionPaths', 'targetPaths'}.intersection(notice.GetChangedFields(path))):
                self.dirty = True
                self.mdl_cache.clear()

    def check(self, *, force=False):
        if self.dirty or force:
            if force:
                self.mdl_cache.clear()
            self.report = check_stage(self.stage, self.output_dir, mdl_cache=self.mdl_cache)
            self.dirty = False
        return self.report

    def close(self):
        self.notice.Revoke()
