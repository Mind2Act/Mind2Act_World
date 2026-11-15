"""Bind symbolic MDL references to the variation's checked release files."""
from pathlib import Path
import re

from . import file_digest


def pinned_module(pin, raw):
    # Absolute/URI/parent-relative references cannot acquire a new capability.
    if (not raw.endswith('.mdl') or raw.startswith(('.', '/'))
            or re.match(r'^[A-Za-z][\w+.-]*:', raw)
            or '..' in Path(raw).parts or '\\' in raw):
        return None
    declared = {p.resolve(): sha for p, sha in pin.dependencies}
    for root in pin.search_paths:
        target = (root / raw).resolve()
        if target in declared:
            if file_digest(target) != declared[target]:
                raise ValueError('variation MDL dependency changed: ' + str(target))
            return target
    return None


def sdk_aliases(pin, layers, actual):
    """Map only authored symbolic tokens, with a pinned target and equal SHA.

    A matching digest alone never authorizes an absolute authored reference.
    These aliases are immediately replaced by release-relative shader opinions
    in localize_bindings; SDK paths are never authored into the runtime layer.
    """
    from pxr import Ar, Sdf, UsdUtils
    aliases, roots, explicit = {}, set(), set()
    for layer in layers:
        tokens = set()
        copy = Sdf.Layer.CreateAnonymous('mdl-token-audit.usda')
        copy.TransferContent(layer)
        UsdUtils.ModifyAssetPaths(copy, lambda raw: tokens.add(raw) or raw)
        for raw in tokens:
            target = pinned_module(pin, raw)
            if target is None:
                if raw.endswith('.mdl'):
                    value = str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(layer, raw)))
                    if value:
                        explicit.add(Path(value.split('[', 1)[0]).resolve())
                continue
            roots.add(target)
            for value in actual:
                outer = Path(value.split('[', 1)[0]).resolve()
                if outer == target or not str(outer).endswith('/' + raw):
                    continue
                if outer.is_file() and file_digest(outer) == file_digest(target):
                    aliases[outer] = target
    # USD dependency extraction does not inspect MDL imports/string textures.
    from mind2act_eval.sim.mdl_dependencies import audit_mdl
    declared = {pin.path.resolve(), *(p.resolve() for p, _ in pin.dependencies)}
    for root in roots:
        report = audit_mdl(root, pin.search_paths)
        if not report['passed']:
            raise ValueError('incomplete pinned MDL closure: ' + str(report['issues']))
        for value in [*report['modules'], *(r['resolved'] for r in report['references'])]:
            if Path(value.split('[', 1)[0]).resolve() not in declared:
                raise ValueError('unpinned recursive MDL dependency: ' + value)
    return {source: target for source, target in aliases.items() if source not in explicit}


def localize_bindings(prim, pin, layer):
    """Author explicit release paths over imported symbolic MDL opinions."""
    from pxr import Sdf, Usd
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        for attr in child.GetAttributes():
            scalar = attr.GetTypeName() == Sdf.ValueTypeNames.Asset
            if not scalar and attr.GetTypeName() != Sdf.ValueTypeNames.AssetArray:
                continue
            original = attr.Get()
            values = [original] if scalar else list(original or [])
            changed = False
            for index, value in enumerate(values):
                if not isinstance(value, Sdf.AssetPath):
                    continue
                target = pinned_module(pin, value.path)
                if target is not None:
                    values[index] = Sdf.AssetPath(relative_asset_path(layer, target))
                    changed = True
            if changed and not attr.Set(values[0] if scalar else values):
                raise ValueError('cannot bind pinned MDL: ' + str(attr.GetPath()))
