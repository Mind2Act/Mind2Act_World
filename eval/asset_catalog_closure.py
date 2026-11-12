"""CPU producer for a complete digest-bound USD/layer/texture dependency closure.

Native admission only hashes this manifest and its files; it does not need pxr
before Kit startup. The producer requires a standalone USD Python environment.
"""
from pathlib import Path

from .instance_catalog import relative_path, sha256
from .release_contract import ReleaseContractError


def build_asset_closure(root, assets):
    from pxr import Ar, Sdf, UsdUtils
    root = Path(root).resolve()
    files, usd_roots, search_roots = set(), set(), []
    usd_suffixes = {'.usd', '.usda', '.usdc', '.usdz'}

    def add(path):
        value = str(path)
        if Ar.IsPackageRelativePath(value):
            value, _ = Ar.SplitPackageRelativePathOuter(value)
        path = Path(value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ReleaseContractError('USD asset closure escapes release or is missing: ' + str(path))
        files.add(path)

    for relative in assets.values():
        path = relative_path(root, relative)
        members = sorted(path.rglob('*')) if path.is_dir() else [path]
        if path.is_dir():
            search_roots.append(path)
        for member in members:
            relative_path(root, str(member.relative_to(root)))
            if member.is_symlink() and member.is_dir():
                raise ReleaseContractError('asset closure requires expanded directory symlinks')
            if member.is_file():
                add(member)
                if member.suffix.lower() in usd_suffixes:
                    usd_roots.add(member)
    for path in sorted(usd_roots):
        layers, dependencies, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
        for layer in layers:
            add(layer.realPath or layer.identifier)
        for dependency in dependencies:
            add(dependency)
        for value in unresolved:
            # Standard MDL names are looked up only in the spec's explicit
            # bundled search directories. No network or machine-wide fallback.
            if Path(value).suffix.lower() != '.mdl' or Path(value).is_absolute():
                raise ReleaseContractError('unresolved USD dependency: ' + value)
            candidates = {str((directory / value).resolve()) for directory in search_roots
                          if (directory / value).is_file()}
            if len(candidates) != 1:
                raise ReleaseContractError('unresolved or ambiguous bundled MDL dependency: ' + value)
            add(candidates.pop())
    return dict(schema='mind2act_eval.asset_closure.v1', root_assets=dict(assets),
                producer='usd-utils-recursive-v1', unresolved=[],
                files=[dict(path=str(path.relative_to(root)), sha256=sha256(path))
                       for path in sorted(files)])
