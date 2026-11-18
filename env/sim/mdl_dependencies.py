"""Inspect transitive MDL imports/resources and optionally invoke NVIDIA MDLC.

The compiler command is explicit. Static resolution never claims compilation,
and neither operation starts Kit, a renderer, or a GPU context.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import re
import subprocess
import zipfile

from .material_closure import MDL_BUILTINS, _mdl_text


def _read(identifier):
    if '[' not in identifier:
        return Path(identifier).read_bytes()
    from pxr import Ar
    outer, member = Ar.SplitPackageRelativePathOuter(identifier)

    def read_member(archive, name):
        if not Ar.IsPackageRelativePath(name):
            return archive.read(name)
        nested, child = Ar.SplitPackageRelativePathOuter(name)
        with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as inner:
            return read_member(inner, child)

    with zipfile.ZipFile(outer) as archive:
        return read_member(archive, member)


def _relative(identifier, name):
    if '[' not in identifier:
        return str(Path(identifier).parent / name)
    from pxr import Ar
    from posixpath import dirname, join, normpath
    outer, member = Ar.SplitPackageRelativePathInner(identifier)
    return Ar.JoinPackageRelativePath(outer, normpath(join(dirname(member), name)))


def _resolve(identifier):
    if '[' not in identifier:
        path = Path(identifier)
        return str(path.resolve()) if path.is_file() else ''
    from pxr import Ar
    return str(Ar.GetResolver().Resolve(identifier))


def audit_mdl(path, search_paths=()):
    """Follow module/entity imports, including parent packages and USDZ files."""
    search_paths = tuple(map(Path, search_paths))
    modules, references, issues = [], [], []
    visited = set()

    def visit(identifier):
        if identifier in visited:
            return
        visited.add(identifier)
        try:
            text = _mdl_text(_read(identifier).decode('utf-8'))
        except (OSError, ValueError, UnicodeError, KeyError, zipfile.BadZipFile) as exc:
            issues.append(dict(kind='unreadable_mdl_module', path=identifier, detail=str(exc)))
            return
        modules.append(identifier)
        # Quoted annotation text is not an import statement.
        statements = re.sub(r'"(?:\\.|[^"\\])*"', ' ', text)
        imports = re.findall(r'\busing\s+(\S+)\s+import\s+[^;]*;|\bimport\s+([^;]+);', statements)
        for using, direct in imports:
            for token in (using or direct).split(','):
                token = token.strip()
                parts = token.strip(':').split('::')
                if parts[0] in MDL_BUILTINS:
                    continue
                resolved = ''
                for count in range(len(parts), 0, -1):
                    name = '/'.join(parts[:count]) + '.mdl'
                    candidates = [str(root / name) for root in search_paths]
                    if not token.startswith('::'):
                        candidates.insert(0, _relative(identifier, name))
                    resolved = next((value for p in candidates if (value := _resolve(p))), '')
                    if resolved:
                        break
                references.append(dict(layer=identifier, kind='mdl_import', path=token, resolved=resolved))
                if resolved:
                    visit(resolved)
                else:
                    issues.append(dict(kind='unresolved_mdl_import', layer=identifier, path=token))
        resources = re.findall(r'\b(?:texture_2d|texture_3d|light_profile|bsdf_measurement)\s*\(\s*"([^"\n]+)"', text)
        for name in resources:
            resolved = _resolve(_relative(identifier, name))
            references.append(dict(layer=identifier, kind='mdl_resource', path=name, resolved=resolved))
            if Path(name).is_absolute() or re.match(r'\w+://', name):
                issues.append(dict(kind='nonportable_mdl_resource', layer=identifier, path=name))
            if not resolved:
                issues.append(dict(kind='unresolved_mdl_resource', layer=identifier, path=name))

    visit(str(path))
    return dict(passed=not issues, modules=modules, references=references, issues=issues,
                compilation='not_run')


def compile_mdlc(path, search_paths, executable, *, timeout_s=60):
    """Compile/type-check an actual module with the explicitly supplied MDLC."""
    path = Path(path).resolve()
    roots = tuple(Path(p).resolve() for p in search_paths)
    relative = next((path.relative_to(root) for root in roots if path.is_relative_to(root)), None)
    if relative is None:
        raise ValueError('MDL compiler requires an explicit module package search root')
    module = '::' + '::'.join(relative.with_suffix('').parts)
    # NVIDIA's exported materials use scalar-to-vector implicit casts. MDLC's
    # strict subset rejects those even though the MDL SDK supports them; use
    # SDK-compatible language mode while retaining full import/type/resource
    # validation. No material code is rewritten to satisfy this checker.
    argv = [str(executable), '--no-strict', '--resolve-resources']
    for root in roots:
        argv.extend(['--path', str(root)])
    argv.append(module)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
        return dict(passed=result.returncode == 0, module=module, argv=argv,
                    exit_code=result.returncode, diagnostics=(result.stdout + result.stderr)[-16000:])
    except subprocess.TimeoutExpired:
        return dict(passed=False, module=module, argv=argv, exit_code=None,
                    diagnostics='MDL compiler timeout')
