"""Authenticate a retained USD pool and its transitive file dependencies.

Pins live in task code. A caller receipt or catalog cannot supply this authority.
"""
import hashlib
import json
import os
from pathlib import Path

from mind2act_eval.verified_files import verify_sha256

_graphs = {}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def layer_bytes(path):
    """Read a fresh anonymous layer, checking the file around USD's parser."""
    from pxr import Usd, Sdf
    path = Path(path)
    before = path.read_bytes()
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    if not layer or path.read_bytes() != before:
        raise ValueError('original asset layer changed or cannot be parsed')
    return before, layer


def dependencies(layer):
    from pxr import Sdf
    result = set(layer.subLayerPaths)
    def collect(value):
        if isinstance(value, Sdf.AssetPath):
            if value.path:
                result.add(value.path)
        elif isinstance(value, (Sdf.Reference, Sdf.Payload)):
            if value.assetPath:
                result.add(value.assetPath)
        elif hasattr(value, 'GetAddedOrExplicitItems'):
            for item in value.GetAddedOrExplicitItems():
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)) or type(value).__name__ == 'AssetPathArray':
            for item in value:
                collect(item)
    def visit(path):
        item = layer.GetObjectAtPath(path)
        if item is not None:
            for key in item.ListInfoKeys():
                collect(item.GetInfo(key))
    layer.Traverse('/', visit)
    return sorted(result)


def _shader_members(root):
    members = []
    def walk(folder):
        with os.scandir(folder) as stream:
            entries = sorted(stream, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError('original shader inventory requires regular files and directories')
            if entry.is_dir(follow_symlinks=False):
                walk(path)
            elif entry.is_file(follow_symlinks=False):
                members.append(str(path.relative_to(root)))
            else:
                raise ValueError('original shader inventory contains an unsupported member')
    try:
        for folder in ('sdk/mdl/core/Base', 'sdk/mdl/core/mdl', 'sdk/mdl/rtx'):
            path = root/folder
            if path.is_symlink():
                raise ValueError('original shader inventory cannot alias its search directory')
            walk(path)
    except OSError as error:
        # pathlib.rglob silently suppresses some enumeration errors. A closed
        # inventory must reject unreadable/new subtrees, including warm reads.
        raise ValueError('original shader inventory cannot be completely enumerated') from error
    return tuple(sorted(members))


def _dependency(parent, token, shader_roots):
    target = Path(token)
    if target.is_absolute() or ':' in token or '[' in token:
        raise ValueError('original asset graph requires local relative dependencies')
    dependency = parent/target
    if not dependency.exists() and target.suffix == '.mdl':
        matches = [folder/target for folder in shader_roots if (folder/target).is_file()]
        if len(matches) != 1:
            raise ValueError('original shader dependency is absent or ambiguous')
        dependency = matches[0]
    return dependency.resolve(strict=True)


def graph_digest(path, root, *, _files=None, _resolutions=None):
    """Bind bytes and resolved edges; relocation of the complete root is safe."""
    root = Path(root).resolve(strict=True)
    # The delivered renderer resolves bare MDL names through these existing
    # search roots. Bind every member, including transitive MDL imports, rather
    # than treating an unresolved shader token as an arbitrary external URI.
    shader_roots = [root/p for p in ('sdk/mdl/core/Base', 'sdk/mdl/core/mdl', 'sdk/mdl/rtx')]
    shader_files = []
    # Preserve the original Path-parts ordering in the signed graph payload;
    # lexical strings order e.g. a.mdl before a/file.mdl differently.
    for relative in sorted(_shader_members(root), key=Path):
        member = root/relative
        if not member.resolve(strict=True).is_relative_to(root):
            raise ValueError('original shader dependency escapes root')
        shader_files.append([relative, _sha(member.read_bytes())])
        if _files is not None:
            _files[str(member)] = shader_files[-1][1]
        if _resolutions is not None:
            _resolutions[(str(member.parent), member.name)] = str(member.resolve(strict=True))
    cache, visiting = {}, set()
    def visit(path):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('original asset graph escapes its root')
        if path in visiting:
            raise ValueError('original asset graph contains a dependency cycle')
        if path in cache:
            return cache[path]
        visiting.add(path)
        raw = path.read_bytes()
        if _files is not None:
            _files[str(path)] = _sha(raw)
        edges = []
        if path.suffix.lower() in ('.usd', '.usda', '.usdc'):
            checked, layer = layer_bytes(path)
            if raw != checked:
                raise ValueError('original asset graph changed while reading')
            for token in dependencies(layer):
                dependency = _dependency(path.parent, token, shader_roots)
                if _resolutions is not None:
                    _resolutions[(str(path.parent), token)] = str(dependency)
                edges.append([token, visit(dependency)])
        if path.read_bytes() != raw:
            raise ValueError('original asset graph changed while reading')
        digest = _sha(json.dumps([_sha(raw), edges], separators=(',', ':')).encode())
        visiting.remove(path)
        cache[path] = digest
        return digest
    return _sha(json.dumps([visit(path), shader_files], separators=(',', ':')).encode())


def verify_graph(path, root, expected):
    root = Path(root).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    key = (str(root), str(path), expected)
    if key not in _graphs:
        files, resolutions = {}, {}
        if graph_digest(path, root, _files=files, _resolutions=resolutions) != expected:
            raise ValueError('original retained pool transitive asset authority differs')
        # Authority is frozen only after the complete graph matches task code.
        _graphs[key] = (tuple(sorted(files.items())), _shader_members(root),
                       tuple(sorted(resolutions.items())))
    files, shaders, resolutions = _graphs[key]
    shader_roots = [root/p for p in ('sdk/mdl/core/Base', 'sdk/mdl/core/mdl', 'sdk/mdl/rtx')]
    def check_resolutions():
        # Re-resolve each logical USD/MDL edge, including search fallback, on
        # every public load. Canonical file freshness alone misses retargeting.
        for (parent, token), previous in resolutions:
            actual = _dependency(Path(parent), token, shader_roots)
            if actual != Path(previous) or not actual.is_relative_to(root):
                raise ValueError('original retained pool edge resolution changed')
    check_resolutions()
    if _shader_members(root) != shaders:
        raise ValueError('original retained pool transitive asset authority differs')
    for name, sha in files:
        member = Path(name).resolve(strict=True)
        if not member.is_relative_to(root):
            raise ValueError('original retained pool dependency escapes root')
        try:
            # Reuse the shared inode/freshness/write guards; never cache an
            # unauthenticated parse or grant authority from stat identity.
            verify_sha256(member, sha)
        except ValueError as error:
            raise ValueError('original retained pool transitive asset authority differs') from error
    check_resolutions()
