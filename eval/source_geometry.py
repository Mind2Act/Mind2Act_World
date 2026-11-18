"""CPU comparisons of complete authored USD opinions and asset byte identities."""
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath
import zipfile
from functools import lru_cache


def finite_value(value, shape=()):
    """Check numeric dimensions and finiteness before comparison or masking."""
    message = 'source geometry requires finite numeric values of the expected shape'
    try:
        if shape:
            if isinstance(value, (str, bytes)) or len(value) != shape[0]:
                raise ValueError(message)
            for component in value:
                finite_value(component, shape[1:])
        elif isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(message)
    except (TypeError, OverflowError) as error:
        raise ValueError(message) from error
    return value


def finite_default(data, path, shape=()):
    try:
        value = data[path]['default']
    except (KeyError, TypeError) as error:
        raise ValueError('source geometry numeric default is missing or malformed') from error
    return finite_value(value, shape)


def asset_identity(value, layer_path, asset_root=None):
    """Bind an image/MDL file or a member of a byte-bound USDZ package."""
    token = value.path
    if not token:
        return None, None
    outer, separator, inner = token.partition('[')
    if separator:
        if not inner.endswith(']') or '[' in inner:
            raise ValueError('source geometry has an unsupported package asset')
        inner = inner[:-1]
        member = PurePosixPath(inner)
        if member.is_absolute() or '..' in member.parts or '\\' in inner:
            raise ValueError('source geometry package member escapes its archive')
    else:
        inner = None
    root = Path(asset_root).resolve(strict=True) if asset_root is not None else None
    path = Path(layer_path).parent/outer
    if not path.exists() and path.suffix == '.mdl' and root is not None:
        from .original_asset_graph import _dependency
        path = _dependency(Path(layer_path).parent, outer,
            [root/p for p in ('sdk/mdl/core/Base', 'sdk/mdl/core/mdl', 'sdk/mdl/rtx')])
    path = path.resolve(strict=True)
    if root is not None and not path.is_relative_to(root):
        raise ValueError('source geometry asset escapes its root')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if inner is not None:
        with zipfile.ZipFile(path) as archive:
            archive.getinfo(inner)
    return sha, inner


def opinions(layer, prefix=None):
    from pxr import Sdf
    prefix = Sdf.Path(prefix) if prefix is not None else None
    result = {}
    def visit(path):
        if prefix is not None and not path.HasPrefix(prefix):
            return
        item = layer.GetObjectAtPath(path)
        if item is not None:
            result[str(path)] = {key: (
                [(o.offset, o.scale) for o in layer.subLayerOffsets]
                if key == 'subLayerOffsets' else item.GetInfo(key)
            ) for key in item.ListInfoKeys()}
    layer.Traverse('/', visit)
    return result


def fingerprint(data, layer_path, asset_root=None, aliases=None, asset_markers=None):
    """Keep every field; normalize only names/asset URIs with byte identity.

    Task code validates source-derived dynamic defaults before replacing them
    with markers. No producer-selected mask or expected digest is accepted.
    """
    from pxr import Sdf
    aliases = sorted((aliases or {}).items(), key=lambda item: -len(item[0]))
    root = Path(asset_root).resolve(strict=True) if asset_root is not None else None
    def path_name(value):
        value = Sdf.Path(value)
        for old, new in aliases:
            if value.HasPrefix(old):
                return str(value.ReplacePrefix(old, new))
        return str(value)
    def normalize(value):
        if isinstance(value, Sdf.Path):
            return path_name(value)
        if isinstance(value, Sdf.AssetPath):
            sha, member = asset_identity(value, layer_path, root)
            return ['asset-sha256', (asset_markers or {}).get(sha, sha), member]
        if isinstance(value, float) and value == 0:
            return 0.0  # Signed zero has the same USD physical value.
        if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
            return ['quaternion', normalize(value.GetReal()), normalize(value.GetImaginary())]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if hasattr(value, 'isExplicit'):
            return {key: normalize(getattr(value, key)) for key in
                ('isExplicit', 'explicitItems', 'addedItems', 'prependedItems',
                 'appendedItems', 'deletedItems', 'orderedItems')}
        try:
            return [normalize(v) for v in value]
        except TypeError:
            return str(value)
    value = {path_name(key): normalize(fields) for key, fields in data.items()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def expected_wrapper(layer, *, standard_clock=False):
    from pxr import Sdf
    expected = Sdf.Layer.CreateAnonymous('source-composition.usda')
    expected.defaultPrim = 'World'
    expected.pseudoRoot.SetInfo('metersPerUnit', 1.)
    expected.pseudoRoot.SetInfo('upAxis', 'Z')
    if standard_clock:
        expected.startTimeCode = 0
        expected.endTimeCode = 1000000
        expected.timeCodesPerSecond = 60
    expected.subLayerPaths = list(layer.subLayerPaths)
    return expected


@lru_cache(maxsize=32)
def retained_namespace(base, code_owned_sha, roots):
    from pxr import Usd
    stage = Usd.Stage.Open(base)
    return tuple(tuple(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root)))
        for root in roots if stage.GetPrimAtPath(root))


def validate_composition(spec, assets, proof, base_sha, graph_sha, roots, *, skip_retired_children=False):
    from pxr import Sdf
    from .original_asset_graph import layer_bytes
    root = Path(assets.root).resolve(strict=True)
    if any(k in spec.config for k in ('station_relocation_manifest', 'station_material_relocation_manifest')):
        raise ValueError('original test must open the authenticated derived station')
    def resolve(name):
        relative = Path(proof[name]['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('original composition descriptor escapes its root')
        path = (root/relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('original composition resource escapes its root')
        return path
    scene, task, base = (resolve(k) for k in ('compiled_scene', 'actual_composed_task', 'common_station'))
    if (scene != Path(assets.resolve('station')).resolve(strict=True)
            or task != Path(assets.resolve('task_source')).resolve(strict=True)
            or proof['common_station']['sha256'] != base_sha):
        raise ValueError('original composition actual role or retained pool differs')

    _, layer = layer_bytes(scene)
    _, task_layer = layer_bytes(task)
    if len(layer.subLayerPaths) != 2 or [(scene.parent/p).resolve(strict=True) for p in layer.subLayerPaths] != [task, base]:
        raise ValueError('original composition requires the exact task and code-owned retained pool')
    expected = expected_wrapper(layer)
    for paths in retained_namespace(str(base), base_sha, tuple(roots)):
        retired = []
        for value in paths:
            path = Sdf.Path(value)
            if skip_retired_children and any(path.HasPrefix(p) for p in retired):
                continue
            if not task_layer.GetPrimAtPath(path):
                Sdf.CreatePrimInLayer(expected, path).active = False
                retired.append(path)
    if opinions(layer) != opinions(expected):
        raise ValueError('original composition contains unauthorized physical or stage opinions')
    return task, task_layer
