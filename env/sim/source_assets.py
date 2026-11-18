"""Runtime access to the assets declared by an instance.

The resolver confines roles to the supplied asset root. Task loaders check their
parameters; historical byte/projection audits live in asset_provenance.
"""
from pathlib import Path


def source_file(assets, asset_id):
    """Resolve a required file without an export-byte whitelist."""
    path = Path(assets.resolve(asset_id))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def asset_bytes(assets, asset_id):
    return source_file(assets, asset_id).read_bytes()


def asset_layer(assets, asset_id):
    """Open a separate layer so callers cannot modify the delivered source."""
    from pxr import Sdf
    layer = Sdf.Layer.OpenAsAnonymous(str(source_file(assets, asset_id)))
    if layer is None:
        raise ValueError('asset is not a readable USD layer: ' + asset_id)
    return layer
