"""Generated-texture helpers. Fonts come from the asset resolver only: a missing font is an asset error,
never a different default font (same instance must render the same pixels on every host)."""
from pathlib import Path

FONT_ASSETS = ('font_sans', 'font_sans_bold')  # public asset ids a spec must declare in InstanceSpec.assets


class AssetError(RuntimeError):
    """A required asset is not declared or not present; recorded in the asset-gap list, never worked around."""


def font_loader(assets, spec):
    """Return load(bold, size) bound to the fonts declared by the spec and resolved by `assets`."""
    from PIL import ImageFont
    paths = {}
    for bold, asset_id in ((False, 'font_sans'), (True, 'font_sans_bold')):
        if asset_id not in spec.assets:
            raise AssetError(f'{spec.task_id}: spec does not declare asset {asset_id}')
        try:
            paths[bold] = Path(assets.resolve(asset_id))
        except (KeyError, FileNotFoundError, OSError) as exc:
            raise AssetError(f'{spec.task_id}: asset {asset_id} ({spec.assets[asset_id]}) is missing') from exc

    def load(bold: bool, size: int):
        return ImageFont.truetype(str(paths[bold]), size)
    return load


def save_verified(image, path: Path) -> Path:
    """Write a PNG and decode it back, so a missing/corrupt texture fails at build time."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    with Image.open(path) as check:
        check.verify()
    return path
