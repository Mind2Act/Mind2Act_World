"""Open the portable station delivered with the instance."""
from .source_assets import source_file


def prepare_open_station(spec, assets, output):
    """Delivered stations already contain their resolved relative asset references."""
    return source_file(assets, 'station')
