"""One acquisition pad geometry asset shared by every CP09 case.

The JSON is the exact source pad dimensions/materials, independently retained
in tests/fixtures/cp09_source_pads.usda. Neither source selection nor case
admission belongs here. Official X5 drives and asset verification remain with
X5Robot; validate_actual checks the composed physical pads before restoration.
"""
import hashlib
from importlib import resources
import json

ASSET_NAME='cp09_acquisition_pads.json'
ASSET_SHA256='5905fad98d436142fb1454e36ab44fe33daee17ee8fcd5d07277590a26d56c01'
ACQUISITION_VERSION='cp09-acquisition-geometry-v1'


def descriptor():
    raw=resources.files(__package__).joinpath(ASSET_NAME).read_bytes()
    value=json.loads(raw)
    if (value['schema']!='mind2act.cp09-acquisition-pad-geometry.v1'
            or value['acquisition_version']!=ACQUISITION_VERSION):
        raise ValueError('CP09 acquisition pad geometry version differs')
    geometry=json.dumps(value['pads'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(task_id='CP09',acquisition_version=ACQUISITION_VERSION,
                source_sha256=ASSET_SHA256,geometry_asset_sha256=ASSET_SHA256,
                source_geometry_asset_sha256=value['source_geometry_asset_sha256'],
                geometry_sha256=hashlib.sha256(geometry).hexdigest(),pads=value['pads'])
