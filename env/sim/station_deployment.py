"""Open a portable material projection while retaining exact task source assets."""
import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_open_station(spec, assets, output):
    source = assets.resolve('station')
    source_manifest = spec.config.get('station_relocation_manifest')
    deployment_key = 'station_material_relocation_manifest'
    has_deployment = deployment_key in spec.config
    selected = spec.config[deployment_key] if has_deployment else source_manifest
    if has_deployment or 'station_relocation_manifest' in spec.config:
        if not isinstance(selected, str) or not selected:
            raise ValueError('station relocation must name a declared asset ID')
    if selected is None:
        return source
    approval = None
    if has_deployment:
        from .source_assets import authorize_station_deployment
        approval = authorize_station_deployment(spec, assets, selected)
    from .asset_relocation import prepare_station
    derived = prepare_station(source, assets, selected, output)
    from .source_assets import write_source_projection_receipt
    write_source_projection_receipt(assets, output)
    if has_deployment:
        # Task validators continue to resolve their unchanged original assets.
        # The shared relocation code checks source/target bytes and proves that
        # all non-asset opinions are identical in the opened layer graph.
        receipt = dict(schema='mind2act_eval.station_material_deployment.v1',
            source_station_path=str(source), source_station_sha256=_sha(source),
            source_manifest_asset_id=source_manifest,
            source_manifest_sha256=_sha(assets.resolve(source_manifest)) if source_manifest else None,
            material_manifest_asset_id=selected,
            material_manifest_sha256=_sha(assets.resolve(selected)),
            runtime_station_path=str(derived), runtime_station_sha256=_sha(derived),
            source_asset_bindings_preserved=True, code_owned_approval=approval)
        (Path(output) / 'station-material-deployment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return derived
