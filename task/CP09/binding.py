"""Fail closed until the independent asset owner pins the station projection."""
import hashlib,json

# Exact, already-portable task-stripped acquisition stations in c2ec5b07.
# Their proofs retain the training origin of the station resource; original
# test case identity and drawing geometry are authenticated separately.
PORTABLE_STATION = {
    'easy': ('7afb4488dded1bf5f9e9bdc1b499cfd4092042677705869c00ed0d7b86fe2948',
             'ff74789b0e7a2c017031a2f8b69ff2354e24352b5424708696322e7f9efd4332'),
    'medium': ('10fb1146b9303c3d426b2c0ec1360aa12bf158a163e0286695833580788d94f7',
               '86ec97d0fd7225b5b40eedb7d26c5cb776bf8eae8003433ccf568a810a612880'),
    'hard': ('30a42cef57b4751d846c50157d2bdebf5955776d7e06d4d08e70afaec3b88e64',
             'a215d0137b4a3ae453370e9b821f2296a533a7026d2fca207a9b54f88fc5c4ab'),
}


def validate_station(row,assets,station_config_sha256,*,spec=None):
    if spec is not None and ('train_source' in spec.config or spec.config.get('layout_split')=='validation'):
        from mind2act_eval.cp09_training_inputs import validate_assets as training_validate_assets
        from mind2act_eval.sim.source_assets import asset_bytes, asset_layer, source_file
        validation=spec.config.get('layout_split')=='validation'
        if validation:
            from mind2act_eval.cp09_validation_inputs import validate_assets
            validate_assets(spec,assets)
        else:
            training_validate_assets(spec,assets)
        source_file(assets, 'station')
        # Own Task is a CPU-authored runtime dependency, never an old selected Task.
        task=asset_layer(assets, 'validation_task' if validation else 'training_task')
        if task.GetPrimAtPath('/World/CP09') is None:
            raise ValueError('CP09 own authored Task root missing')
        return None
    test=spec is not None and 'test_source' in spec.config
    expected_scene=row.get('station_sha256');expected_proof=row.get('station_proof_sha256')
    expected_relocation=row.get('station_relocation_sha256')
    if test:
        from .instances import original_case
        original_case(spec,assets)
        expected_scene,expected_proof=PORTABLE_STATION[spec.difficulty]
    if not expected_scene or not expected_proof or not expected_relocation:
        raise RuntimeError('CP09 awaiting authoritative selected station/proof SHA256 anchors')
    from mind2act_eval.sim.source_assets import asset_bytes, source_file
    try:
        source_file(assets, 'station')
        raw=asset_bytes(assets, 'station_proof')
        if not test:asset_bytes(assets, 'station_relocation')
    except ValueError as exc:
        raise ValueError('CP09 station/proof bytes differ from authoritative selected anchors: '+str(exc)) from exc
    proof=json.loads(raw)
    if (proof['source_id']!=row['source_id'] or proof['station_config_sha256']!=station_config_sha256
        or proof['scene_sha256']!=row['station_sha256'] or proof['source_h5_sha256']!=row['h5_sha256']
        or proof['original_source_scene']['sha256']!=row['source_scene_sha256']
        or proof['allowed_projection']['removed_prim_roots']!=['/World/CP09']
        or proof['allowed_projection']['all_other_authored_prim_subtrees_equal'] is not True):
        raise ValueError('CP09 authoritative station projection fields differ')
    return proof
