"""Fail closed until the independent asset owner pins the station projection."""
import hashlib,json

# Exact, already-portable task-stripped acquisition stations in c2ec5b07.
# Their proofs retain the training origin of the station resource; original
# test case identity and drawing geometry are authenticated separately.
PORTABLE_STATION = {
    'easy': ('7afb4488dded1bf5f9e9bdc1b499cfd4092042677705869c00ed0d7b86fe2948',
             'e031532f864ca9268d7708cbb22e016f6f9dab530d78c47975c8a32c7cc5033c'),
    'medium': ('10fb1146b9303c3d426b2c0ec1360aa12bf158a163e0286695833580788d94f7',
               'c6930b1603a132477f7da36add97116c05239e42490390d81fa31efd66882daa'),
    'hard': ('30a42cef57b4751d846c50157d2bdebf5955776d7e06d4d08e70afaec3b88e64',
             '642eb85042c82da25c076ec745dea6c562eb768b43e279fd4f8d1287914d7c07'),
}


def validate_station(row,assets,station_config_sha256,*,spec=None):
    if spec is not None and ('train_source' in spec.config or spec.config.get('layout_split')=='validation'):
        from mind2act_eval.cp09_training_inputs import validate_assets as training_validate_assets
        from mind2act_eval.sim.source_assets import read_source_bytes, source_layer, verify_source_file
        validation=spec.config.get('layout_split')=='validation'
        if validation:
            from mind2act_eval.cp09_validation_inputs import validate_assets
            runtime_row=validate_assets(spec,assets)
        else:
            runtime_row=training_validate_assets(spec,assets)
        scene_sha,proof_sha=PORTABLE_STATION[spec.difficulty]
        if (runtime_row['station_source_sha256']!=scene_sha
                or runtime_row['station_proof_sha256']!=proof_sha):
            raise ValueError('CP09 training common station pins differ')
        verify_source_file(assets,'station',runtime_row['station_runtime_sha256'] if validation else scene_sha)
        proof=json.loads(read_source_bytes(assets,'station_proof',proof_sha))
        original_scenes={
            'easy':'635b1ce5aca3596e8ae05c12f2cd6a3be3b14ca43e0c29c62a937c850666afdf',
            'medium':'d5d1ab22e9ddada81db9177a9d7e416f6d5942ed7449a2f0f19a1b9da4df1786',
            'hard':'0b627b2c8aea1aae5695aaf85586b4f7f4ad4c129355ad57fef4dc4eabe28a79'}
        deletion_scene_sha={'easy':'e9cb1fb39fbab0a6f37e5d41eb151c4d26682c454cdd4fcb7bb256040d245fa0',
                            'medium':'90f8621988215f172bec99a6789ca732bca98f23fb8675a8954075349b321ae7',
                            'hard':'5f66ed393cb261d56950b06294a7426a9dbfc8cc7b91fe95505bd7bc76a2245b'}
        if (proof['station_config_sha256']!=station_config_sha256
                or proof['scene_sha256']!=deletion_scene_sha[spec.difficulty]
                or proof['original_source_scene']['sha256']!=original_scenes[spec.difficulty]
                or proof['allowed_projection']['removed_prim_roots']!=['/World/CP09']
                or proof['allowed_projection']['all_other_authored_prim_subtrees_equal'] is not True):
            raise ValueError('CP09 training common station projection differs')
        # Own Task is a CPU-authored runtime dependency, never an old selected Task.
        task=source_layer(assets,'validation_task' if validation else 'training_task',spec.config['authored_task_source_sha256'])
        if task.GetPrimAtPath('/World/CP09') is None:
            raise ValueError('CP09 own authored Task root missing')
        return proof
    test=spec is not None and 'test_source' in spec.config
    expected_scene=row.get('station_sha256');expected_proof=row.get('station_proof_sha256')
    expected_relocation=row.get('station_relocation_sha256')
    if test:
        from .instances import original_case
        original_case(spec,assets)
        expected_scene,expected_proof=PORTABLE_STATION[spec.difficulty]
    if not expected_scene or not expected_proof or not expected_relocation:
        raise RuntimeError('CP09 awaiting authoritative selected station/proof SHA256 anchors')
    from mind2act_eval.sim.source_assets import read_source_bytes, verify_source_file
    try:
        verify_source_file(assets,'station',expected_scene)
        raw=read_source_bytes(assets,'station_proof',expected_proof)
        if not test:read_source_bytes(assets,'station_relocation',expected_relocation)
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
