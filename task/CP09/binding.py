"""Fail closed until the independent asset owner pins the station projection."""
import hashlib,json

# Exact, already-portable task-stripped acquisition stations in c2ec5b07.
# Their proofs retain the training origin of the station resource; original
# test case identity and drawing geometry are authenticated separately.
PORTABLE_STATION = {
    'easy': ('7afb4488dded1bf5f9e9bdc1b499cfd4092042677705869c00ed0d7b86fe2948',
             '45a9227f01ad3b479e4a9a27fef593b0ee0c3390e0e21b3175d485468e1dc07c'),
    'medium': ('10fb1146b9303c3d426b2c0ec1360aa12bf158a163e0286695833580788d94f7',
               'cbabf63ed228d1ff5412096299f0c1ad47a390bfe4d54ed8fae38d30dd786479'),
    'hard': ('30a42cef57b4751d846c50157d2bdebf5955776d7e06d4d08e70afaec3b88e64',
             'ac0533a8de6049f1e86a2270bb56dc1317f42a02feae80de65631f879bc6674a'),
}


def validate_station(row,assets,station_config_sha256,*,spec=None):
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
