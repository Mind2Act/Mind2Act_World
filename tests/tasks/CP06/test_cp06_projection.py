from copy import deepcopy
import pytest
from pxr import Sdf
from mind2act_eval.tasks.CP06.projection import validate_manifest,SCHEMA


def data():
    row=dict(prepared_scene_sha256='a'*64,derived_task_layer_sha256='b'*64,stand_sha256='c'*64)
    layer=Sdf.Layer.CreateAnonymous('cp06-role-test.usda')
    Sdf.CreatePrimInLayer(layer,'/World/Task').referenceList.prependedItems=[Sdf.Reference('authored/task.usda')]
    Sdf.CreatePrimInLayer(layer,'/World/CameraStand').referenceList.prependedItems=[Sdf.Reference('authored/stand.usd')]
    original=dict(config=dict(native_inputs=dict(manifest=dict(asset_closure=[dict(path='/raw/task.usda',sha256='d'*64)],
        station_asset_closure=[dict(path='/raw/stand.usd',sha256='c'*64)]))))
    manifest=dict(schema=SCHEMA,source_sha256='a'*64,mappings=[
        dict(source_layer_sha256='a'*64,authored_path='authored/task.usda',target_asset_id='task_source',target_sha256='b'*64),
        dict(source_layer_sha256='a'*64,authored_path='authored/stand.usd',target_asset_id='selected_stand',target_sha256='c'*64)])
    return manifest,row,original,layer


def test_original_and_prepared_task_are_separate_approved_layers():
    manifest,row,original,layer=data()
    assert validate_manifest(manifest,row,original,layer,{'uri_prefix':'/unit-test-mirror'})==manifest


def test_e2_stand_cannot_replace_selected_repaired_cp06_stand():
    manifest,row,original,layer=data();manifest['mappings'][1]['target_sha256']='6c4b9ec4cd325f6e47d862b87e7dddbe9adc5e9a5aa0f04224c47d0f16a600a8'
    with pytest.raises(ValueError,match='selected closure'):validate_manifest(manifest,row,original,layer,{'uri_prefix':'/unit-test-mirror'})


def test_task_reference_cannot_point_at_another_approved_geometry_file():
    manifest,row,original,layer=data();manifest['mappings'][0]['target_sha256']='c'*64
    with pytest.raises(ValueError,match='Task or repaired stand'):validate_manifest(manifest,row,original,layer,{'uri_prefix':'/unit-test-mirror'})


def test_exact_original_uri_cannot_be_rebound_to_different_selected_file():
    manifest,row,original,layer=data()
    manifest['mappings'].append(dict(source_layer_sha256='b'*64,authored_path='/raw/stand.usd',target_asset_id='wrong',target_sha256='d'*64))
    with pytest.raises(ValueError,match='wrong selected dependency'):validate_manifest(manifest,row,original,layer,{'uri_prefix':'/unit-test-mirror'})


def provenance_data():
    import json,hashlib
    row=dict(instance_id='cp06-unit',source_spec_sha256='e'*64,declared_task_sha256='f'*64,
        original_task_layer_sha256='d'*64,derived_task_layer_sha256='b'*64)
    proof=dict(schema='cp06_original_portable_source_proof.v1',original_binding_changed=False,cells=[dict(
        case_id='cp06-unit',original_spec={'sha256':'e'*64},original_config_sha256='f'*64,
        declared_config_sha256='f'*64,original_task=dict(sha256='d'*64,source_uri='/src/task.usda'),
        runtime_task=dict(sha256='b'*64,source_mirror_path='/mirror/src/task.usda',inverse_prefix_sha_matches_original=True),
        semantic_field_changes_outside_allowed_projection=[])])
    data=json.dumps(proof).encode();row['resource_projection_proof_sha256']=hashlib.sha256(data).hexdigest()
    return data,row


def test_private_uri_prefix_only_comes_from_digest_bound_external_proof():
    from mind2act_eval.tasks.CP06.projection import load_provenance
    data,row=provenance_data()
    assert load_provenance(data,row)=={'uri_prefix':'/mirror'}
    with pytest.raises(ValueError,match='digest'):load_provenance(data+b' ',row)
    row['source_spec_sha256']='0'*64
    with pytest.raises(ValueError,match='identity'):load_provenance(data,row)


def test_external_prefix_keeps_wrong_authored_dependency_rejection():
    manifest,row,original,layer=data()
    manifest['mappings'].append(dict(source_layer_sha256='b'*64,authored_path='/unit-test-mirror/raw/stand.usd',target_asset_id='wrong',target_sha256='d'*64))
    with pytest.raises(ValueError,match='wrong selected dependency'):
        validate_manifest(manifest,row,original,layer,{'uri_prefix':'/unit-test-mirror'})
