"""Explicit original/portable asset binding, independent of legacy spec mutation."""
from pathlib import PurePosixPath
import posixpath
from mind2act_eval.sim.asset_relocation import SCHEMA

import hashlib
import json


def load_provenance(data,row):
    if hashlib.sha256(data).hexdigest()!=row['resource_projection_proof_sha256']:
        raise ValueError('CP06 external projection provenance digest differs')
    proof=json.loads(data)
    if proof.get('schema')!='cp06_original_portable_source_proof.v1' or proof.get('original_binding_changed') is not False:
        raise ValueError('CP06 external original/portable provenance schema differs')
    cells=[c for c in proof['cells'] if c['case_id']==row['instance_id']]
    if len(cells)!=1:raise ValueError('CP06 projection provenance must uniquely bind the selected case')
    cell=cells[0]
    if (cell['original_spec']['sha256']!=row['source_spec_sha256']
            or cell['original_config_sha256']!=row['declared_task_sha256']
            or cell['declared_config_sha256']!=row['declared_task_sha256']
            or cell['original_task']['sha256']!=row['original_task_layer_sha256']
            or cell['runtime_task']['sha256']!=row['derived_task_layer_sha256']
            or cell['semantic_field_changes_outside_allowed_projection']
            or cell['runtime_task']['inverse_prefix_sha_matches_original'] is not True):
        raise ValueError('CP06 projection provenance original/derived identity differs')
    original=cell['original_task']['source_uri'];mirror=cell['runtime_task']['source_mirror_path']
    if not original.startswith('/') or not mirror.endswith(original) or mirror==original:
        raise ValueError('CP06 projection provenance lacks its exact proved URI prefix')
    return dict(uri_prefix=mirror[:-len(original)])


def validate_manifest(manifest,row,original,station_layer,provenance):
    if manifest.get('schema')!=SCHEMA or manifest.get('source_sha256')!=row['prepared_scene_sha256']:
        raise ValueError('CP06 station resource projection source differs')
    source=original['config']['native_inputs']['manifest']
    records=[*source['asset_closure'],*source['station_asset_closure']]
    expected={r['path']:r['sha256'] for r in records}
    owners={}
    for r in records:owners.setdefault(r['sha256'],[]).append(r['path'])
    allowed=set(expected.values())|{row['derived_task_layer_sha256'],row['prepared_scene_sha256']}
    prefix=provenance['uri_prefix']
    mappings={}
    for mapping in manifest['mappings']:
        key=(mapping['source_layer_sha256'],mapping['authored_path'])
        if key in mappings:raise ValueError('CP06 duplicate resource projection')
        mappings[key]=mapping
        if mapping['target_sha256'] not in allowed:
            raise ValueError('CP06 resource projection target is outside selected closure')
        raw=mapping['authored_path']
        canonical=raw[len(prefix):] if raw.startswith(prefix+'/') else raw
        if canonical in expected and mapping['target_sha256']!=expected[canonical]:
            raise ValueError('CP06 authored URI mapped to wrong selected dependency')
        if not PurePosixPath(canonical).is_absolute():
            candidates=[posixpath.normpath(str(PurePosixPath(p).parent/canonical)) for p in owners.get(mapping['source_layer_sha256'],[])]
            pins={expected[p] for p in candidates if p in expected}
            if pins and mapping['target_sha256'] not in pins:
                raise ValueError('CP06 relative resource mapped to wrong selected dependency')
    def references(prim):
        if prim is None:return []
        result=list(prim.referenceList.GetAddedOrExplicitItems())
        for child in prim.nameChildren.values():result.extend(references(child))
        return [r for r in result if r.assetPath]
    task_refs=references(station_layer.GetPrimAtPath('/World/Task'))
    stand_refs=references(station_layer.GetPrimAtPath('/World/CameraStand'))
    if len(task_refs)!=1 or not stand_refs:
        raise ValueError('CP06 prepared station Task/selected stand reference contract differs')
    for refs,expected_sha in ((task_refs,row['derived_task_layer_sha256']),(stand_refs,row['stand_sha256'])):
        for ref in refs:
            rule=mappings.get((row['prepared_scene_sha256'],ref.assetPath))
            if rule is None or rule['target_sha256']!=expected_sha:
                raise ValueError('CP06 Task or repaired stand projection differs from source')
    return manifest
