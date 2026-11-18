"""Exact original held-out inputs; private source data never enters a controller.

These pins identify the inspected 2250-case request and its verified raw-byte
snapshot. They authorize input identity, not budgets or native qualification.
"""
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from .verified_files import verified_bytes

REQUEST_SHA256 = '762e1a6dedba760c10f317865feb99d44f9a17ccb7f4d364e4bae825a86219e3'
SNAPSHOT_SHA256 = '917fa443a3ac000e4050837ebf08a84b09052e5707f72d17146d4fc799545103'
MODE = 'source_bound_test'
SCHEMA = 'mind2act_eval.original_test_source.v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _immutable(value):
    if isinstance(value,dict):
        return MappingProxyType({key:_immutable(item) for key,item in value.items()})
    if isinstance(value,list):return tuple(_immutable(item) for item in value)
    return value


@lru_cache(maxsize=4)
def _request(raw):
    rows = json.loads(raw)['records']
    result = {row['logical_slot']: row for row in rows}
    if len(result) != len(rows) or len(rows) != 2250:
        raise ValueError('original test source request grid differs')
    return _immutable(result)


def request_records(path, *, strict=False):
    path = Path(path).resolve(strict=True)
    # Authenticate current bytes through the shared freshness/inode/name guard
    # before any parsed-cache lookup. Parsed authority never escapes mutable.
    raw=verified_bytes(path,REQUEST_SHA256,max_bytes=4*1024**2) if strict else path.read_bytes()
    return _request(raw)


def source_identity(spec):
    source = spec.config.get('test_source')
    if (not isinstance(source, Mapping) or source.get('schema') != SCHEMA
            or source.get('instance_set') != 'test50-r5' or source.get('split') != 'test'
            or source.get('logical_slot') != spec.instance_id
            or type(source.get('source_seed')) is not int or type(spec.seed) is not int
            or source.get('source_seed') != spec.seed):
        raise ValueError('explicit original test source identity required')
    return source


def checked_record(spec, records):
    source = source_identity(spec)
    row = records.get(spec.instance_id)
    if (row is None or row['task_id'] != spec.task_id or row['difficulty'] != spec.difficulty
            or type(spec.seed) is not int or spec.seed != row['seed']
            or any(source.get(k) != row[k] for k in
                   ('logical_slot', 'native_case_id'))):
        raise ValueError('original test source case/seed/config/index binding differs')
    return row


def read(spec, assets, *, strict=False):
    """Read the exact original case, with no case-ID or seed substitution."""
    row = checked_record(spec, request_records(assets.resolve('test_source_request'), strict=strict))
    raw = assets.resolve('test_source_raw').read_bytes()
    if strict and digest(raw) != row['source_configuration_sha256']:
        raise ValueError('original test case bytes SHA mismatch')
    value = json.loads(raw)
    document_id = value.get('case_id', value.get('instance_id'))
    if source_identity(spec).get('source_document_id') != document_id:
        raise ValueError('original test raw document identity differs')
    return value


def audit_source(spec, assets):
    # The independent scorer rechecks the same immutable request and original
    # case bytes, rather than trusting a caller-supplied condition hash.
    read(spec, assets)
    return dict(test_source=dict(source_identity(spec)),
                test_source_request=str(assets.resolve('test_source_request')),
                test_source_raw=str(assets.resolve('test_source_raw')))


def read_audit(spec, declaration):
    from types import SimpleNamespace
    roles = {'test_source_request': declaration['test_source_request'],
             'test_source_raw': declaration['test_source_raw']}
    if dict(source_identity(spec)) != declaration['test_source']:
        raise ValueError('original test audit identity differs')
    return read(spec, SimpleNamespace(resolve=lambda key: Path(roles[key])))


def geometry(spec, assets, *, original_scene_sha256, available, method, roots):
    """Keep derived resources separate from the original scene-byte claim."""
    from .sim.source_assets import asset_bytes, source_file
    read(spec, assets)
    proof=json.loads(asset_bytes(assets, 'source_geometry_proof'))
    source=source_identity(spec)
    if (proof.get('schema')!='mind2act_eval.original_test_geometry.v1'
            or proof.get('task_id')!=spec.task_id or proof.get('logical_slot')!=spec.instance_id
            or proof.get('source_configuration_sha256')!=source['source_configuration_sha256']
            or proof.get('method')!=method or proof['original_scene']['available'] is not available
            or proof['original_scene']['sha256']!=original_scene_sha256
            or proof.get('copied_roots')!=roots):
        raise ValueError('original test geometry provenance differs')
    for role,key in (('station','compiled_scene'),('task_source','actual_composed_task')):
        source_file(assets, role)
    return proof
