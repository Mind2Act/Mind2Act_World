"""Authenticate original inputs independently of caller-provided asset hashes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind2act_eval import original_test_inputs as source


@pytest.fixture
def original(tmp_path, monkeypatch):
    raw = json.dumps(dict(task_id='CP09', case_id='raw-document-id', seed=50)).encode()
    row = dict(task_id='CP09', difficulty='easy', seed=50, logical_slot='CP09_easy_clean_000',
        native_case_id='CP09-heldout-easy-f0-i0-l0', source_configuration_sha256=source.digest(raw),
        source_index=dict(sha256='a'*64))
    request = json.dumps(dict(records=[row]+[dict(logical_slot=f'other-{i}') for i in range(2249)])).encode()
    monkeypatch.setattr(source, 'REQUEST_SHA256', source.digest(request))
    (tmp_path/'request').write_bytes(request);(tmp_path/'raw').write_bytes(raw)
    metadata = dict(schema=source.SCHEMA, instance_set='test50-r5', split='test',
        logical_slot=row['logical_slot'], native_case_id=row['native_case_id'],
        source_document_id='raw-document-id', source_configuration_sha256=row['source_configuration_sha256'],
        source_index_sha256='a'*64, source_seed=50, source_request_sha256=source.REQUEST_SHA256,
        snapshot_sha256=source.SNAPSHOT_SHA256)
    spec = SimpleNamespace(task_id='CP09',difficulty='easy',seed=50,instance_id=row['logical_slot'],
        config=dict(test_source=metadata))
    assets = SimpleNamespace(resolve=lambda k: tmp_path/('request' if k=='test_source_request' else 'raw'))
    return spec,assets


def test_keeps_raw_document_id_distinct_from_native_and_logical_ids(original):
    spec,assets = original
    assert source.read(spec,assets)['case_id']=='raw-document-id'
    assert source.read_audit(spec,source.audit_source(spec,assets))==source.read(spec,assets)


@pytest.mark.parametrize('seed,alias',[(0,False),(50,50.)])
def test_seed_identity_rejects_json_boolean_and_float_aliases(original,seed,alias):
    spec,_=original;spec.seed=seed;spec.config['test_source']['source_seed']=alias
    with pytest.raises(ValueError):source.source_identity(spec)


@pytest.mark.parametrize('mutation',['outer','row','nested_index'])
def test_returned_authority_cannot_authorize_substituted_raw_or_index(original,mutation):
    spec,assets=original
    genuine=source.read(spec,assets)
    rows=source.request_records(assets.resolve('test_source_request'))
    metadata=spec.config['test_source'];original_metadata=dict(metadata)
    altered=dict(genuine,substituted=True);raw=json.dumps(altered).encode()
    try:
        if mutation=='nested_index':
            metadata['source_index_sha256']='b'*64
            rows[spec.instance_id]['source_index']['sha256']='b'*64
        else:
            metadata['source_configuration_sha256']=source.digest(raw)
            assets.resolve('test_source_raw').write_bytes(raw)
            if mutation=='outer':
                rows[spec.instance_id]=dict(rows[spec.instance_id],source_configuration_sha256=source.digest(raw))
            else:rows[spec.instance_id]['source_configuration_sha256']=source.digest(raw)
    except TypeError:
        pass  # Deep immutability or detached caller copies are both valid.
    with pytest.raises(ValueError):source.read(spec,assets)
    metadata.update(original_metadata)
    assets.resolve('test_source_raw').write_bytes(json.dumps(genuine).encode())
    assert source.read(spec,assets)==genuine


def test_warm_request_rejects_changed_bytes_with_same_observed_metadata(original,monkeypatch):
    from mind2act_eval import verified_files
    spec,assets=original;path=assets.resolve('test_source_request')
    observed=path.stat();inode=(observed.st_dev,observed.st_ino)
    real_stat,real_lstat,real_fstat=Path.stat,Path.lstat,os.fstat
    # Model Linux metadata coarsening within one recent write tick, including
    # the open-inode/name checks of the shared verified-byte reader.
    monkeypatch.setattr(Path,'stat',lambda p,*a,**kw:observed if p==path else real_stat(p,*a,**kw))
    monkeypatch.setattr(Path,'lstat',lambda p,*a,**kw:observed if p==path else real_lstat(p,*a,**kw))
    def coarsened(fd):
        info=real_fstat(fd)
        return observed if (info.st_dev,info.st_ino)==inode else info
    monkeypatch.setattr(os,'fstat',coarsened)
    monkeypatch.setattr(verified_files,'_clock_ns',lambda:max(observed.st_mtime_ns,observed.st_ctime_ns)+1)
    assert source.read(spec,assets)
    raw=path.read_bytes();altered=raw.replace(b'CP09',b'CP10',1)
    assert len(altered)==len(raw) and source.digest(altered)!=source.REQUEST_SHA256
    path.write_bytes(altered)
    with pytest.raises(ValueError):source.read(spec,assets)


@pytest.mark.parametrize('mutation',['request','raw','wrong_split','native','document','seed',
    'source_index','self_asserted_hash','audit_source'])
def test_rejects_cross_source_tamper_and_wrong_split(original,mutation):
    spec,assets=original
    declaration=source.audit_source(spec,assets)
    m=spec.config['test_source']
    if mutation in ('request','raw'):
        assets.resolve('test_source_'+mutation).write_bytes(b'{}')
    elif mutation=='wrong_split':m['split']='train'
    elif mutation=='native':m['native_case_id']='another-native'
    elif mutation=='document':m['source_document_id']=spec.instance_id
    elif mutation=='seed':spec.seed=51;m['source_seed']=51
    elif mutation=='source_index':m['source_index_sha256']='b'*64
    elif mutation=='self_asserted_hash':
        assets.resolve('test_source_raw').write_bytes(b'{}');m['source_configuration_sha256']=source.digest(b'{}')
    elif mutation=='audit_source':declaration['test_source']['logical_slot']='another-case'
    with pytest.raises(ValueError):
        if mutation=='audit_source':source.read_audit(spec,declaration)
        else:source.read(spec,assets)
