"""Narrow, audited repair of a never-launched quarantine campaign's missing GPU."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from mind2act_eval.distributed_campaign import NodeIdentity, load_partition, node_lock_key
from mind2act_eval.shared_queue import _json_bytes, _now


def _pinned_bytes(path, digest):
    if not Path(path).is_absolute():
        raise ValueError('amendment evidence paths must be absolute')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('amendment evidence must be a regular file')
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('amendment input SHA256 mismatch')
    return payload


def _pinned(path, digest):
    return json.loads(_pinned_bytes(path, digest))


def amend_gpu_omission(store, job_id, *, quarantine_id, expected_generation,
                       expected_definition_sha256, expected_external_batch, old_manifest,
                       new_manifest, new_manifest_sha256, unstarted_evidence,
                       unstarted_evidence_sha256, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('amendment requires an explicit reason')
    q = store.queue
    if expected_external_batch.get('hostname') != q.node.hostname:
        raise ValueError('GPU repair controller must run on the reserved node')
    old_raw = _pinned(old_manifest, expected_external_batch['manifest_sha256'])
    new_raw = _pinned(new_manifest, new_manifest_sha256)
    identity = NodeIdentity(q.node.hostname, q.node.machine_id_sha256)
    old = load_partition(old_manifest, expected_sha256=expected_external_batch['manifest_sha256'], local_identity=identity)
    new = load_partition(new_manifest, expected_sha256=new_manifest_sha256, local_identity=identity)
    if (len(old.campaign_partitions) != 1 or len(new.campaign_partitions) != 1
            or old.campaign_id != expected_external_batch['campaign_id']
            or any(getattr(old, key) != getattr(new, key) for key in (
                'campaign_id', 'output_root', 'lock_root', 'label', 'hostname',
                'machine_id_sha256', 'resource_config_sha256'))
            or old.lock_root.resolve() != q.lock_root.resolve()
            or not old.output_root.resolve().is_relative_to(q.lock_root.parent.resolve())):
        raise ValueError('amendment cannot change campaign, node, output, lock or resources')
    old_entries, new_entries = json.loads(old.manifest_bytes), json.loads(new.manifest_bytes)
    if (len(old_entries) != 1 or len(new_entries) != 1 or 'gpu' in old_entries[0]
            or {k:v for k,v in new_entries[0].items() if k != 'gpu'} != old_entries[0]
            or type(new_entries[0].get('gpu')) is not int
            or new_entries[0]['gpu'] not in q.node.gpu_indices):
        raise ValueError('only a missing GPU may be repaired with an allocated integer GPU')
    # Exact same CPU input admission as the distributed batch CLI, including H5.
    from mind2act_eval.cli import _batch_jobs, _job_defaults
    _batch_jobs(new_entries, _job_defaults(), distributed=True)
    proof = _pinned(unstarted_evidence, unstarted_evidence_sha256)
    if (proof.get('schema') != 'mind2act_eval.external_unstarted.v1'
            or proof.get('quarantine_id') != quarantine_id
            or proof.get('external_batch') != expected_external_batch
            or proof.get('launchers_disabled') is not True
            or any(type(proof.get(key)) is not int or proof[key] != 0 for key in ('launch_count', 'result_count'))
            or proof.get('owned_processes') != [] or not proof.get('sources')):
        raise ValueError('GPU amendment requires pinned positive unstarted evidence')
    for source in proof['sources']:
        _pinned_bytes(source['path'], source['sha256'])
    with ExitStack() as gates:
        held_gates = []
        for path, mode in ((q.lock_root/'batch.lock', fcntl.LOCK_SH),
                           (q.lock_root/f'batch-node-{node_lock_key(q.node.hostname)}.lock', fcntl.LOCK_EX),
                           (q.lock_root/f'campaign-{old.campaign_id}.lock', fcntl.LOCK_EX)):
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            gates.callback(os.close, fd)
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
            gates.callback(fcntl.flock, fd, fcntl.LOCK_UN)
            held_gates.append((path, fd))
        campaign_root = old.output_root/f'campaign-{old.campaign_id}'
        try:
            os.lstat(campaign_root)
        except FileNotFoundError:
            pass
        else:
            raise ValueError('external campaign output/descriptor exists; not an unstarted omission')
        with store._transaction(operation='amend_quarantine_external_gpu', maintenance_locks=held_gates) as state:
            row = store._quarantine_target(state, job_id, expected_generation,
                                           expected_definition_sha256, expected_external_batch)
            record = row.get('quarantine', {})
            if (row['status'] != 'blocked' or record.get('status') != 'quarantined'
                    or record.get('quarantine_id') != quarantine_id
                    or record.get('external_batch') != expected_external_batch):
                raise ValueError('quarantine amendment CAS differs from its unstarted reservation')
            if (old_entries[0]['episode_key'] != row['definition']['episode_key']
                    or any(old_entries[0].get(k) != v for k,v in row['definition']['options'].items())):
                raise ValueError('external job differs from the original quarantined inputs')
            # Recheck reviewed bytes after the maintenance gates, before commit.
            _pinned(old_manifest, expected_external_batch['manifest_sha256'])
            _pinned(new_manifest, new_manifest_sha256)
            _pinned(unstarted_evidence, unstarted_evidence_sha256)
            for partition in (old, new):
                _pinned_bytes(partition.manifest_path, partition.manifest_sha256)
                _pinned_bytes(partition.resource_config_path, partition.resource_config_sha256)
            for source in proof['sources']:
                _pinned_bytes(source['path'], source['sha256'])
            for path, fd in held_gates:
                held, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
                if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                    raise RuntimeError('amendment maintenance lock identity changed')
            new_binding = dict(expected_external_batch, manifest_sha256=new_manifest_sha256)
            audit = dict(kind='quarantine_external_gpu_amendment', quarantine_id=quarantine_id,
                old_external_batch=json.loads(_json_bytes(expected_external_batch)),
                new_external_batch=new_binding, old_manifest=str(old_manifest), new_manifest=str(new_manifest),
                old_manifest_contents=old_raw, new_manifest_contents=new_raw,
                old_jobs=old_entries, new_jobs=new_entries, gpu=new_entries[0]['gpu'],
                unstarted_evidence=dict(path=str(unstarted_evidence), sha256=unstarted_evidence_sha256, contents=proof),
                reason=reason, at=_now())
            row['history'].append(audit)
            record['external_batch'] = new_binding
            return json.loads(_json_bytes(audit))
