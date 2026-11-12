"""Real retained-seal / owned scratch boundary; no GPU or SDK."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from test_sealing import _successful_attempt
from mind2act_eval.sealing import seal_worker_outputs, validate_worker_seal


def test_cleanup_preserves_rejudge_evidence_and_verifiable_seal(tmp_path):
    from mind2act_eval.episode_storage import prepare_scratch, cleanup_scratch
    owner = prepare_scratch(tmp_path)
    _successful_attempt(tmp_path, pid=os.getpid())
    (tmp_path / 'PR4').mkdir()
    (tmp_path / 'PR4/private-scoring.npz').write_bytes(b'exact private scoring evidence')
    (tmp_path / 'worker.log').write_text('original diagnostic cause\n')
    scratch = tmp_path / 'kit/data/large.bin'
    scratch.write_bytes(b'x' * 65536)
    (tmp_path / 'cache/shader.bin').write_bytes(b'y' * 65536)
    seal = seal_worker_outputs(tmp_path)
    receipt = cleanup_scratch(tmp_path, owner, worker_pid=os.getpid(), exited=True)
    assert receipt['status'] == 'complete'
    assert receipt['deleted_bytes'] == 131072
    assert receipt['before_bytes'] > receipt['after_bytes']
    assert (tmp_path / 'PR4/private-scoring.npz').read_bytes() == b'exact private scoring evidence'
    assert (tmp_path / 'worker.log').read_text() == 'original diagnostic cause\n'
    assert validate_worker_seal(tmp_path)[1] == seal
    ledger = json.loads((tmp_path / 'scratch-ledger.json').read_text())
    assert any(r['sha256'] == hashlib.sha256(b'x' * 65536).hexdigest() for r in ledger['files'])
    assert not (tmp_path / 'cache').exists() and not (tmp_path / 'kit/data').exists()


@pytest.mark.parametrize('failure', ['live', 'unsealed', 'changed', 'symlink', 'wrong_owner', 'wrong_pid'])
def test_cleanup_refuses_unsafe_boundary_without_deleting(tmp_path, failure):
    from mind2act_eval.episode_storage import prepare_scratch, cleanup_scratch
    owner = prepare_scratch(tmp_path)
    _successful_attempt(tmp_path, pid=os.getpid())
    sentinel = tmp_path / 'cache/keep'
    sentinel.write_text('keep')
    seal_worker_outputs(tmp_path)
    if failure == 'unsealed':
        (tmp_path / 'worker-seal.json').unlink()
    if failure == 'changed':
        (tmp_path / 'evidence.json').write_text('tampered')
    if failure == 'symlink':
        (tmp_path / 'tmp/escape').symlink_to(tmp_path.parent)
    if failure == 'wrong_owner':
        owner = dict(owner, token='not this generation')
    with pytest.raises((ValueError, OSError)):
        cleanup_scratch(tmp_path, owner, worker_pid=os.getpid() + (failure == 'wrong_pid'),
                        exited=failure != 'live')
    assert sentinel.read_text() == 'keep'


def test_result_cannot_reference_excluded_ephemeral_evidence(tmp_path):
    from mind2act_eval.episode_storage import prepare_scratch
    prepare_scratch(tmp_path)
    _successful_attempt(tmp_path, artifacts={'private_evidence': 'cache/private.npz'})
    (tmp_path / 'cache/private.npz').write_bytes(b'essential evidence')
    with pytest.raises(ValueError, match='ephemeral'):
        seal_worker_outputs(tmp_path)


def test_node_cache_shares_only_cache_and_keeps_an_active_use_lease(tmp_path, monkeypatch):
    import fcntl
    from mind2act_eval.episode_storage import node_cache, cache_use_lock
    monkeypatch.setenv('M2A_CACHE_ROOT', str(tmp_path / 'cluster-cache'))
    cache = node_cache(tmp_path / 'attempt')
    assert cache.is_relative_to(tmp_path / 'cluster-cache/kit-cache')
    assert cache != node_cache(tmp_path / 'different', hostname='another-node')
    with cache_use_lock(cache):
        with (cache / 'active.lock').open('r+') as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cache_does_not_serialize_eight_active_workers(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from mind2act_eval.episode_storage import node_cache, cache_use_lock
    monkeypatch.setenv('M2A_CACHE_ROOT', str(tmp_path / 'cluster-cache'))
    cache = node_cache(tmp_path / 'attempt')
    with ExitStack() as stack:
        descriptors = [stack.enter_context(cache_use_lock(cache, blocking=False)) for _ in range(8)]
        assert len(set(descriptors)) == 8


def test_eight_real_children_inherit_concurrent_cache_use(tmp_path, monkeypatch):
    import fcntl
    import subprocess
    import sys
    import time
    from mind2act_eval.episode_storage import node_cache, cache_use_lock
    monkeypatch.setenv('M2A_CACHE_ROOT', str(tmp_path / 'cluster-cache'))
    cache = node_cache(tmp_path / 'attempt')
    release = tmp_path / 'release'
    script = '''from pathlib import Path
import sys,time
from mind2act_eval.episode_storage import worker_cache
cache=worker_cache()
Path(sys.argv[1]).write_text(str(cache))
deadline=time.monotonic()+10
while not Path(sys.argv[2]).exists() and time.monotonic()<deadline:time.sleep(.01)
assert Path(sys.argv[2]).exists()
'''
    children = []
    try:
        for index in range(8):
            with cache_use_lock(cache, blocking=False) as fd:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                    PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'),
                    MIND2ACT_EVAL_NODE_CACHE=str(cache), MIND2ACT_EVAL_CACHE_LOCK_FD=str(fd))
                children.append(subprocess.Popen([sys.executable, '-c', script,
                    str(tmp_path / f'ready-{index}'), str(release)], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(fd,)))
        deadline = time.monotonic()+5
        while len(list(tmp_path.glob('ready-*'))) != 8 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(list(tmp_path.glob('ready-*'))) == 8
        # All parent descriptors have closed: only the eight child leases remain.
        with (cache / 'active.lock').open('r+') as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.touch()
        for child in children:
            try:
                stdout, stderr = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.communicate()
                raise
            assert child.returncode == 0, stderr.decode()


def test_cleanup_removes_whole_temporary_kit_and_preserves_same_seal(tmp_path):
    from mind2act_eval.episode_storage import prepare_scratch, cleanup_scratch
    owner = prepare_scratch(tmp_path)
    _successful_attempt(tmp_path, pid=os.getpid())
    (tmp_path / 'kit/kit.log').write_text('startup table, duplicated in worker.log')
    (tmp_path / 'kit/user.config.json').write_text('{}')
    seal = seal_worker_outputs(tmp_path)
    cleanup_scratch(tmp_path, owner, worker_pid=os.getpid(), exited=True)
    assert not (tmp_path / 'kit').exists()
    assert validate_worker_seal(tmp_path)[1] == seal


def test_saved_owner_receipt_cannot_adopt_an_existing_attempt(tmp_path):
    from mind2act_eval.episode_storage import prepare_scratch, cleanup_scratch
    prepare_scratch(tmp_path)
    _successful_attempt(tmp_path, pid=os.getpid())
    sentinel = tmp_path / 'cache/keep'
    sentinel.write_text('old result must remain')
    seal_worker_outputs(tmp_path)
    saved_owner = json.loads((tmp_path / 'scratch-owner.json').read_text())
    with pytest.raises(ValueError, match='capability|ownership'):
        cleanup_scratch(tmp_path, saved_owner, worker_pid=os.getpid(), exited=True)
    assert sentinel.exists()
