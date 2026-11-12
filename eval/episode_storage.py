"""Node caches and supervisor-owned, post-seal scratch reclamation.

No existing episode is adopted. Only the in-memory capability returned while
creating a fresh attempt can authorize deletion, after its child's verified exit.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import uuid
import weakref

SCRATCH = ('cache', 'tmp', 'kit')
OWNER = 'scratch-owner.json'
LEDGER = 'scratch-ledger.json'
RECEIPT = 'scratch-cleanup.json'
_CAPABILITIES = weakref.WeakValueDictionary()


class _ScratchOwner(dict):
    """Only an instance returned for a fresh tree authorizes parent cleanup."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def durable_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('receipt cannot be a symlink')
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _physical(path):
    path = Path(path).absolute()
    if '..' in path.parts or path.resolve() != path:
        raise ValueError('storage path must be physical, without symlink components')
    return path


def node_cache(attempt, *, hostname=None, fallback_root=None):
    """M2A_CACHE_ROOT is the cluster plan's cache_root; UID/node isolation is mandatory.

    With no cluster configuration the fallback is stable beside the run root,
    never a user-home path or the installed SDK. Data/config/logs stay private.
    """
    configured = os.environ.get('M2A_CACHE_ROOT')
    root = Path(configured) if configured else Path(fallback_root or Path(attempt).absolute().parent / '.mind2act-cache')
    if not root.is_absolute():
        raise ValueError('M2A_CACHE_ROOT must be absolute')
    root = _physical(root)
    attempt = _physical(attempt)
    if root == attempt or root.is_relative_to(attempt):
        raise ValueError('shared cache must be outside episode output')
    node = hashlib.sha256((hostname or socket.gethostname()).encode()).hexdigest()[:24]
    cache = _physical(root / 'kit-cache' / f'uid-{os.getuid()}' / node)
    if attempt.is_relative_to(cache):
        raise ValueError('episode output must be outside the shared cache')
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    _physical(cache)
    for name in ('shaders', 'driver-shaders', 'cuda', 'extensions', 'textures', 'station-layers'):
        directory = _physical(cache / name)
        directory.mkdir(exist_ok=True, mode=0o700)
        _physical(directory)
    return cache


@contextmanager
def cache_use_lock(cache, *, blocking=True, cancel_event=None):
    """Hold a shared active-user lease; parallel episode children never serialize.

    Only dedicated shader/CUDA/extension/texture caches are shared. Mutable Kit
    app state and cooking databases remain private. Existing startup gates still
    serialize extension startup; this lease adds no exclusive worker gate.
    """
    import fcntl
    fd = os.open(_physical(cache) / 'active.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        import time
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError('cancelled while waiting for node cache writer')
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not blocking:
                    raise
                time.sleep(.1)
        yield fd
    finally:
        os.close(fd)


def worker_cache():
    """Use shared Kit storage only with the supervisor's inherited lock FD."""
    configured = os.environ.get('MIND2ACT_EVAL_NODE_CACHE')
    if configured is None:
        return None
    cache = _physical(configured)
    try:
        fd = int(os.environ['MIND2ACT_EVAL_CACHE_LOCK_FD'])
        held, named = os.fstat(fd), (cache / 'active.lock').lstat()
    except (KeyError, ValueError, OSError) as exc:
        raise ValueError('shared Kit cache requires its inherited supervisor active-use lock') from exc
    if not stat.S_ISREG(named.st_mode) or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError('shared Kit cache active-use lock identity changed')
    return cache


def prepare_scratch(directory):
    root = _physical(directory)
    info = root.stat()
    owner = _ScratchOwner(schema='mind2act_eval.scratch_owner.v1', token=uuid.uuid4().hex,
                          root=str(root), device=info.st_dev, inode=info.st_ino, paths=[])
    if (root / OWNER).exists():
        raise ValueError('existing scratch ownership cannot be adopted')
    for relative in SCRATCH:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        info = path.stat()
        owner['paths'].append(dict(path=relative, device=info.st_dev, inode=info.st_ino))
    for relative in ('kit/cache', 'kit/data'):
        (root / relative).mkdir()
    durable_json(root / OWNER, owner)
    _CAPABILITIES[id(owner)] = owner
    return owner


def _files(root):
    files = []
    for path in root.rglob('*'):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('scratch contains a symlink: ' + str(path))
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('scratch contains a non-regular file: ' + str(path))
        files.append(path)
    return files


def disk_usage(root):
    """Actual regular-file logical and allocated bytes; never follow links."""
    files = _files(Path(root))
    return dict(bytes=sum(p.stat().st_size for p in files),
                allocated_bytes=sum(getattr(p.stat(), 'st_blocks', 0) * 512 for p in files))


def process_group_exited(pid):
    """Read-only conservative exit proof for the Popen start_new_session group."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    # Dead descendants can remain as zombies until their new parent reaps them.
    # They own no open descriptors and cannot write. Match PRO's treatment of Z/X
    # while retaining conservative failure on an unavailable/incomplete probe.
    import subprocess
    try:
        rows = subprocess.run(['ps', '-axo', 'pgid=,stat='], check=True,
                              capture_output=True, text=True, timeout=5).stdout.splitlines()
        if not rows:
            return False
        for row in rows:
            group, state = row.split()
            if not group.isdecimal() or not state:
                return False
            if int(group) == pid and state[0] not in {'Z', 'X'}:
                return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return True


def cleanup_scratch(directory, owner, *, worker_pid, exited, expected_consumer=None):
    """Call only from the supervising parent after it proves its child/tree exited."""
    from mind2act_eval.sealing import SEAL_NAME, validate_worker_seal
    root = _physical(directory)
    if exited is not True:
        raise ValueError('scratch cleanup requires verified child exit')
    if type(owner) is not _ScratchOwner or _CAPABILITIES.get(id(owner)) is not owner:
        raise ValueError('scratch cleanup requires this parent\'s fresh ownership capability')
    info = root.stat()
    if (owner.get('root') != str(root) or owner.get('device') != info.st_dev
            or owner.get('inode') != info.st_ino or (root / OWNER).is_symlink()
            or json.loads((root / OWNER).read_text()) != owner
            or [p['path'] for p in owner['paths']] != list(SCRATCH)):
        raise ValueError('scratch ownership differs from this new generation')
    _, seal = validate_worker_seal(root, durable=True, expected_consumer=expected_consumer)
    if seal['worker_pid'] != worker_pid:
        raise ValueError('scratch seal belongs to a different worker')
    # Preflight ALL paths before deleting ANY. A replaced directory, escaping
    # symlink, or special file leaves the complete scratch tree untouched.
    rows = []
    for owned in owner['paths']:
        path = _physical(root / owned['path'])
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (owned['device'], owned['inode']):
            raise ValueError('owned scratch directory was replaced')
        for file in _files(path):
            rows.append(dict(path=file.relative_to(root).as_posix(), bytes=file.stat().st_size,
                             sha256=sha256(file)))
    before = disk_usage(root)
    ledger = dict(schema='mind2act_eval.scratch_ledger.v1', token=owner['token'],
                  worker_pid=worker_pid, seal_sha256=sha256(root / SEAL_NAME), files=rows)
    durable_json(root / LEDGER, ledger)
    receipt = dict(schema='mind2act_eval.scratch_cleanup.v1', status='planned',
                   ledger_sha256=sha256(root / LEDGER), seal_sha256=ledger['seal_sha256'],
                   before_bytes=before['bytes'], before_allocated_bytes=before['allocated_bytes'],
                   deleted_bytes=0, deleted_paths=[])
    durable_json(root / RECEIPT, receipt)
    try:
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ValueError('scratch cleanup needs symlink-resistant directory deletion')
        for owned in owner['paths']:
            path = _physical(root / owned['path'])
            info = path.lstat()
            if (info.st_dev, info.st_ino) != (owned['device'], owned['inode']):
                raise ValueError('owned scratch directory changed before deletion')
            shutil.rmtree(path)
            receipt['deleted_paths'].append(owned['path'])
            receipt['deleted_bytes'] += sum(r['bytes'] for r in rows
                if Path(r['path']).is_relative_to(owned['path']))
            durable_json(root / RECEIPT, receipt)
        validate_worker_seal(root, durable=True, expected_consumer=expected_consumer)
        after = disk_usage(root)
        receipt.update(after_bytes=after['bytes'], after_allocated_bytes=after['allocated_bytes'],
                       measurement_phase='after deletion, before final cleanup receipt rewrite')
        receipt['status'] = 'complete'
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        durable_json(root / RECEIPT, receipt)
    return receipt


def inventory(directory):
    """Read-only reclaimable-size inventory, including legacy/unsealed results.

    Reclaimable means regenerable bytes observed, never deletion authorization.
    Private scoring/replay evidence is always counted as retained.
    """
    root = _physical(directory)
    usage = disk_usage(root)
    candidates = []
    for relative in SCRATCH:
        path = root / relative
        if path.is_dir():
            candidates.append(dict(path=relative, **disk_usage(path)))
    station = root / 'station-relocation'
    return dict(schema='mind2act_eval.storage_inventory.v1', directory=str(root),
                total=usage, regenerable_scratch=candidates,
                regenerable_bytes=sum(row['bytes'] for row in candidates),
                station_layers=disk_usage(station) if station.is_dir() else None,
                legacy_cleanup_authorized=False)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Read-only episode storage inventory; never deletes old results')
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    rows = []
    for directory in args.directories:
        try:
            rows.append(inventory(directory))
        except (OSError, ValueError) as exc:
            rows.append(dict(directory=str(directory), error=f'{type(exc).__name__}: {exc}'))
    result = dict(episodes=rows, regenerable_bytes=sum(row.get('regenerable_bytes', 0) for row in rows),
                  policy='inventory only; existing episodes are never adopted for cleanup')
    if args.output:
        durable_json(args.output, result)
    print(json.dumps(result, indent=2))
    return int(any('error' in row for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
