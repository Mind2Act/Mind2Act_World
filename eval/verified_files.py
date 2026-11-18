"""Bounded process-local memo of bytes actually matched to an expected SHA.

Callers still validate every descriptor, binding and transitive member. There
are no cached authorization verdicts, mutable parsed objects or disk markers.
"""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import stat
import threading
import time

_MAX_ENTRIES = 4096
_MAX_RETAINED_BYTES = 32 * 1024**2
_MIN_AGE_NS = 1_000_000_000
_clock_ns = time.time_ns
_memo = OrderedDict()
_retained_bytes = 0
_lock = threading.RLock()


def _after_fork():
    global _memo, _retained_bytes, _lock
    _memo = OrderedDict()
    _retained_bytes = 0
    _lock = threading.RLock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _old_enough(info):
    # Linux can coarsen inode updates within one write tick. A newly created or
    # just-modified inode must not earn reuse during that ambiguous interval.
    return _clock_ns() - max(info.st_mtime_ns, info.st_ctime_ns) >= _MIN_AGE_NS


def _remember(key, identity, raw):
    global _retained_bytes
    with _lock:
        previous = _memo.pop(key, None)
        if previous is not None and previous[1] is not None:
            _retained_bytes -= len(previous[1])
        _memo[key] = (identity, raw)
        if raw is not None:
            _retained_bytes += len(raw)
        while len(_memo) > _MAX_ENTRIES or _retained_bytes > _MAX_RETAINED_BYTES:
            _, (_, discarded) = _memo.popitem(last=False)
            if discarded is not None:
                _retained_bytes -= len(discarded)


def _verified(path, expected, *, max_bytes=None, retain=False):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('a complete SHA256 is required')
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path  # Preserve symlink/.. spelling and semantics.
    key = (str(path), expected)  # Distinct aliases earn independent entries.
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('verified input must be a regular non-symlink file')
    identity = file_identity(before)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or file_identity(opened) != identity:
            raise ValueError('file identity changed before verification: ' + str(path))
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ValueError('pinned configuration must be a bounded regular file')
        with _lock:
            cached = _memo.get(key)
            if cached is not None:
                _memo.move_to_end(key)
        eligible = _old_enough(opened)
        if eligible and cached is not None and cached[0] == identity and (not retain or cached[1] is not None):
            raw = cached[1]
        else:
            digest = hashlib.sha256()
            blocks = [] if retain else None
            total = 0
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                total += len(block)
                if max_bytes is not None and total > max_bytes:
                    raise ValueError('pinned configuration must be a bounded regular file')
                digest.update(block)
                if retain:
                    blocks.append(block)
            if digest.hexdigest() != expected:
                raise ValueError('file SHA256 mismatch: ' + str(path))
            raw = b''.join(blocks) if retain else None
        # Recheck the open inode and its current name even on a memo hit.
        if (file_identity(os.fstat(stream.fileno())) != identity
                or file_identity(path.lstat()) != identity):
            raise ValueError('file identity changed during verification: ' + str(path))
    if eligible:
        _remember(key, identity, raw)
    return raw


def verify_sha256(path, expected):
    _verified(path, expected)


def verified_bytes(path, expected, *, max_bytes=16 * 1024**2):
    """Immutable bytes only: consumers construct their own fresh parsed objects."""
    return _verified(path, expected, max_bytes=max_bytes, retain=True)
