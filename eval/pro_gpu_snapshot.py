"""Host-wide raw GPU telemetry, never cached ownership or admission verdicts.

A separate mutex coalesces collectors across parents. The capacity mutex must
never be held here. Expired or failed samples cannot authorize new work.
"""
import fcntl
import json
import os
from pathlib import Path
import stat
import time
import uuid
import xml.etree.ElementTree as ET

CACHE_REUSE_S = 5
MAX_AGE_S = 10
MIN_REMAINING_S = 1
SCHEMA = 'mind2act_eval.pro_gpu_host_snapshot.v2'


class SnapshotExpired(ValueError):
    """Telemetry needs refreshing; this is not an ownership verdict."""


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def validate_age(value):
    age = time.monotonic() - value['sampled_monotonic_s']
    if value['schema'] != SCHEMA or value['boot_id'] != _boot_id() or age < 0:
        raise ValueError('GPU host snapshot schema/boot/clock identity mismatch')
    if not age < MAX_AGE_S:
        raise SnapshotExpired('GPU host snapshot expired: age=%.3fs limit=%ds; resample required'
                              % (age, MAX_AGE_S))


def _regular(fd, path):
    info = os.fstat(fd)
    linked = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or info.st_mode & 0o077
            or (info.st_dev, info.st_ino) != (linked.st_dev, linked.st_ino)):
        raise ValueError('GPU snapshot must retain its owned private regular identity')


def _collect():
    from .cluster_profile import _query, _host_device_pids, process_identity
    started = time.monotonic()
    at = time.time()
    devices = _query(['--query-gpu=index,name,uuid,pci.bus_id,memory.free,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'])
    contexts = _query(['-q', '-x'])
    compute = _query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    fd_pids = {k: sorted(v) for k, v in _host_device_pids().items()}
    # Bind observed contexts to their births. Every consumer rechecks identity;
    # a reused PID must never inherit a cached context's ownership.
    pids = set()
    for process in ET.fromstring(contexts).findall('gpu/processes/process_info'):
        pids.add(int(process.findtext('pid')))
    for line in compute.splitlines():
        pids.add(int(line.split(',')[1].strip()))
    births = {}
    for pid in pids:
        try:
            births[str(pid)] = process_identity(pid)['birth']
        except (FileNotFoundError, ProcessLookupError):
            continue
    value = dict(schema=SCHEMA, boot_id=_boot_id(), sampled_unix_s=at,
                 sampled_monotonic_s=started, devices=devices, contexts=contexts,
                 compute=compute, fd_pids=fd_pids, births=births,
                 collected_monotonic_s=time.monotonic())
    validate_age(value)
    return value


def host_gpu_snapshot(root, *, force_refresh=False):
    root = Path(root)
    lock_path, path = root / 'pro-gpu-snapshot.lock', root / 'pro-gpu-snapshot.json'
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _regular(fd, lock_path)
        try:
            input_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            value = None
        else:
            with os.fdopen(input_fd, 'r') as stream:
                _regular(stream.fileno(), path)
                value = json.load(stream)
            try:
                validate_age(value)
                now = time.monotonic()
                completed = value['collected_monotonic_s']
                if (force_refresh or not value['sampled_monotonic_s'] <= completed <= now
                        or now - completed >= CACHE_REUSE_S
                        or now - value['sampled_monotonic_s'] >= MAX_AGE_S - MIN_REMAINING_S):
                    value = None
            except ValueError:
                value = None
        if value is not None:
            return value
        value = _collect()  # Failure never serves an old PASS/sample.
        temporary = root / ('.pro-gpu-snapshot-' + uuid.uuid4().hex)
        try:
            output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(output, 'w') as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                # Ephemeral raw telemetry needs atomic visibility, not a
                # durable reservation commit. CFS fsync can consume its TTL.
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        validate_age(value)
        return value
    finally:
        os.close(fd)
