"""Startup domains and an explicit, quiescent legacy-to-node transition.

The marker is a maintenance receipt, never proof that a process has exited.
Until maintenance activates it, all callers retain the legacy global domain.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket

from mind2act_eval.distributed_campaign import _canonical_hostname, node_lock_key

GATE_SCHEMA = 'mind2act_eval.kit_start_gate.v1'
DOMAIN_SCHEMA = 'mind2act_eval.node_startup.v1'
MARKER = 'startup-domain.json'


def read_gate_state(raw):
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError('Kit start gate state is corrupt') from error
    if (not isinstance(value, dict) or set(value) != {'schema', 'state'}
            or value['schema'] != GATE_SCHEMA or value['state'] not in {'active', 'cooling', 'idle'}):
        raise RuntimeError('Kit start gate state is invalid')
    return value['state']


def _marker(root):
    path = root / MARKER
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except FileNotFoundError:
        return None, None
    with os.fdopen(fd, 'rb') as stream:
        raw = stream.read()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError('startup domain maintenance receipt is corrupt') from error
    if (not isinstance(value, dict) or set(value) != {'schema', 'uid', 'evidence', 'legacy'}
            or value['schema'] != DOMAIN_SCHEMA or type(value['uid']) is not int
            or value['uid'] != os.getuid() or not isinstance(value['evidence'], str)
            or not value['evidence'].strip() or not isinstance(value['legacy'], dict)
            or set(value['legacy']) != {'worker-launch', 'kit'}):
        raise RuntimeError('startup domain maintenance receipt is invalid')
    for record in value['legacy'].values():
        if (not isinstance(record, dict) or set(record) != {'sha256', 'state'}
                or record['state'] not in {None, 'idle', 'active', 'cooling'}
                or not isinstance(record['sha256'], str) or len(record['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in record['sha256'])):
            raise RuntimeError('startup domain legacy receipt is invalid')
    return value, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class StartupPaths:
    root: Path
    hostname: str
    node_key: str
    worker_launch: Path
    kit: Path
    marker_sha256: str | None
    inherited_worker_cooldown: bool = False
    inherited_kit_cooldown: bool = False

    @property
    def node_scoped(self):
        return self.marker_sha256 is not None


def startup_paths(lock_root):
    """Derive runtime paths solely from the actual kernel hostname, never env/manifest identity."""
    root = Path(lock_root).resolve()
    hostname = _canonical_hostname(socket.gethostname())
    key = node_lock_key(hostname)
    marker, digest = _marker(root)
    directory = root / 'startup-nodes' / key if marker is not None else root
    return StartupPaths(root, hostname, key, directory / 'worker-launch.lock', directory / 'kit.lock', digest,
        marker is not None and marker['legacy']['worker-launch']['state'] in {'active', 'cooling'},
        marker is not None and marker['legacy']['kit']['state'] in {'active', 'cooling'})


def startup_lock_paths(lock_root, hostnames):
    """Startup-only maintenance locks; caller also holds global batch/node/queue/attempt guards.

    Foreign hostnames are accepted only here to cover all reviewed nodes during
    maintenance. Runtime callers always use startup_paths()' actual hostname.
    Paths may not exist yet; create without truncation and use NOFOLLOW + EX/NB.
    """
    root = Path(lock_root).resolve()
    paths = [root / 'worker-launch.lock', root / 'kit.lock']
    if _marker(root)[0] is not None:
        for key in sorted({node_lock_key(host) for host in hostnames}):
            directory = root / 'startup-nodes' / key
            paths.extend((directory / 'worker-launch.lock', directory / 'kit.lock'))
    return tuple(paths)


def verify_startup_domain(paths):
    if _marker(paths.root)[1] != paths.marker_sha256:
        raise RuntimeError('startup domain changed while waiting; retry from the reviewed generation')


@contextmanager
def startup_compatibility(paths, *, cancel_event=None):
    """Keep old global EX startup writers blocked throughout a new runtime.

    Parent holds these SH guards until ready; child holds them until teardown.
    A driver holds them for its lifetime. All acquire worker-launch then Kit.
    """
    if not paths.node_scoped:
        yield
        return
    from mind2act_eval.sim import startup_gate
    with ExitStack() as guards:
        for path in (paths.root / 'worker-launch.lock', paths.root / 'kit.lock'):
            guards.enter_context(startup_gate(path, shared=True, cancel_event=cancel_event))
        verify_startup_domain(paths)
        yield


def activate_node_startup(lock_root, *, all_prior_owners_stopped, evidence):
    """Maintenance only: caller must verify all old owners exited and disable old launchers.

    Free gates cannot prove that an old ready worker exited. This explicit
    operator attestation is required in addition to nonblocking exclusive locks.
    Existing state bytes are read, hashed and preserved; no timestamps migrate.
    """
    if all_prior_owners_stopped is not True or not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('node startup activation requires reviewed exit evidence for all prior owners')
    import fcntl
    root = Path(lock_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        descriptors = {}
        for name in ('batch.lock', 'worker-launch.lock', 'kit.lock'):
            fd = os.open(root / name, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            locks.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors[name] = fd
        if _marker(root)[0] is not None:
            raise RuntimeError('node startup is already activated; maintenance receipt is immutable')
        legacy = {}
        for name in ('worker-launch', 'kit'):
            fd = descriptors[name + '.lock']
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 65537)
            if len(raw) > 65536:
                raise RuntimeError('legacy startup gate state is oversized')
            legacy[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), state=read_gate_state(raw))
        value = dict(schema=DOMAIN_SCHEMA, uid=os.getuid(), evidence=evidence, legacy=legacy)
        payload = (json.dumps(value, sort_keys=True) + '\n').encode()
        fd = os.open(root / MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return value
