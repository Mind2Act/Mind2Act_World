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

def _canonical_hostname(value):
    return value.lower()


def node_lock_key(hostname):
    return hashlib.sha256(_canonical_hostname(hostname).encode()).hexdigest()


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


