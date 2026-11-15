"""Single-node batch resource opt-in, not a scheduler or memory reservation.

The budget remains an aggregate current-UID RSS ceiling. Expansion normally
needs finite cgroup-v2 evidence; BSCC 4090 operators may explicitly warn on
unavailable introspection. Default CPU use tolerates absent cgroups.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import time

GIB = 1024 ** 3
DEFAULT_RSS_LIMIT_BYTES = 36 * GIB
MAX_RSS_LIMIT_BYTES = 128 * GIB
PROC_CGROUP = Path('/proc/self/cgroup')
PROC_MOUNTINFO = Path('/proc/self/mountinfo')


def _positive_int(value, name, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f'{name} must be a positive integer <= {maximum}')


@dataclass(frozen=True)
class BatchResourceConfig:
    rss_limit_bytes: int
    max_parallel: int
    source: str = 'API BatchResourceConfig'
    source_sha256: str | None = None
    resource_introspection_policy: str = 'strict'

    def __post_init__(self):
        _positive_int(self.rss_limit_bytes, 'rss_limit_bytes', MAX_RSS_LIMIT_BYTES)
        _positive_int(self.max_parallel, 'max_parallel', 6)
        if self.resource_introspection_policy not in ('strict', 'bscc_4090_warn'):
            raise ValueError('resource_introspection_policy must be strict or bscc_4090_warn')

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        data = json.loads(raw)
        required = {'schema', 'rss_limit_bytes', 'max_parallel'}
        if (not isinstance(data, dict) or not required <= set(data)
                or set(data) - required - {'resource_introspection_policy'}
                or data['schema'] != 'mind2act_eval.batch_resources.v1'):
            raise ValueError('invalid batch resource config schema/fields')
        return cls(data['rss_limit_bytes'], data['max_parallel'], str(path), hashlib.sha256(raw).hexdigest(),
                   data.get('resource_introspection_policy', 'strict'))


class ResourceAdmissionError(ValueError):
    def __init__(self, reason, receipt):
        super().__init__(reason)
        self.receipt = dict(receipt, admission='rejected', reason=reason)


def _unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)


def _counter(path, *, limit=False):
    raw = path.read_text().strip()
    if limit and raw == 'max':
        return None
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f'invalid cgroup counter: {path}')
    value = int(raw)
    if value < 0 or (limit and value == 0):
        raise ValueError(f'invalid cgroup counter: {path}')
    return value


def _memory_scope(path, *, mount, leaf):
    """Read one scope; only a proven full v2 root may omit all memory files."""
    try:
        maximum = _counter(path/'memory.max', limit=True)
        high = _counter(path/'memory.high', limit=True)
        current = _counter(path/'memory.current')
        if (_counter(path/'memory.max', limit=True) != maximum
                or _counter(path/'memory.high', limit=True) != high):
            raise ValueError('cgroup memory limit changed during read')
        return dict(path=str(path), limit_bytes=maximum, high_bytes=high, current_bytes=current)
    except FileNotFoundError:
        # The caller already proved a cgroup2 mount with hierarchy root '/'.
        # Missing files in a leaf, an intermediate ancestor, or an ambiguous
        # namespace root never acquire this exception.
        if path != mount or leaf == mount:
            raise
        evidence = {name:(path/name).read_text().split()
                    for name in ('cgroup.controllers', 'cgroup.subtree_control')}
        if any('memory' not in value for value in evidence.values()):
            raise ValueError('full cgroup-v2 root memory controller cannot be verified')
        # Kernel memory.max/high/current are non-root interfaces. Require the
        # complete expected absence, not a generic missing-file → unlimited.
        for name in ('memory.max', 'memory.high', 'memory.current'):
            try:
                _counter(path/name, limit=name != 'memory.current')
            except FileNotFoundError:
                continue
            raise ValueError('partial cgroup root memory interfaces are unsupported')
        if any((path/name).read_text().split() != value for name,value in evidence.items()):
            raise ValueError('cgroup root controller changed during read')
        return dict(path=str(path), limit_bytes=None, high_bytes=None, current_bytes=None,
                    root_controller_evidence=evidence)


def read_memory_snapshot():
    """Inspect membership and every visible cgroup-v2 ancestor, never a chosen fixture.

    Unmappable namespaces, unreadable counters, changing limits and no finite
    visible limit are unsupported. No v1/hardware-total guess substitutes for a
    missing effective limit. Paths are taken only from this process's proc data.
    """
    scopes = []
    try:
        membership = PROC_CGROUP.read_text()
        mounts = PROC_MOUNTINFO.read_text()
        groups = [line[3:] for line in membership.splitlines() if line.startswith('0::')]
        if len(groups) != 1:
            raise ValueError('cgroup-v2 membership unavailable')
        group = PurePosixPath(groups[0])
        if not group.is_absolute() or '..' in group.parts:
            raise ValueError('unmappable cgroup-v2 membership')
        candidates = []
        for line in mounts.splitlines():
            before, sep, after = line.partition(' - ')
            if sep and after.split()[0] == 'cgroup2':
                fields = before.split()
                root = PurePosixPath(_unescape_mount(fields[3]))
                mount = Path(_unescape_mount(fields[4]))
                if group.is_relative_to(root):
                    candidates.append((len(root.parts), mount, root))
        if not candidates:
            raise ValueError('cgroup-v2 mount/membership cannot be mapped')
        # Prefer the widest visible hierarchy so tighter ancestor limits are read.
        _, mount, root = min(candidates, key=lambda item: item[0])
        if root != PurePosixPath('/'):
            raise ValueError('cgroup ancestors outside the visible mount cannot be verified')
        leaf = mount / str(group.relative_to(root))
        path = leaf
        while True:
            scopes.append(_memory_scope(path, mount=mount, leaf=leaf))
            if path == mount:
                break
            if len(scopes) >= 64 or not path.is_relative_to(mount):
                raise ValueError('cgroup hierarchy cannot be bounded')
            path = path.parent
        if PROC_CGROUP.read_text() != membership or PROC_MOUNTINFO.read_text() != mounts:
            raise ValueError('cgroup membership/mounts changed during read')
        for scope in scopes:
            fresh = _memory_scope(Path(scope['path']), mount=mount, leaf=leaf)
            if any(fresh.get(k) != scope.get(k) for k in
                   ('limit_bytes', 'high_bytes', 'root_controller_evidence')):
                raise ValueError('cgroup memory limit changed during snapshot')
            if scope['current_bytes'] is not None:
                scope['current_bytes'] = max(scope['current_bytes'], fresh['current_bytes'])
        if PROC_CGROUP.read_text() != membership or PROC_MOUNTINFO.read_text() != mounts:
            raise ValueError('cgroup membership/mounts changed during snapshot')
        finite = [s for s in scopes if s['limit_bytes'] is not None]
        if not finite:
            raise ValueError('no finite visible cgroup memory limit')
        high_limits = [s['high_bytes'] for s in scopes if s['high_bytes'] is not None]
        return dict(status='verified', controller='cgroup-v2', membership=str(group),
                    mount_root=str(root), visibility='mounted hierarchy only', scopes=scopes,
                    effective_limit_bytes=min(s['limit_bytes'] for s in finite),
                    effective_high_bytes=min(high_limits) if high_limits else None,
                    current_bytes=scopes[0]['current_bytes'],
                    available_bytes=min(s[k]-s['current_bytes'] for s in scopes
                                        for k in ('limit_bytes', 'high_bytes') if s[k] is not None))
    except (OSError, ValueError, IndexError) as exc:
        # Partial scope observations remain facts, never an effective hierarchy
        # limit or an assertion of unlimited memory. Admission sees only nulls.
        return dict(status='unavailable', reason=f'{type(exc).__name__}: {exc}',
                    scopes=scopes, effective_limit_bytes=None, effective_high_bytes=None,
                    current_bytes=None, available_bytes=None)


def resolve_resources(rss_limit, max_parallel, resource_config=None):
    """Validate before any worker launch; caller rechecks after batch.lock acquisition."""
    if resource_config is None:
        _positive_int(rss_limit, 'rss_limit', DEFAULT_RSS_LIMIT_BYTES)
        _positive_int(max_parallel, 'max_parallel', 4)
    else:
        if not isinstance(resource_config, BatchResourceConfig):
            raise ValueError('resource_config must be a BatchResourceConfig')
        # Legacy parameters must be left at their API defaults; one source wins.
        if type(rss_limit) is not int or type(max_parallel) is not int or rss_limit != DEFAULT_RSS_LIMIT_BYTES or max_parallel != 1:
            raise ValueError('resource_config cannot be combined with legacy RSS/parallel overrides')
        rss_limit = resource_config.rss_limit_bytes
        max_parallel = resource_config.max_parallel
    policy = 'strict' if resource_config is None else resource_config.resource_introspection_policy
    bscc_identity = None
    if policy == 'bscc_4090_warn':
        # This is an explicit operator policy for the BSCC 4090 deployment, not
        # automatic GPU-model detection. Reuse the real Slurm/storage binding;
        # the existing allocation UUID/PCI validation still runs at launch.
        from mind2act_eval.gpu_allocation import allocation_identity
        bscc_identity = allocation_identity()
    snapshot = read_memory_snapshot()
    reserve = None if resource_config is None or snapshot.get('status') != 'verified' else max(8 * GIB, snapshot['effective_limit_bytes'] // 16)
    try:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    except OSError:
        affinity = None
    receipt = dict(schema='mind2act_eval.batch_resource_receipt.v1', node=platform.node(),
                   observed_at_unix_s=time.time(), host_cpu_count=os.cpu_count(), affinity_cpu_count=affinity,
                   opt_in=resource_config is not None, rss_limit_bytes=rss_limit, max_parallel=max_parallel,
                   reserve_bytes=reserve, reserve_rule='max(8GiB, effective visible cap/16)',
                   budget_semantics='aggregate UID RSS ceiling, not per-worker or reservation',
                   qualification='resource admission only; native parallel stability unverified',
                   memory=snapshot, config_source=None if resource_config is None else resource_config.source,
                   config_sha256=None if resource_config is None else resource_config.source_sha256,
                   gpu_selection_source='explicit EpisodeJob.gpu; CLI/manifest resolves before batch',
                   parent_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    receipt['resource_introspection_policy'] = policy
    receipt['resource_warnings'] = []
    if bscc_identity is not None:
        receipt['bscc_allocation_identity'] = bscc_identity
    if resource_config is not None:
        if snapshot.get('status') != 'verified':
            if policy != 'bscc_4090_warn':
                raise ResourceAdmissionError('expanded batch resources require verified cgroup-v2 evidence: ' + snapshot.get('reason', 'unknown'), receipt)
            # An unreadable ancestor cannot erase an already observed finite
            # max/high contradiction. Only the proven reserve floor applies:
            # incomplete ancestry cannot establish the cap/16 reserve either.
            observed_headroom = [scope[key] - scope['current_bytes']
                                 for scope in snapshot.get('scopes', [])
                                 for key in ('limit_bytes', 'high_bytes')
                                 if scope.get(key) is not None and scope.get('current_bytes') is not None]
            if observed_headroom and rss_limit + 8 * GIB > min(observed_headroom):
                raise ResourceAdmissionError('batch RSS budget plus minimum reserve exceeds observed cgroup headroom', receipt)
            receipt['resource_warnings'].append(dict(code='cgroup_memory_introspection_unavailable',
                                                    reason=snapshot.get('reason', 'unknown')))
        # Full new UID ceiling plus existing usage and reserve is deliberately
        # conservative: current cgroup pages may already belong to this UID.
        elif rss_limit + reserve > snapshot['available_bytes']:
            raise ResourceAdmissionError('batch RSS budget plus reserve exceeds current cgroup headroom', receipt)
    receipt['admission'] = 'admitted'
    return rss_limit, max_parallel, receipt
