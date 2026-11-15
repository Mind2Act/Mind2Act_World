"""Shared, file-locked episode queue for long-lived per-node batch drivers."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import gzip
import json
from math import isfinite
import os
from pathlib import Path
import re
import signal
import threading
import time
import uuid

from mind2act_eval.batch_resources import BatchResourceConfig, DEFAULT_RSS_LIMIT_BYTES, ResourceAdmissionError, resolve_resources
from mind2act_eval.distributed_campaign import NodeIdentity, _canonical_hostname, node_identity

GIB = 1024 ** 3
_QUEUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_OPTIONS = {
    "spec", "spec_sha256", "asset_root", "backend", "adapter", "adapter_config",
    "adapter_config_sha256", "h5", "replay_metadata", "h5_sha256",
    "replay_metadata_sha256", "physics_ticks", "wall_s", "variation_manifest", "video_hz",
    "policy_kind", "budget_version",
}


@dataclass(frozen=True)
class QueueNodeConfig:
    label: str
    hostname: str
    machine_id_sha256: str
    gpu_indices: tuple[int, ...]
    rss_limit_bytes: int
    initial_parallel: int
    stable_parallel: int
    stable_window_s: int
    resource_config: Path | None = None
    input_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SharedQueue:
    queue_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: bytes
    output_root: Path
    lock_root: Path
    state_root: Path
    jobs: tuple[dict, ...]
    nodes: dict[str, QueueNodeConfig]
    node: QueueNodeConfig
    profile: object | None = None
    rolling: dict | None = None


@dataclass
class QueueClaim:
    queue: SharedQueue
    job_id: str
    options: dict
    episode_key: dict
    attempt_number: int
    token: str
    directory: Path
    gpu: int | None
    lock_fd: int
    inputs: dict | None = None

    def close(self):
        """Close this process's lease descriptor; an inherited worker fd keeps it held."""
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _process_birth(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def failure_cause(result):
    error = result.get('error')
    if result.get('infra_valid') is not False or not isinstance(error, str) or not error.strip():
        return None
    text = [line.strip() for line in error.splitlines() if line.strip()][-1]
    text = re.sub(r'\b(pid|birth|starttime|gpu)(\s*[=: ]\s*)\d+\b', r'\1\2#', text, flags=re.I)
    text = re.sub(r'0x[0-9a-fA-F]+', '0x#', text)
    text = re.sub(r'\battempt-\d+\b', 'attempt-#', text)
    text = re.sub(r'\bline \d+\b', 'line #', text)
    value = dict(termination=result.get('termination'), error=' '.join(text.split()))
    return dict(sha256=hashlib.sha256(_json_bytes(value)).hexdigest(), **value)


def _failure_hold(row):
    if len(row['attempts']) < 2:
        return
    previous, current = row['attempts'][-2:]
    a = previous.get('failure_cause') or failure_cause(previous.get('result') or {})
    b = current.get('failure_cause') or failure_cause(current.get('result') or {})
    if a is not None and b is not None and a['sha256'] == b['sha256']:
        row['failure_hold'] = dict(status='held', generation=row['generation'],
            consecutive_failures=2, cause=b, attempt_tokens=[previous['token'], current['token']],
            release='publish a genuinely changed generation with evidence')


def _write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    os.replace(temp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _under_shared(path):
    shared = Path("/shared").resolve(strict=True)
    return Path(path).resolve(strict=False).is_relative_to(shared)


def _validate_absolute_path(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"queue {name} must be an absolute path without parent traversal")
    return Path(value)


def load_queue(path, *, expected_sha256, local_identity=None, require_shared=True):
    """Load a pinned global queue and select this node's lane configuration."""
    path = Path(path).resolve(strict=True)
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256) or digest != expected_sha256:
        raise ValueError("shared queue manifest digest mismatch")
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("shared queue manifest is invalid JSON") from exc
    rolling_schema = isinstance(raw, dict) and raw.get('schema') == 'mind2act_eval.pro_rolling_queue.v1'
    required = ({'schema', 'queue_id', 'state_root', 'jobs', 'nodes', 'rolling'} if rolling_schema else
                {"schema", "queue_id", "output_root", "lock_root", "state_root", "jobs", "nodes"})
    if (not isinstance(raw, dict) or set(raw) != required
            or raw['schema'] not in {'mind2act_eval.shared_queue.v1', 'mind2act_eval.pro_rolling_queue.v1'}):
        raise ValueError("invalid shared queue manifest schema/fields")
    queue_id = raw["queue_id"]
    if not isinstance(queue_id, str) or not _QUEUE_ID.fullmatch(queue_id):
        raise ValueError("queue_id must be a safe path component")
    state_root = _validate_absolute_path(raw["state_root"], "state_root")
    from mind2act_eval.cluster_profile import load_cluster_profile
    profile = load_cluster_profile()
    if rolling_schema:
        if profile is None:
            raise ValueError('rolling dispatch requires a pinned PRO profile')
        profile.verify_queue_control(path)
        profile.verify_queue_control(state_root)
        candidates = raw['nodes']
        host = _canonical_hostname(profile.hostname)
        if not isinstance(candidates, list):
            raise ValueError('rolling nodes must be a list')
        selected = [n for n in candidates if isinstance(n, dict) and n.get('hostname') == host]
        if len(selected) != 1:
            raise ValueError('rolling queue must identify this actual PRO hostname exactly once')
        output_root = _validate_absolute_path(selected[0].get('output_root'), 'node output_root')
        lock_root = _validate_absolute_path(selected[0].get('lock_root'), 'node lock_root')
    else:
        output_root = _validate_absolute_path(raw["output_root"], "output_root")
        lock_root = _validate_absolute_path(raw["lock_root"], "lock_root")
    if profile is not None:
        profile.verify_storage(output_root, lock_root)
        if not rolling_schema:
            profile.verify_storage(state_root, lock_root)
            profile.verify_storage(path, lock_root)
        if profile.job_manifest is None:
            raise ValueError('PRO shared queue requires its frozen catalog')
    elif require_shared and not all(_under_shared(p) for p in (path, output_root, lock_root, state_root)):
        raise ValueError("production queue manifest, outputs, state, and locks must resolve under /shared")
    rolling = raw.get("rolling")
    if "rolling" in raw:
        from mind2act_eval.pro_rolling_queue import validate_manifest
        if not isinstance(rolling, dict):
            raise ValueError("rolling dispatch config must be an object")
        validate_manifest(rolling, profile)
        from mind2act_eval.cluster_profile import pinned_json
        catalog = pinned_json(profile.job_manifest["path"], profile.job_manifest["sha256"])
        if not isinstance(catalog, list) or not 1 <= len(catalog) <= 64:
            raise ValueError("rolling frozen catalog must contain 1..64 real cases")
    jobs_raw = raw["jobs"]
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("shared queue jobs must be a nonempty list")
    jobs, job_ids, episode_keys = [], set(), set()
    for item in jobs_raw:
        if (not isinstance(item, dict) or not {"job_id", "episode_key", "options"} <= set(item)
                or set(item) - {"job_id", "episode_key", "options", "initial_status", "generation", "inputs", "history"}
                   - ({'eligible_nodes', 'eligible_gpu'} if rolling is not None else set())):
            raise ValueError("invalid shared queue job fields")
        initial_status = item.get("initial_status", "ready")
        if initial_status not in {"ready", "blocked", "completed"}:
            raise ValueError("invalid initial queue status")
        job_id = item["job_id"]
        key = item["episode_key"]
        options = item["options"]
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id) or job_id in job_ids:
            raise ValueError("queue job IDs must be unique safe path components")
        if (not isinstance(key, dict) or not {"task_id", "instance_id", "difficulty"} <= set(key)
                or set(key) - {"task_id", "instance_id", "difficulty", "variation_sha256"}
                or any(not isinstance(v, str) or not v for v in key.values())):
            raise ValueError("queue episode_key must contain task_id, instance_id, and difficulty")
        if "variation_sha256" in key and not _SHA256.fullmatch(key["variation_sha256"]):
            raise ValueError("queue variation identity must be lowercase SHA256")
        key_tuple = (*tuple(key[k] for k in ("task_id", "instance_id", "difficulty")), key.get("variation_sha256"))
        if key_tuple in episode_keys:
            raise ValueError("shared queue contains a duplicate episode")
        if initial_status != "ready":
            if not isinstance(options, dict):
                raise ValueError("blocked/completed queue payload must be an object")
            job_ids.add(job_id)
            episode_keys.add(key_tuple)
            jobs.append(dict(item))
            continue
        if (not isinstance(options, dict) or set(options) - _JOB_OPTIONS
                or not {"spec", "spec_sha256", "asset_root", "backend", "adapter", "physics_ticks", "wall_s"} <= set(options)):
            raise ValueError("invalid or incomplete shared queue job options")
        if not isinstance(options["spec"], str) or not Path(options["spec"]).is_absolute():
            raise ValueError("queued spec path must be absolute")
        if not isinstance(options["spec_sha256"], str) or not _SHA256.fullmatch(options["spec_sha256"]):
            raise ValueError("queued spec_sha256 must be lowercase SHA256")
        variation_manifest = options.get("variation_manifest")
        if variation_manifest is not None and (not isinstance(variation_manifest, str)
                or not variation_manifest or Path(variation_manifest).is_absolute()
                or ".." in Path(variation_manifest).parts):
            raise ValueError("queued variation_manifest must be relative to asset_root")
        if (not isinstance(options["backend"], str) or options["backend"] not in {"isaac", "demo"}
                or not isinstance(options["adapter"], str)
                or type(options["physics_ticks"]) is not int or options["physics_ticks"] <= 0
                or type(options["wall_s"]) not in (int, float) or not isfinite(options["wall_s"])
                or options["wall_s"] <= 0):
            raise ValueError("queued backend, adapter, physics_ticks, or wall_s is invalid")
        if 'video_hz' in options and (type(options['video_hz']) is not int or options['video_hz'] <= 0):
            raise ValueError("queued video_hz must be a positive integer")
        if options.get('policy_kind') is not None or options.get('budget_version') is not None:
            from mind2act_eval.runner import EpisodeLimits
            EpisodeLimits(options['physics_ticks'], options['wall_s'],
                          options.get('policy_kind'), options.get('budget_version'))
            if options.get('h5'):
                raise ValueError('versioned policy budgets require a model consumer')
        for path_name in ("spec", "asset_root", "h5", "replay_metadata", "adapter_config"):
            value = options.get(path_name)
            if value is None:
                if path_name in {"spec", "asset_root"}:
                    raise ValueError(f"queued {path_name} path is required")
                continue
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"queued {path_name} path must be absolute")
            if profile is None and require_shared and not _under_shared(value):
                raise ValueError(f"queued {path_name} must resolve under /shared")
        for path_key, digest_key in (("h5", "h5_sha256"),
                                     ("replay_metadata", "replay_metadata_sha256"),
                                     ("adapter_config", "adapter_config_sha256")):
            if options.get(path_key) and (not isinstance(options.get(digest_key), str)
                                          or not _SHA256.fullmatch(options[digest_key])):
                raise ValueError(f"queued {path_key} requires a pinned {digest_key}")
        if options.get("gpu") is not None or options.get("retries") is not None:
            raise ValueError("queue allocates GPUs locally and owns the single infrastructure retry")
        job_ids.add(job_id)
        episode_keys.add(key_tuple)
        jobs.append({**item, "job_id": job_id, "episode_key": dict(key), "options": dict(options)})

    protocols = {(item['options'].get('policy_kind'), item['options'].get('budget_version'))
                 for item in jobs if item.get('initial_status', 'ready') == 'ready'}
    if len(protocols) > 1:
        raise ValueError('separate queues are required for different policy budget versions')
    if rolling is not None and len(jobs) > 192:
        raise ValueError("rolling catalog is bounded to 192 real cases")
    nodes_raw = raw["nodes"]
    single_lane = profile is not None and rolling is not None and isinstance(nodes_raw, list) and len(nodes_raw) == 1
    if not isinstance(nodes_raw, list) or (not single_lane and len(nodes_raw) != (3 if profile is not None else 4)):
        raise ValueError("shared queue requires exactly four node configurations")
    if single_lane:
        label = nodes_raw[0].get('label') if isinstance(nodes_raw[0], dict) else None
        if label not in {'A', 'B', 'C'}:
            raise ValueError('single-node rolling queue requires a declared PRO label')
        labels_expected = {label: None}
    else:
        labels_expected = ({'A': None, 'B': None, 'C': None} if profile is not None
                           else {"N1": 3, "N2": 5, "N3": 5, "N4": 4})
    nodes, hosts, labels = {}, set(), set()
    for item in nodes_raw:
        fields = {"label", "hostname", "machine_id_sha256", "gpu_indices", "rss_limit_bytes",
                  "initial_parallel", "stable_parallel", "stable_window_s"}
        if rolling is not None:
            fields |= {'output_root', 'lock_root', 'input_roots'}
        if not isinstance(item, dict) or not fields <= set(item) or set(item) - fields - {"resource_config"}:
            raise ValueError("invalid shared queue node config fields")
        if rolling is not None:
            roots = item['input_roots']
            if not isinstance(roots, list) or not 1 <= len(roots) <= 32:
                raise ValueError('rolling node requires 1..32 explicit immutable input roots')
            input_roots = tuple(_validate_absolute_path(p, 'node input root') for p in roots)
            for input_path in input_roots:
                profile.verify_control_input(input_path)
            for key in ('output_root', 'lock_root'):
                worker_path = _validate_absolute_path(item[key], 'node ' + key).resolve()
                if (worker_path.is_relative_to(profile.queue_control_root)
                        or profile.queue_control_root.is_relative_to(worker_path)):
                    raise ValueError('rolling control root must stay separate from every node output/lock domain')
        label, hostname, machine_sha = item["label"], item["hostname"], item["machine_id_sha256"]
        if label not in labels_expected or label in labels:
            raise ValueError("queue node labels must be unique N1/N2/N3/N4")
        if not isinstance(hostname, str) or _canonical_hostname(hostname) != hostname or hostname in hosts:
            raise ValueError("queue node hostnames must be unique canonical hostnames")
        if not isinstance(machine_sha, str) or not _SHA256.fullmatch(machine_sha):
            raise ValueError("queue node machine_id_sha256 must be lowercase SHA256")
        gpu_indices = item["gpu_indices"]
        if (not isinstance(gpu_indices, list) or any(type(g) is not int or g < 0 for g in gpu_indices)
                or len(set(gpu_indices)) != len(gpu_indices)):
            raise ValueError("queue GPU indices must be unique nonnegative integers")
        if profile is not None and any(g > 7 for g in gpu_indices):
            raise ValueError("PRO queue GPU indices must belong to the eight-card device pool")
        initial, target, stable_s, rss = (item["initial_parallel"], item["stable_parallel"],
                                           item["stable_window_s"], item["rss_limit_bytes"])
        experiment = (profile is not None and getattr(profile, 'concurrency_experiment', None)
                      == 'g05-clean-concurrency-4-6-8-v1')
        experiment_target = (experiment and len(gpu_indices) == 1
                             and initial == target == profile.max_workers_per_gpu
                             and target in (3, 4, 6, 8))
        valid_pro = (profile is not None and type(initial) is int and type(target) is int and 1 <= initial <= target <= 24
                     and ((rolling is None and initial == target <= 8 and stable_s == 0)
                          or (rolling is not None and gpu_indices and initial % len(gpu_indices) == 0
                              and target % len(gpu_indices) == 0
                              and (experiment_target or (1 <= initial // len(gpu_indices) <= 2
                              and initial // len(gpu_indices) <= target // len(gpu_indices) <= 3))
                              and stable_s == 1200))
                     and type(stable_s) is int and type(rss) is int
                     and 8 * GIB <= rss <= min(profile.max_campaign_rss_bytes,
                                              profile.rss_bytes_per_worker * max(target, len(gpu_indices)))
                     and not item.get('resource_config'))
        valid_l20 = (type(initial) is int and type(target) is int and 1 <= initial <= target
                     and target == labels_expected[label] and type(stable_s) is int and 0 <= stable_s <= 1200
                     and type(rss) is int and rss == 96 * GIB)
        if not (valid_pro if profile is not None else valid_l20):
            raise ValueError("queue node runtime exceeds the authorized target/96GiB/ramp bounds")
        if rolling is None and target > len(gpu_indices) and any(job["options"].get("backend", "isaac") == "isaac" for job in jobs):
            raise ValueError("queue node needs one distinct physical GPU per target lane")
        hosts.add(hostname)
        labels.add(label)
        resource_path = None
        if item.get("resource_config"):
            resource_path = _validate_absolute_path(item["resource_config"], "node resource_config")
            if require_shared and not _under_shared(resource_path):
                raise ValueError("node resource config must resolve under /shared")
        nodes[hostname] = QueueNodeConfig(label, hostname, machine_sha, tuple(gpu_indices), rss,
                                          initial, target, stable_s,
                                          resource_path, input_roots if rolling is not None else ())
    if labels != set(labels_expected):
        raise ValueError("queue node config must include N1, N2, N3, and N4")
    # PRO already verified the exact hostname and raw machine-id bytes. L20's
    # legacy identity hashes normalized text and must retain its own semantics.
    identity = (NodeIdentity(_canonical_hostname(profile.hostname), profile.machine_id_sha256)
                if profile is not None and local_identity is None else
                node_identity() if local_identity is None else local_identity)
    if not isinstance(identity, NodeIdentity):
        raise ValueError("local queue node identity is invalid")
    node = nodes.get(identity.hostname)
    if node is None:
        raise ValueError("this hostname is not assigned to the shared queue")
    if node.machine_id_sha256 != identity.machine_id_sha256:
        raise ValueError("queue machine_id_sha256 does not match this hostname")
    if profile is not None and (node.hostname != _canonical_hostname(profile.hostname) or node.machine_id_sha256 != profile.machine_id_sha256
            or node.gpu_indices != tuple(d['index'] for d in profile.devices)
            or node.stable_parallel != profile.max_parallel or node.rss_limit_bytes != profile.rss_limit_bytes):
        raise ValueError('PRO queue node differs from its inspected profile/pool/budget')
    if rolling is not None:
        if rolling["controller_hostname"] not in nodes:
            raise ValueError("rolling controller must belong to this queue")
        if node.stable_parallel // len(node.gpu_indices) != profile.max_workers_per_gpu:
            raise ValueError("rolling hard GPU ceiling differs from frozen profile")
        from mind2act_eval.cli import _queue_job
        templates = profile.frozen_jobs()
        def covered(path, roots):
            return any(Path(path).resolve().is_relative_to(root.resolve()) for root in roots)
        if any(not covered(path, node.input_roots) for path in profile.queue_input_paths()):
            raise ValueError('rolling node input roots omit an actual profile/code/authorization input')
        local_jobs = []
        node_counts = {host: 0 for host in nodes}
        for entry in jobs:
            eligible = entry.get('eligible_nodes')
            gpu = entry.get('eligible_gpu')
            if (not isinstance(eligible, list) or len(eligible) != 1 or eligible[0] not in nodes
                    or type(gpu) is not int or gpu not in nodes[eligible[0]].gpu_indices):
                raise ValueError('rolling case requires one frozen eligible node and physical GPU route')
            node_counts[eligible[0]] += 1
            for key in ('asset_root', 'spec', 'h5', 'replay_metadata', 'adapter_config'):
                if entry['options'].get(key):
                    profile.verify_control_input(entry['options'][key])
                    if not covered(entry['options'][key], nodes[eligible[0]].input_roots):
                        raise ValueError('rolling node input roots omit a routed case input')
            if entry['options'].get('variation_manifest'):
                profile.verify_control_input(Path(entry['options']['asset_root']) / entry['options']['variation_manifest'])
            if entry.get('initial_status', 'ready') != 'ready':
                raise ValueError('rolling intake requires actual ready cases, not placeholder jobs')
            if node.hostname in eligible:
                local_jobs.append(_queue_job(entry, gpu))
        if any(count > 64 for count in node_counts.values()):
            raise ValueError('rolling host catalog is bounded to 64 real cases')
        if len(local_jobs) != len(templates) or any(job not in templates for job in local_jobs):
            raise ValueError('rolling local cases differ from the complete frozen catalog/routes')
    return SharedQueue(queue_id, path, digest, raw_bytes, output_root, lock_root, state_root,
                       tuple(jobs), nodes, node, profile, rolling)


class QueueStore:
    """Atomic shared queue state; a per-job flock protects each live attempt."""
    def __init__(self, queue: SharedQueue):
        self.queue = queue
        self.state_path = queue.state_root / f"queue-{queue.queue_id}.json"
        self.lock_path = queue.state_root / f"queue-{queue.queue_id}.lock"
        self.marker_path = queue.state_root / f"queue-{queue.queue_id}.initialized.json"
        self.checkpoint_root = queue.state_root / f"queue-{queue.queue_id}.checkpoints"

    @property
    def max_attempts(self):
        return 1 if self.queue.rolling is not None else 2

    def _initial_state(self):
        state = {"schema": "mind2act_eval.queue_state.v1", "queue_id": self.queue.queue_id,
                "manifest_sha256": self.queue.manifest_sha256, "created_at": _now(),
                "updated_at": _now(), "drivers": {},
                "jobs": {j["job_id"]: {"status": {"ready": "queued", "blocked": "blocked", "completed": "reviewed"}[j.get("initial_status", "ready")],
                         "generation": j.get("generation", "1"), "definition": j,
                         "history": j.get("history", []), "attempts": [], "result": None}
                         for j in self.queue.jobs}}
        if self.queue.rolling is not None:
            from mind2act_eval.pro_rolling_queue import initial_controls
            state.update(dispatch=initial_controls(self.queue),
                rolling_contract_sha256=self._rolling_contract_sha256(), catalog_revision=0,
                catalog_manifests={self.queue.manifest_sha256: dict(job_ids=[j['job_id'] for j in self.queue.jobs])})
        return state

    @staticmethod
    def _exclusive_file(path, payload):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def initialize(self, *, evidence):
        """Explicit first creation only; an existing lock is a permanent tombstone.

        Never retry this to recover an absent state. Partial initialization also
        requires operator investigation; no existing queue file is overwritten.
        """
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("queue initialization requires explicit evidence")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError as exc:
            raise RuntimeError("queue lock already exists; initialization cannot reset an existing campaign") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                existing = os.open(self.state_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            except FileNotFoundError:
                pass
            else:
                os.close(existing)
                raise RuntimeError("queue state already exists; initialization cannot replace it")
            state = self._initial_state()
            self._exclusive_file(self.marker_path, _json_bytes({
                "schema": "mind2act_eval.queue_initialized.v1", "queue_id": self.queue.queue_id,
                "manifest_sha256": self.queue.manifest_sha256, "evidence": evidence,
                "created_at": state["created_at"], "pid": os.getpid()}))
            self._checkpoint(state, before_sha256=None, operation="initialize")
            self._exclusive_file(self.state_path, _json_bytes(state))
        finally:
            os.close(fd)
        return state

    def _rolling_contract_sha256(self):
        raw = json.loads(self.queue.manifest_bytes)
        return hashlib.sha256(_json_bytes({k: v for k, v in raw.items() if k != 'jobs'})).hexdigest()

    def _read_state(self, *, allow_unregistered_catalog=False):
        import stat
        try:
            fd = os.open(self.state_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError as exc:
            raise RuntimeError("queue state missing; automatic initialization/recovery is forbidden") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RuntimeError("queue state is not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                payload = stream.read()
            state = json.loads(payload)
        finally:
            os.close(fd)
        if (not isinstance(state, dict) or state.get("schema") != "mind2act_eval.queue_state.v1"
                or state.get("queue_id") != self.queue.queue_id):
            raise ValueError("shared queue state does not match its pinned manifest")
        if self.queue.rolling is None:
            if (state.get("manifest_sha256") != self.queue.manifest_sha256
                    or set(state.get("jobs", {})) != {j["job_id"] for j in self.queue.jobs}):
                raise ValueError("shared queue state does not match its pinned manifest")
        else:
            if (state.get('rolling_contract_sha256') != self._rolling_contract_sha256()
                    or set(state.get('dispatch', {})) != set(self.queue.nodes)
                    or type(state.get('catalog_revision')) is not int
                    or state.get('manifest_sha256') not in state.get('catalog_manifests', {})):
                raise ValueError('rolling queue lifetime/node/storage contract changed')
            if not allow_unregistered_catalog:
                record = state.get('catalog_manifests', {}).get(self.queue.manifest_sha256)
                if record is None or record['job_ids'] != [j['job_id'] for j in self.queue.jobs]:
                    raise ValueError('rolling catalog is not durably registered')
                if any(state.get('jobs', {}).get(j['job_id'], {}).get('definition') != j for j in self.queue.jobs):
                    raise ValueError('rolling immutable case definition changed')
        return state, hashlib.sha256(payload).hexdigest()

    def _validate_marker(self, state):
        try:
            fd = os.open(self.marker_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError as exc:
            raise RuntimeError("queue initialization marker missing; reviewed legacy adoption required") from exc
        try:
            with os.fdopen(fd, "rb", closefd=False) as stream:
                marker = json.load(stream)
        finally:
            os.close(fd)
        if (marker.get("schema") != "mind2act_eval.queue_initialized.v1"
                or marker.get("queue_id") != self.queue.queue_id
                or marker.get("manifest_sha256") != (state.get('manifest_sha256')
                    if self.queue.rolling is not None else self.queue.manifest_sha256)
                or marker.get("created_at") != state.get("created_at")):
            raise RuntimeError("queue lifetime identity differs from its durable initialization marker")

    def adopt_legacy(self, *, expected_state_sha256, evidence):
        """Install a guard on reviewed legacy bytes; never restore or replace them.

        Canonical batch/node gates and all job leases refuse live drivers or
        any still-owned attempt. Operator evidence
        and exact bytes are mandatory; a reset state cannot be repaired this way.
        """
        if (not isinstance(evidence, str) or not evidence.strip()
                or not isinstance(expected_state_sha256, str) or not _SHA256.fullmatch(expected_state_sha256)):
            raise ValueError("legacy adoption requires exact state SHA256 and evidence")
        from mind2act_eval.distributed_campaign import node_lock_key
        fd = None
        gates = []
        leases = []
        try:
            gate_paths = [self.queue.lock_root / "batch.lock"] + [
                self.queue.lock_root / f"batch-node-{node_lock_key(hostname)}.lock"
                for hostname in sorted(self.queue.nodes)]
            for path in gate_paths:
                gate = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                gates.append(gate)
                fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd = os.open(self.lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for job in self.queue.jobs:
                lease = self._try_job_lock(job["job_id"])
                if lease is None:
                    raise RuntimeError("legacy adoption refuses a live attempt lease")
                leases.append(lease)
            state, digest = self._read_state()
            if digest != expected_state_sha256:
                raise RuntimeError("legacy adoption state changed from the reviewed SHA256")
            if any(row["status"] == "running" for row in state["jobs"].values()):
                raise RuntimeError("legacy adoption requires resolved stopped attempts, not running records")
            self._checkpoint(state, before_sha256=digest, operation="adopt_legacy")
            if self._read_state()[1] != digest:
                raise RuntimeError("legacy adoption state changed before guard publication")
            self._exclusive_file(self.marker_path, _json_bytes({
                "schema": "mind2act_eval.queue_initialized.v1", "queue_id": self.queue.queue_id,
                "manifest_sha256": self.queue.manifest_sha256, "created_at": state["created_at"],
                "adopted_state_sha256": digest, "evidence": evidence, "pid": os.getpid()}))
        finally:
            for lease in leases:
                os.close(lease)
            if fd is not None:
                os.close(fd)
            for gate in reversed(gates):
                os.close(gate)

    def _checkpoint(self, state, *, before_sha256, operation):
        import socket
        payload = _json_bytes(state)
        digest = hashlib.sha256(payload).hexdigest()
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        record = {"schema": "mind2act_eval.queue_checkpoint.v1", "state_sha256": digest,
                  "before_sha256": before_sha256, "operation": operation,
                  "writer": {"hostname": socket.gethostname(), "pid": os.getpid(),
                             "process_birth": _process_birth(os.getpid()), "uid": os.getuid(),
                             "code_sha": os.environ.get("MIND2ACT_EVAL_CODE_SHA")},
                  "state": state}
        # A prepared checkpoint is evidence, not automatic authorization to
        # restore or proof the canonical replace subsequently committed.
        path = self.checkpoint_root / f"{digest}-{uuid.uuid4().hex}.json.gz"
        self._exclusive_file(path, gzip.compress(_json_bytes(record), mtime=0))

    def _job_lock_path(self, job_id):
        return self.queue.state_root / f"queue-{self.queue.queue_id}-{job_id}.lock"

    @contextmanager
    def _transaction(self, *, write=True, operation="transaction", maintenance_locks=(), allow_unregistered_catalog=False):
        if self.queue.rolling is not None:
            self.queue.profile.verify_queue_control(self.queue.state_root)
            for node in self.queue.nodes.values():
                for path in node.input_roots:
                    self.queue.profile.verify_control_input(path)
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.lock_path, flags)
        except FileNotFoundError as exc:
            raise RuntimeError("queue lock missing; explicit first initialization or reviewed recovery required") from exc
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            state, before_sha256 = self._read_state(allow_unregistered_catalog=allow_unregistered_catalog)
            self._validate_marker(state)
            before_content = _json_bytes(state)
            yield state
            if write and _json_bytes(state) != before_content:
                state["updated_at"] = _now()
                self._checkpoint(state, before_sha256=before_sha256, operation=operation)
                # Detect a replaced lock or an outside writer before publishing.
                held = os.fstat(fd)
                named = os.stat(self.lock_path, follow_symlinks=False)
                if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                    raise RuntimeError("queue lock identity changed during transaction")
                if self._read_state(allow_unregistered_catalog=allow_unregistered_catalog)[1] != before_sha256:
                    raise RuntimeError("queue state changed outside its held lock")
                self._validate_marker(state)
                for path, maintenance_fd in maintenance_locks:
                    held, named = os.fstat(maintenance_fd), os.stat(path, follow_symlinks=False)
                    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                        raise RuntimeError('maintenance lock identity changed before queue commit')
                _write_atomic(self.state_path, state)
                if self._read_state()[1] != hashlib.sha256(_json_bytes(state)).hexdigest():
                    raise RuntimeError("queue state readback differs after atomic replace")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _try_job_lock(self, job_id):
        path = self._job_lock_path(job_id)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd

    def _recover_or_fail(self, row, job):
        previous = row["attempts"][-1]
        directory = Path(previous.get("directory", ""))
        if self.queue.rolling is not None and (directory / 'batch-exit.json').is_file():
            receipt = json.loads((directory / 'batch-exit.json').read_text())
            if (receipt.get('dispatch_deferred') is True and receipt.get('worker_started') is False
                    and receipt.get('worker_exit_verified') is True
                    and previous.get('child_started_at_unix_s') is None):
                row['history'].append(dict(kind='recovered_before_start', claim=row['attempts'].pop(),
                                           exit_receipt=receipt))
                row.update(status='queued', result=None)
                return True
        if (previous.get('hardware') == 'pro5000' or job['options'].get('backend', 'isaac') == 'isaac'
                or (directory/'batch-exit.json').is_file()):
            if json.loads((directory/'batch-exit.json').read_text()).get('worker_exit_verified') is not True:
                raise ValueError('worker exit unknown; recovery cannot manufacture an outcome/retry')
        try:
            from mind2act_eval.contracts import ConsumerKind, InstanceSpec, to_json
            from mind2act_eval.sealing import validate_worker_seal
            result, _receipt = validate_worker_seal(directory, durable=True)
            saved_spec_path = directory / "spec.json"
            source = Path(job["options"]["spec"])
            source_bytes = source.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != job["options"]["spec_sha256"]:
                raise ValueError("pinned queue spec changed during attempt recovery")
            saved_spec = InstanceSpec.from_dict(json.loads(saved_spec_path.read_text()))
            saved_spec.assert_binding(InstanceSpec.from_dict(json.loads(source_bytes)))
            expected_key = job["episode_key"]
            if (result.task_id, result.instance_id) != (expected_key["task_id"], expected_key["instance_id"]):
                raise ValueError("sealed result does not match the queue episode key")
            expected_consumer = (ConsumerKind.EXPERT_REPLAY if job["options"].get("h5")
                                 else ConsumerKind.DIAGNOSTIC if job["options"]["backend"] == "demo"
                                 else ConsumerKind.MODEL)
            if result.consumer_kind != expected_consumer:
                raise ValueError("sealed result consumer does not match queued job")
            raw_result = to_json(result)
            previous.update(status="recovered", finished_at=_now(), result=raw_result,
                            recovery="verified worker seal after driver loss")
            row["result"] = raw_result
            if result.infra_valid:
                row["status"] = "await_review"
            elif previous["attempt_number"] < self.max_attempts:
                row["status"] = "queued"
            else:
                row["status"] = "infra_failed"
                _failure_hold(row)
            return True
        except Exception as exc:
            error = f"attempt owner disappeared before queue commit; {type(exc).__name__}: {exc}"
            previous.update(status="infra_failed", finished_at=_now(), error=error,
                            result={"infra_valid": False, "score": None, "error": error})
            if len(row["attempts"]) >= self.max_attempts:
                row["status"] = "infra_failed"
                row["result"] = previous["result"]
                _failure_hold(row)
            else:
                row["status"] = "queued"
            return False

    def claim_next(self, *, hostname, pid, gpu, resource_config_sha256=None):
        if hostname != self.queue.node.hostname or hostname not in self.queue.nodes:
            raise ValueError("queue claim hostname differs from this driver's actual hostname")
        if gpu is not None and gpu not in self.queue.node.gpu_indices:
            raise ValueError("queue claim GPU is not assigned to this node")
        if self.queue.rolling is not None and (type(pid) is not int or pid != os.getpid() or type(gpu) is not int):
            raise ValueError('rolling claims require this actual owner PID and physical GPU')
        with self._transaction(operation="claim_next") as state:
            if self.queue.rolling is not None:
                from mind2act_eval.pro_rolling_queue import control
                current = control(self, state)
                if gpu is None or current['drain']:
                    return None
                owned = [row['attempts'][-1] for row in state['jobs'].values() if row['status'] == 'running']
                if sum(a['owner_hostname'] == hostname and a['gpu'] == gpu for a in owned) >= current['workers_per_gpu']:
                    return None
            for job in self.queue.jobs:
                job_id = job["job_id"]
                if self.queue.rolling is not None and (hostname not in job['eligible_nodes'] or gpu != job['eligible_gpu']):
                    continue
                row = state["jobs"][job_id]
                job = row.get("definition", job)
                if row["status"] in {"await_review", "reviewed", "blocked", "infra_failed"}:
                    continue
                fd = self._try_job_lock(job_id)
                if fd is None:
                    continue
                if row["status"] == "running":
                    # A free lease is not proof the owner has exited. Recovery
                    # requires explicit evidence; never use a timeout to steal.
                    os.close(fd)
                    continue
                if row["status"] != "queued":
                    os.close(fd)
                    continue
                attempt_number = len(row["attempts"]) + 1
                if attempt_number > self.max_attempts:
                    row["status"] = "infra_failed"
                    row["result"] = {"infra_valid": False, "score": None,
                                     "error": "infrastructure retry limit exhausted"}
                    os.close(fd)
                    continue
                token = uuid.uuid4().hex
                generation_dir = hashlib.sha256(str(row["generation"]).encode()).hexdigest()[:16]
                directory = (self.queue.output_root / f"queue-{self.queue.queue_id}" / job_id / generation_dir
                             / f"attempt-{attempt_number:02d}")
                if self.queue.rolling is not None:
                    directory = directory.with_name(directory.name + "-" + token)
                row["status"] = "running"
                row["attempts"].append({"attempt_number": attempt_number, "token": token,
                                        "generation": row["generation"],
                                        "inputs": {"code_sha": os.environ.get("MIND2ACT_EVAL_CODE_SHA"),
                                                   "input_sha256": hashlib.sha256(_json_bytes(job)).hexdigest(),
                                                   "spec_sha256": job["options"].get("spec_sha256"),
                                                   "h5_sha256": job["options"].get("h5_sha256"),
                                                   "metadata_sha256": job["options"].get("replay_metadata_sha256"),
                                                   **job.get("inputs", {})},
                                        "owner_hostname": hostname, "owner_pid": pid, "gpu": gpu,
                                        "hardware": 'pro5000' if self.queue.profile is not None else 'l20',
                                        "profile_sha256": None if self.queue.profile is None else self.queue.profile.sha256,
                                        "owner_starttime": _process_birth(pid),
                                        "resource_config_sha256": resource_config_sha256,
                                        "directory": str(directory), "started_at": _now(),
                                        "status": "running"})
                claim = QueueClaim(self.queue, job_id, job["options"], job["episode_key"],
                                   attempt_number, token, directory, gpu, fd, job.get("inputs", {}))
                return claim
        return None

    def finish(self, claim: QueueClaim, *, result, archive=None):
        try:
            execution = None
            exit_path = claim.directory / 'batch-exit.json'
            if self.queue.profile is not None or claim.options.get('backend', 'isaac') == 'isaac' or exit_path.is_file():
                if json.loads(exit_path.read_text()).get('worker_exit_verified') is not True:
                    raise ValueError('worker exit unknown; retain its original claim')
            if self.queue.profile is not None:
                exit_path = claim.directory / 'batch-exit.json'
                exit_raw = exit_path.read_bytes()
                if json.loads(exit_raw).get('worker_exit_verified') is not True:
                    raise ValueError('PRO worker exit unknown; retain its original claim')
                admission = claim.directory / 'pro-admission.json'
                execution = dict(hostname=self.queue.node.hostname, profile_sha256=self.queue.profile.sha256,
                    code_sha=os.environ.get('MIND2ACT_EVAL_CODE_SHA'), batch_exit_sha256=hashlib.sha256(exit_raw).hexdigest(),
                    admission_sha256=hashlib.sha256(admission.read_bytes()).hexdigest() if admission.is_file() else None)
                if result.get('infra_valid') is True and execution['admission_sha256'] is None:
                    raise ValueError('PRO valid result requires actual GPU admission evidence')
            if not isinstance(result, dict) or type(result.get("infra_valid")) is not bool:
                raise ValueError("queue result must include strict bool infra_valid")
            score = result.get("score")
            if score is not None and (type(score) not in (int, float) or not isfinite(score)):
                raise ValueError("queue result score must be numeric or null")
            if (result["infra_valid"] and score is None) or (not result["infra_valid"] and score is not None):
                raise ValueError("queue score must match infrastructure validity")
            with self._transaction(operation="finish") as state:
                row = state["jobs"].get(claim.job_id)
                for old in row.get("attempts", []) if row else []:
                    if old.get("token") == claim.token and old.get("status") != "running":
                        if old.get("result") == result and old.get("archive") == archive:
                            return
                        raise ValueError("repeated finish differs from committed outcome")
                if not row or row["status"] != "running" or not row["attempts"]:
                    raise ValueError("queue claim is no longer active")
                attempt = row["attempts"][-1]
                if attempt.get("token") != claim.token or attempt.get("attempt_number") != claim.attempt_number:
                    raise ValueError("queue claim token does not match active attempt")
                attempt.update(status="completed" if result["infra_valid"] else "infra_failed",
                               finished_at=_now(), result=result, archive=archive)
                if execution is not None:
                    attempt['execution_receipt'] = execution
                cause = failure_cause(result)
                if cause is not None:
                    attempt['failure_cause'] = cause
                row["result"] = result
                if self.queue.rolling is not None and not result['infra_valid']:
                    driver = state['drivers'].get(self.queue.node.hostname)
                    if driver is not None:
                        driver['healthy_since_unix_s'] = time.time()
                if result["infra_valid"]:
                    row["status"] = "await_review"
                elif claim.attempt_number < self.max_attempts:
                    row["status"] = "queued"
                else:
                    row["status"] = "infra_failed"
                    _failure_hold(row)
        finally:
            claim.close()

    def register_driver(self, *, hostname, pid, current_parallel, stable_parallel, status="running", details=None):
        if self.queue.rolling is not None and (hostname != self.queue.node.hostname or pid != os.getpid()):
            raise ValueError('rolling driver registration requires this actual node/PID')
        with self._transaction(operation="register_driver") as state:
            previous = state["drivers"].get(hostname, {})
            started_at = previous.get("started_at", _now()) if previous.get("status") == "running" and previous.get("pid") == pid else _now()
            state["drivers"][hostname] = {"pid": pid, "status": status,
                                          "current_parallel": current_parallel,
                                          "stable_parallel": stable_parallel,
                                          "started_at": started_at,
                                          "last_heartbeat_at": _now(), **(details or {})}
            if self.queue.rolling is not None:
                driver = state['drivers'][hostname]
                driver.update(heartbeat_unix_s=time.time(), uid=os.getuid(),
                    profile_sha256=self.queue.profile.sha256,
                    code_sha=os.environ.get('MIND2ACT_EVAL_CODE_SHA'), process_birth=_process_birth(pid),
                    queue_manifest_sha256=self.queue.manifest_sha256)
                generation_changed = any(previous.get(key) != driver[key] for key in
                    ('pid', 'process_birth', 'uid', 'profile_sha256', 'code_sha', 'queue_manifest_sha256'))
                restarted = previous.get('status') != 'running' and status == 'running'
                driver['healthy_since_unix_s'] = (time.time() if generation_changed or restarted
                    else previous.get('healthy_since_unix_s', time.time()))
                if generation_changed or restarted:
                    driver['started_at'] = _now()
                    from mind2act_eval.pro_rolling_queue import control
                    old = dict(control(self, state, hostname))
                    initial = self.queue.node.initial_parallel // len(self.queue.node.gpu_indices)
                    if old['workers_per_gpu'] > initial:
                        # Permission belongs to this live generation, not the
                        # durable target. Leave drain and all attempts intact.
                        new = dict(old, workers_per_gpu=initial, revision=old['revision'] + 1)
                        state['dispatch'][hostname] = new
                        state.setdefault('dispatch_history', []).append(dict(hostname=hostname,
                            before=old, after=new, evidence='driver generation replaced; fresh fleet qualification required',
                            at_unix_s=time.time()))

    def is_finished(self):
        return all(row["status"] == "reviewed" and row.get("review", {}).get("decision", "pass") == "pass"
                   for row in self.snapshot()["jobs"].values())

    def block(self, claim, error):
        try:
            with self._transaction(operation="block") as state:
                row = state["jobs"][claim.job_id]
                if row["attempts"][-1]["token"] != claim.token:
                    raise ValueError("blocked claim token mismatch")
                attempt = row["attempts"].pop()
                row["history"].append({"kind": "blocked_before_execution", "claim": attempt, "error": error})
                row["status"] = "blocked"
                row["blocked_error"] = error
        finally:
            claim.close()

    def publish_generation(self, definition, *, evidence):
        """Publish prepared inputs without changing any live attempt or old history."""
        if self.queue.rolling is not None:
            raise ValueError("rolling generations require new immutable queue/profile receipts")
        if not evidence or not isinstance(definition, dict):
            raise ValueError("new generation requires definition and evidence")
        with self._transaction(operation="publish_generation") as state:
            row = state["jobs"][definition["job_id"]]
            if 'quarantine' in row:
                raise ValueError('quarantined jobs cannot be republished or automatically released')
            if 'maintenance_hold' in row:
                raise ValueError('release the owning maintenance hold before publishing a generation')
            if row["status"] == "running":
                raise ValueError("cannot change a live queue generation")
            if row["status"] in {"await_review", "reviewed"}:
                review = row.get("review", {})
                if (row["status"] != "reviewed" or review.get("decision") != "fail"
                        or review.get("generation") != row["generation"]):
                    raise ValueError("replacement requires current generation reviewed with QA fail")
            generation = definition.get("generation")
            used = {str(row["generation"])} | {str(h.get("generation")) for h in row["history"] if "generation" in h}
            if not isinstance(generation, str) or not generation or generation in used:
                raise ValueError("published generation must be a new nonempty string")
            def identity(value):
                return _json_bytes({"options": value.get("options", {}), "inputs": value.get("inputs", {})})
            if any(identity(definition) == identity(old) for old in [row["definition"]] +
                   [h["definition"] for h in row["history"] if "definition" in h]):
                raise ValueError("same code/inputs cannot reset the retry budget under a new generation")
            if definition.get("episode_key") != row["definition"]["episode_key"]:
                raise ValueError("new generation cannot change the grid episode key")
            required = {"spec", "spec_sha256", "asset_root", "backend", "adapter", "physics_ticks", "wall_s"}
            if not required <= set(definition.get("options", {})):
                raise ValueError("new generation requires prepared execution options")
            options = definition["options"]
            def protocol(value):
                return tuple(value.get('options', {}).get(key)
                             for key in ('policy_kind', 'budget_version'))
            original = next(item for item in self.queue.jobs
                            if item['job_id'] == definition['job_id'])
            existing = [item for item in self.queue.jobs
                        if item.get('initial_status', 'ready') == 'ready']
            existing += [item['definition'] for item in state['jobs'].values()
                         if item['status'] in {'queued', 'running', 'await_review', 'reviewed'}]
            if (protocol(definition) != protocol(original)
                    or any(protocol(item) != protocol(definition) for item in existing)):
                raise ValueError('new generation cannot change the immutable queue policy budget protocol')
            for path_key, sha_key in (("spec", "spec_sha256"), ("h5", "h5_sha256"),
                                      ("replay_metadata", "replay_metadata_sha256"),
                                      ("adapter_config", "adapter_config_sha256")):
                if options.get(path_key) and (not isinstance(options.get(sha_key), str)
                                             or not _SHA256.fullmatch(options[sha_key])):
                    raise ValueError("published input paths require pinned SHA256 values")
            row["history"].append({"generation": row["generation"], "definition": row["definition"],
                                   "attempts": row["attempts"], "result": row["result"],
                                   "review": row.get("review"), "status": row["status"], "evidence": evidence,
                                   **({'failure_hold': row['failure_hold']} if 'failure_hold' in row else {})})
            row.update(generation=generation, definition=definition, attempts=[], result=None, status="queued")
            row.pop("review", None)
            row.pop("blocked_error", None)
            row.pop('failure_hold', None)

    def recheck_blocked(self):
        # Existing source files do not establish prepared assets/cameras/binding.
        # Only publish_generation with explicit evidence can release a block.
        return self.snapshot()

    def maintenance_snapshot(self):
        """Controller CAS preview under the existing queue lock; never initialize."""
        with self._transaction(write=False) as state:
            if self.queue.manifest_path.read_bytes() != self.queue.manifest_bytes:
                raise ValueError('maintenance manifest bytes changed')
            return dict(manifest_sha256=self.queue.manifest_sha256,
                        state_sha256=self._read_state()[1], state=json.loads(_json_bytes(state)),
                        row_sha256={key: hashlib.sha256(_json_bytes(row)).hexdigest()
                                    for key, row in state['jobs'].items()})

    def _maintenance_cas(self, state, manifest_sha256, state_sha256, rows):
        if (manifest_sha256 != self.queue.manifest_sha256
                or self.queue.manifest_path.read_bytes() != self.queue.manifest_bytes):
            raise ValueError('maintenance manifest CAS mismatch')
        if (not isinstance(state_sha256, str) or not _SHA256.fullmatch(state_sha256)
                or self._read_state()[1] != state_sha256):
            raise ValueError('maintenance state CAS mismatch')
        if not isinstance(rows, dict) or not rows:
            raise ValueError('maintenance requires explicit nonempty row hashes')
        for key, expected in rows.items():
            if (key not in state['jobs'] or not isinstance(expected, str) or not _SHA256.fullmatch(expected)
                    or hashlib.sha256(_json_bytes(state['jobs'][key])).hexdigest() != expected):
                raise ValueError('maintenance row CAS mismatch')

    @staticmethod
    def _held_maintenance_row(hold_id, original):
        row = json.loads(_json_bytes(original))
        row.update(status='blocked', maintenance_hold=dict(hold_id=hold_id,
            generation=original['generation'],
            definition_sha256=hashlib.sha256(_json_bytes(original['definition'])).hexdigest(),
            original_row_sha256=hashlib.sha256(_json_bytes(original)).hexdigest()))
        return row

    def _maintenance_change(self, hold_id, *, controller_id, expected_manifest_sha256,
                            expected_state_sha256, expected_rows, evidence, release):
        if (not isinstance(hold_id, str) or not _QUEUE_ID.fullmatch(hold_id)
                or not isinstance(controller_id, str) or not _QUEUE_ID.fullmatch(controller_id)
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError('maintenance requires hold/controller identity and evidence')
        leases = []
        try:
            with self._transaction(operation='maintenance_release' if release else 'maintenance_hold',
                                   maintenance_locks=leases) as state:
                self._maintenance_cas(state, expected_manifest_sha256, expected_state_sha256, expected_rows)
                records = state.get('maintenance_holds', {})
                record = records.get(hold_id)
                if record is not None:
                    if (record['controller_id'] != controller_id
                            or record['manifest_sha256'] != expected_manifest_sha256
                            or set(record['original_rows']) != set(expected_rows)):
                        raise ValueError('foreign maintenance hold or row binding')
                    originals = record['original_rows']
                    expected_status = record['status']
                    if expected_status not in {'held', 'released'} or (not release and expected_status != 'held'):
                        raise ValueError('maintenance hold lifecycle differs')
                    for key, original in originals.items():
                        expected = (self._held_maintenance_row(hold_id, original)
                                    if expected_status == 'held' else original)
                        if state['jobs'][key] != expected:
                            raise ValueError('maintenance original binding or held row changed')
                    if not release or expected_status == 'released':
                        return json.loads(_json_bytes(record))  # fresh-CAS idempotent; no state rewrite
                elif release:
                    raise ValueError('unknown or foreign maintenance hold')
                else:
                    originals = {}
                    for key in expected_rows:
                        row = state['jobs'][key]
                        if (row['status'] != 'queued' or row['attempts'] != [] or row['result'] is not None
                                or 'quarantine' in row or 'maintenance_hold' in row):
                            raise ValueError('maintenance only holds unclaimed queued-zero rows')
                        originals[key] = json.loads(_json_bytes(row))
                # Do not create lease files, or infer that an extant holder exited.
                for key in sorted(expected_rows):
                    path = self._job_lock_path(key)
                    try:
                        fd = os.open(path, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
                    except FileNotFoundError:
                        continue
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BaseException:
                        os.close(fd)
                        raise ValueError('maintenance row lease is held')
                    leases.append((path, fd))
                if release:
                    for key, original in originals.items():
                        state['jobs'][key] = json.loads(_json_bytes(original))
                    record.update(status='released', released_at=_now(), release_evidence=evidence,
                                  release_before_state_sha256=expected_state_sha256)
                else:
                    record = dict(hold_id=hold_id, controller_id=controller_id, status='held',
                                  manifest_sha256=expected_manifest_sha256,
                                  before_state_sha256=expected_state_sha256,
                                  original_row_sha256=dict(expected_rows), original_rows=originals,
                                  held_at=_now(), hold_evidence=evidence)
                    state.setdefault('maintenance_holds', {})[hold_id] = record
                    for key, original in originals.items():
                        state['jobs'][key] = self._held_maintenance_row(hold_id, original)
                if self.queue.manifest_path.read_bytes() != self.queue.manifest_bytes:
                    raise ValueError('maintenance manifest changed before commit')
                return json.loads(_json_bytes(record))
        finally:
            for _, fd in leases:
                os.close(fd)

    def hold_queued(self, hold_id, *, controller_id, expected_manifest_sha256,
                    expected_state_sha256, expected_rows, evidence):
        """Controller only: atomically fence explicit queued-zero rows as legacy blocked.

        Every call, including an idempotent retry, requires a fresh exact CAS.
        Generation, attempts, result, token and row history are never edited.
        """
        return self._maintenance_change(hold_id, controller_id=controller_id,
            expected_manifest_sha256=expected_manifest_sha256, expected_state_sha256=expected_state_sha256,
            expected_rows=expected_rows, evidence=evidence, release=False)

    def release_maintenance_hold(self, hold_id, *, controller_id, expected_manifest_sha256,
                                 expected_state_sha256, expected_rows, evidence):
        """Restore only this controller's unchanged original rows; never quarantine."""
        return self._maintenance_change(hold_id, controller_id=controller_id,
            expected_manifest_sha256=expected_manifest_sha256, expected_state_sha256=expected_state_sha256,
            expected_rows=expected_rows, evidence=evidence, release=True)

    def _quarantine_target(self, state, job_id, generation, definition_sha256, external_batch):
        if (not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id)
                or not isinstance(generation, str) or not generation
                or not isinstance(definition_sha256, str) or not _SHA256.fullmatch(definition_sha256)
                or not isinstance(external_batch, dict)
                or set(external_batch) != {'campaign_id', 'manifest_sha256', 'hostname', 'code_sha'}
                or not isinstance(external_batch['campaign_id'], str) or not _QUEUE_ID.fullmatch(external_batch['campaign_id'])
                or not isinstance(external_batch['manifest_sha256'], str) or not _SHA256.fullmatch(external_batch['manifest_sha256'])
                or not isinstance(external_batch['hostname'], str) or external_batch['hostname'] not in self.queue.nodes
                or not isinstance(external_batch['code_sha'], str) or not re.fullmatch(r'[0-9a-f]{40}', external_batch['code_sha'])):
            raise ValueError('quarantine requires exact generation/definition and external batch identity')
        row = state['jobs'].get(job_id)
        if (row is None or row['generation'] != generation or row['attempts'] != [] or row['result'] is not None
                or hashlib.sha256(_json_bytes(row['definition'])).hexdigest() != definition_sha256):
            raise ValueError('quarantine CAS differs from the original unexecuted generation')
        return row

    def quarantine_queued(self, job_id, *, expected_generation, expected_definition_sha256,
                          external_batch, reason, evidence):
        """Fence a queued-zero job using existing blocked state; never touch its lease.

        This reserves external execution; it is not proof an unknown lock holder
        exited. Old resident claim_next already skips blocked rows. No release
        operation is provided, including after external completion.
        """
        if any(not isinstance(value, str) or not value.strip() for value in (reason, evidence)):
            raise ValueError('quarantine requires a reason and explicit evidence')
        with self._transaction(operation='quarantine_queued') as state:
            row = self._quarantine_target(state, job_id, expected_generation,
                                          expected_definition_sha256, external_batch)
            if row['status'] != 'queued' or 'quarantine' in row:
                raise ValueError('only an unquarantined queued-zero job may be quarantined')
            record = dict(quarantine_id=uuid.uuid4().hex, job_id=job_id,
                          generation=expected_generation, definition_sha256=expected_definition_sha256,
                          original_definition=json.loads(_json_bytes(row['definition'])),
                          external_batch=json.loads(_json_bytes(external_batch)), reason=reason,
                          evidence=evidence, status='quarantined', created_at=_now())
            row.update(status='blocked', blocked_error='explicit queued-zero quarantine: ' + reason,
                       quarantine=record)
            row['history'].append(dict(kind='queued_quarantine', **record))
            return json.loads(_json_bytes(record))

    def record_quarantine_completion(self, job_id, *, quarantine_id, expected_generation,
                                     expected_definition_sha256, external_batch, evidence):
        """Audit an external completion reference while retaining the blocked fence."""
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('external completion requires an evidence reference')
        with self._transaction(operation='quarantine_external_completion') as state:
            row = self._quarantine_target(state, job_id, expected_generation,
                                          expected_definition_sha256, external_batch)
            record = row.get('quarantine')
            if (row['status'] != 'blocked' or not isinstance(record, dict)
                    or record.get('quarantine_id') != quarantine_id
                    or record.get('external_batch') != external_batch
                    or record.get('status') not in {'quarantined', 'external_completed'}):
                raise ValueError('quarantine identity differs from the current external reservation')
            if record['status'] == 'external_completed':
                if record.get('completion_evidence') != evidence:
                    raise ValueError('external completion differs from its committed audit')
                return json.loads(_json_bytes(record))
            record.update(status='external_completed', completion_evidence=evidence, completed_at=_now())
            row['history'].append(dict(kind='quarantine_external_completion',
                quarantine_id=quarantine_id, external_batch=json.loads(_json_bytes(external_batch)),
                evidence=evidence, at=record['completed_at']))
            return json.loads(_json_bytes(record))

    def amend_quarantine_external_batch(self, **request):
        """Repair only an omitted GPU before external launch; retain the fence."""
        from mind2act_eval.quarantine_amendment import amend_gpu_omission
        return amend_gpu_omission(self, **request)

    def recover(self, job_id, *, confirmed_exited, evidence):
        if confirmed_exited is not True or not evidence:
            raise ValueError("recovery requires confirmed owner/worker exit evidence")
        with self._transaction(operation="recover") as state:
            row = state["jobs"][job_id]
            if row["status"] != "running":
                raise ValueError("only running claims need recovery")
            previous = row["attempts"][-1]
            import socket
            if previous["owner_hostname"] == socket.gethostname().lower():
                birth = _process_birth(previous["owner_pid"])
                if birth is not None and birth == previous.get("owner_starttime"):
                    raise ValueError("claim owner is still alive")
                if birth is None:
                    try:
                        os.kill(previous["owner_pid"], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ValueError("claim owner exit cannot be established")
            fd = self._try_job_lock(job_id)
            if fd is None:
                raise ValueError("attempt lease is still held by driver or worker")
            try:
                row["history"].append({"kind": "confirmed_recovery", "evidence": evidence, "at": _now()})
                self._recover_or_fail(row, row["definition"])
            finally:
                os.close(fd)

    def review(self, job_id, *, decision, reviewer, evidence):
        if decision not in {"pass", "fail"} or not reviewer or not evidence:
            raise ValueError("review needs pass/fail, reviewer and evidence")
        with self._transaction(operation="review") as state:
            row = state["jobs"][job_id]
            if row["status"] != "await_review":
                raise ValueError("only executed results may be reviewed")
            row.update(status="reviewed", review={"generation": row["generation"],
                                                   "attempt_token": row["attempts"][-1]["token"],
                                                   "decision": decision, "reviewer": reviewer,
                                                   "evidence": evidence, "at": _now()})

    def snapshot(self):
        with self._transaction(write=False) as state:
            return json.loads(json.dumps(state))

    def correct_review(self, job_id, *, generation, attempt_token, expected_review,
                       decision, reviewer, evidence, reason):
        """Correct one reviewed attempt, retaining its prior review in an audit ledger.

        Historical corrections never certify the current generation. The exact
        expected review prevents a stale request from undoing a later correction.
        """
        if (decision not in {"pass", "fail"} or not isinstance(expected_review, dict)
                or any(not isinstance(value, str) or not value.strip()
                       for value in (job_id, generation, attempt_token, reviewer, evidence, reason))):
            raise ValueError("review correction needs exact target, prior review, reviewer, evidence and reason")
        with self._transaction(operation="correct_review") as state:
            row = state["jobs"][job_id]
            candidates = [entry for entry in [row, *row["history"]]
                          if entry.get("generation") == generation
                          and entry.get("status") == "reviewed"
                          and entry.get("attempts")
                          and entry["attempts"][-1].get("token") == attempt_token]
            if len(candidates) != 1:
                raise ValueError("review correction must identify one reviewed generation/attempt")
            target = candidates[0]
            previous = target.get("review")
            if (previous != expected_review or not isinstance(previous, dict)
                    or previous.get("generation") != generation
                    or previous.get("attempt_token") != attempt_token):
                raise ValueError("review correction prior review is stale or belongs to another attempt")
            corrected = {"generation": generation, "attempt_token": attempt_token,
                         "decision": decision, "reviewer": reviewer, "evidence": evidence,
                         "at": _now(), "correction_id": uuid.uuid4().hex}
            state.setdefault("review_corrections", []).append({
                "job_id": job_id, "generation": generation, "attempt_token": attempt_token,
                "previous_review": previous, "corrected_review": corrected, "reason": reason})
            target["review"] = corrected


def run_queue_driver(queue: SharedQueue, *, job_factory, require_shared=True, poll_s=.2,
                     legacy_cooldown_s=60., run_attempt_fn=None, archive_fn=None):
    """Hold canonical launcher locks once, drain claims with bounded local lanes."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from mind2act_eval.batch import (_archive_attempt, _campaign_launcher_gates, _launcher_lock_root,
                                     run_attempt)
    from mind2act_eval.contracts import to_json

    if not 0 < poll_s <= 5:
        raise ValueError("queue poll interval must be in (0, 5]")
    output_root = queue.output_root.resolve(strict=False)
    lock_root = _launcher_lock_root(output_root, require_shared=require_shared)
    if lock_root != queue.lock_root:
        raise ValueError("queue lock root differs from MIND2ACT_EVAL_LOCK_ROOT")
    profile, cluster_scope = queue.profile, None
    if profile is None and require_shared and not all(_under_shared(p) for p in (queue.manifest_path, queue.state_root, output_root)):
        raise ValueError("production queue inputs, state, and output must resolve under /shared")
    node = queue.node
    if require_shared and not re.fullmatch(r"[0-9a-f]{40}", os.environ.get("MIND2ACT_EVAL_CODE_SHA", "")):
        raise ValueError("resident production driver requires fixed MIND2ACT_EVAL_CODE_SHA")
    store = QueueStore(queue)
    store.snapshot()  # Fail before admission, launcher locks or output creation.
    def operator_resources():
        if node.resource_config is None:
            return None
        config = BatchResourceConfig.read(node.resource_config)
        if config.rss_limit_bytes != node.rss_limit_bytes or not 1 <= config.max_parallel <= node.stable_parallel:
            raise ValueError("resident resource config exceeds authorized node profile")
        return config
    def resource_config(parallel):
        if profile is not None:
            return SimpleNamespace(rss_limit_bytes=node.rss_limit_bytes, max_parallel=parallel,
                                   source_sha256=profile.sha256)
        config = operator_resources()
        if config is not None:
            # Pin the exact operator bytes/policy at every admission. Initial
            # and scale requests cannot exceed that file's current lane cap.
            return replace(config, max_parallel=min(parallel, config.max_parallel))
        return BatchResourceConfig(node.rss_limit_bytes, parallel, str(queue.manifest_path), queue.manifest_sha256)
    def admit(config):
        if profile is None:
            return resolve_resources(DEFAULT_RSS_LIMIT_BYTES, 1, config)
        if config.rss_limit_bytes != profile.rss_limit_bytes or not 1 <= config.max_parallel <= profile.max_parallel:
            raise ValueError('PRO resident resources differ from its pinned profile')
        receipt = dict(profile_sha256=profile.sha256) if cluster_scope is None else cluster_scope.snapshot()
        return config.rss_limit_bytes, config.max_parallel, receipt
    resource_initial = resource_config(node.initial_parallel)
    _rss, _parallel, admission_receipt = admit(resource_initial)
    initial = resource_initial.max_parallel
    target = node.stable_parallel
    run_attempt_fn = run_attempt if run_attempt_fn is None else run_attempt_fn
    archive_fn = _archive_attempt if archive_fn is None else archive_fn
    output_root.mkdir(parents=True, exist_ok=True)
    active = {}
    active_gpus = set()
    stable_since = None
    last_heartbeat = 0.0
    last_scale_check = 0.0
    scaled = initial == target
    exit_code = 0
    launcher_identity = SimpleNamespace(hostname=node.hostname)
    from mind2act_eval.pro_device_leases import ProCampaignAdmission
    capacity_admission = ProCampaignAdmission.from_env(profile)
    stop_event = threading.Event()
    old_sigterm = None
    if threading.current_thread() is threading.main_thread():
        old_sigterm = signal.signal(signal.SIGTERM, lambda _sig, _frame: stop_event.set())
    try:
        with ExitStack() as scopes:
            scopes.enter_context(_campaign_launcher_gates(lock_root, distributed_partition=launcher_identity,
                                 legacy_cooldown_s=legacy_cooldown_s, cancel_event=stop_event,
                                 pro_admission=capacity_admission))
            if profile is not None:
                from mind2act_eval.campaign_resources import CampaignResources
                receipt_dir = output_root / f'queue-{queue.queue_id}'
                receipt_dir.mkdir(parents=True, exist_ok=True)
                cluster_scope = scopes.enter_context(CampaignResources(profile,
                    receipt_dir / f'pro-resources-{node.hostname}-{os.getpid()}.json'))
                if capacity_admission is not None:
                    scopes.callback(capacity_admission.close)
            # Recheck after both the global compatibility SH lock and the
            # hostname/UID EX lock are held.
            resource_initial = resource_config(initial)
            _rss, _parallel, admission_receipt = admit(resource_initial)
            initial = resource_initial.max_parallel
            store.register_driver(hostname=node.hostname, pid=os.getpid(), current_parallel=initial,
                                  stable_parallel=target, details={"queue_manifest_sha256": queue.manifest_sha256,
                                                                  "machine_id_sha256": node.machine_id_sha256,
                                                                  "rss_limit_bytes": node.rss_limit_bytes,
                                                                  "resource_admission": admission_receipt})
            with ThreadPoolExecutor(max_workers=target) as pool:
                while True:
                    now = time.monotonic()
                    for future, active_item in list(active.items()):
                        claim, job = active_item
                        if not future.done():
                            continue
                        try:
                            result = future.result()
                        except BaseException as exc:
                            from mind2act_eval.batch import _invalid
                            from mind2act_eval.contracts import Termination
                            result = _invalid(job, Termination.ERROR, f"{type(exc).__name__}: {exc}")
                        if queue.rolling is not None:
                            exit_path = claim.directory / 'batch-exit.json'
                            if exit_path.is_file() and json.loads(exit_path.read_text()).get('dispatch_deferred') is True:
                                from mind2act_eval.pro_rolling_queue import release_unstarted
                                release_unstarted(store, claim)
                                del active[future]
                                continue
                        archive = None
                        if claim.directory.is_dir():
                            try:
                                archive_path = archive_fn(claim.directory, claim.directory.with_suffix(".tar.gz"))
                                archive = {"path": archive_path.name,
                                           "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
                            except Exception as exc:
                                archive = {"error": f"{type(exc).__name__}: {exc}"}
                        result_raw = to_json(result)
                        store.finish(claim, result=result_raw, archive=archive)
                        if result.infra_valid is False:
                            stable_since = None
                        elif stable_since is None:
                            stable_since = now
                        if claim.gpu is not None:
                            active_gpus.discard(claim.gpu)
                        del active[future]

                    if (queue.rolling is None and not scaled and stable_since is not None
                            and now - stable_since >= node.stable_window_s
                            and now - last_scale_check >= 60.):
                        last_scale_check = now
                        target_config = resource_config(target)
                        try:
                            _rss, _parallel, admission_receipt = admit(target_config)
                        except ResourceAdmissionError as exc:
                            admission_receipt = exc.receipt
                            store.register_driver(hostname=node.hostname, pid=os.getpid(),
                                                  current_parallel=initial, stable_parallel=target,
                                                  details={"scale_blocked": str(exc), "scale_checked_at": _now(),
                                                           "resource_admission": exc.receipt})
                        else:
                            initial = target_config.max_parallel
                            scaled = initial == target
                            store.register_driver(hostname=node.hostname, pid=os.getpid(),
                                                  current_parallel=initial, stable_parallel=target,
                                                  details={"scaled_at": _now(), "resource_admission": admission_receipt})

                    draining = False
                    if queue.rolling is not None:
                        from mind2act_eval.pro_rolling_queue import control
                        current = control(store, store.snapshot())
                        draining = current['drain']
                        initial = len(node.gpu_indices) * current['workers_per_gpu']
                    while not draining and not stop_event.is_set() and len(active) < initial:
                        runtime_config = (operator_resources() if node.resource_config
                                          else resource_config(initial))
                        try:
                            _rss, _parallel, admission_receipt = admit(runtime_config)
                        except ResourceAdmissionError as exc:
                            admission_receipt = exc.receipt
                            store.register_driver(hostname=node.hostname, pid=os.getpid(),
                                                  current_parallel=initial, stable_parallel=target,
                                                  details={"admission_blocked": str(exc), "resource_admission": exc.receipt})
                            break
                        if node.resource_config:
                            # An explicit operator resource file already states
                            # the authorized cap, bounded above by this node's
                            # fixed target and the unchanged 96GiB admission.
                            allowed = runtime_config.max_parallel
                            initial = allowed
                            if len(active) >= initial:
                                break
                        if queue.rolling is not None:
                            counts = {gpu: sum(claim.gpu == gpu for claim, _ in active.values()) for gpu in node.gpu_indices}
                            free_gpus = sorted((gpu for gpu in node.gpu_indices if counts[gpu] < current['workers_per_gpu']), key=counts.get)
                        else:
                            free_gpus = [gpu for gpu in node.gpu_indices if gpu not in active_gpus]
                        if queue.rolling is None and any(job["options"].get("backend", "isaac") == "isaac" for job in queue.jobs):
                            from mind2act_eval.batch import check_gpu_idle
                            idle = []
                            for candidate in free_gpus:
                                try:
                                    if cluster_scope is None:
                                        check_gpu_idle(candidate)
                                    else:
                                        cluster_scope.check_gpu(candidate)
                                except (ValueError, OSError):
                                    continue
                                idle.append(candidate)
                            free_gpus = idle
                        if node.gpu_indices and not free_gpus:
                            break
                        candidates = (free_gpus if queue.rolling is not None else
                                      [free_gpus[0] if free_gpus else None])
                        claim = None
                        for gpu in candidates:
                            claim = store.claim_next(hostname=node.hostname, pid=os.getpid(), gpu=gpu,
                                                     resource_config_sha256=runtime_config.source_sha256)
                            if claim is not None:
                                break
                        if claim is None:
                            break
                        claim.directory.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            expected_code = (claim.inputs or {}).get("code_sha")
                            if expected_code and expected_code != os.environ.get("MIND2ACT_EVAL_CODE_SHA"):
                                raise ValueError("queued generation requires a different fixed code SHA")
                            job = job_factory({"job_id": claim.job_id, "episode_key": claim.episode_key,
                                               "options": claim.options}, gpu)
                            if profile is not None:
                                cluster_scope.verify_job(job, dynamic_gpu=queue.rolling is None)
                        except BaseException as exc:
                            store.block(claim, f"{type(exc).__name__}: {exc}")
                            stable_since = None
                            continue
                        scope_options = {} if cluster_scope is None else dict(cluster_scope=cluster_scope)
                        if capacity_admission is not None:
                            scope_options['capacity_admission'] = capacity_admission
                        if queue.rolling is not None:
                            from functools import partial
                            from mind2act_eval.pro_rolling_queue import launch_guard
                            scope_options['launch_guard'] = partial(launch_guard, store, claim)
                        future = pool.submit(run_attempt_fn, job, claim.directory,
                                             rss_limit=node.rss_limit_bytes, lock_root=lock_root,
                                             claim_lock_fd=claim.lock_fd, parent_start_gate=True,
                                             cancel_event=stop_event, **scope_options)
                        # Only workers spawned by this driver receive cancellation.
                        active[future] = (claim, job)
                        if gpu is not None:
                            active_gpus.add(gpu)

                    state = store.snapshot()
                    if (stop_event.is_set() or draining) and not active:
                        break
                    if not active and store.is_finished():
                        break
                    if now - last_heartbeat >= 10:
                        store.recheck_blocked()
                        store.register_driver(hostname=node.hostname, pid=os.getpid(),
                                              current_parallel=initial, stable_parallel=target,
                                              details={"active_jobs": [claim.job_id for claim, _job in active.values()],
                                                       "queue_manifest_sha256": queue.manifest_sha256,
                                                       "machine_id_sha256": node.machine_id_sha256,
                                                       "rss_limit_bytes": node.rss_limit_bytes,
                                                       "resource_admission": admission_receipt})
                        last_heartbeat = now
                    if not active and store.is_finished():
                        break
                    time.sleep(poll_s)
        state = store.snapshot()
        if stop_event.is_set() or any(row["status"] == "infra_failed" for row in state["jobs"].values()):
            exit_code = 1
        store.register_driver(hostname=node.hostname, pid=os.getpid(), current_parallel=initial,
                              stable_parallel=target, status="stopped" if stop_event.is_set() else "drained" if draining else "completed",
                              details={"queue_manifest_sha256": queue.manifest_sha256,
                                       "resource_admission": admission_receipt})
        return exit_code
    except BaseException:
        for future, active_item in list(active.items()):
            claim, _job = active_item
            if future.done():
                try:
                    result = future.result()
                except BaseException as exc:
                    result = {"infra_valid": False, "score": None,
                              "error": f"{type(exc).__name__}: {exc}"}
                if hasattr(result, "infra_valid"):
                    from mind2act_eval.contracts import to_json
                    result = to_json(result)
                try:
                    store.finish(claim, result=result)
                except Exception:
                    claim.close()
            # If still active, keep the lease open in the child via pass_fds;
            # closing the parent descriptor cannot release its inherited lock.
            else:
                claim.close()
        store.register_driver(hostname=node.hostname, pid=os.getpid(), current_parallel=initial,
                              stable_parallel=target, status="failed",
                              details={"queue_manifest_sha256": queue.manifest_sha256,
                                       "resource_admission": admission_receipt})
        raise
    finally:
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
