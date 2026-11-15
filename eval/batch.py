"""Serial child supervisor: user RSS budget, bounded retries, immutable attempts.

Only process groups created here are terminated. Never search command lines or
signal pre-existing user processes. All output/cache/temp files live externally.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from contextlib import ExitStack, contextmanager, nullcontext

from mind2act_eval.contracts import ConsumerKind, Result, Termination
from mind2act_eval.runner import write_json
from mind2act_eval.budgets import AGENT_POLICY_KINDS
from mind2act_eval.sealing import SEAL_NAME, validate_worker_seal
from mind2act_eval.sim import startup_gate
from mind2act_eval.startup_paths import startup_paths, startup_compatibility, verify_startup_domain
from mind2act_eval.batch_resources import DEFAULT_RSS_LIMIT_BYTES, ResourceAdmissionError, resolve_resources

RSS_LIMIT_BYTES = DEFAULT_RSS_LIMIT_BYTES


def user_rss_bytes() -> int:
    """Sum all current-uid RSS, including unrelated jobs; inaccessible => fail."""
    values = subprocess.run(["ps", "-U", str(os.getuid()), "-o", "rss="],
                            check=True, capture_output=True, text=True)
    return sum(int(value) for value in values.stdout.split()) * 1024


@dataclass(frozen=True)
class EpisodeJob:
    spec: object
    argv: tuple[str, ...]  # worker args; output directory appended by supervisor
    consumer_kind: ConsumerKind = ConsumerKind.MODEL
    wall_s: float = 3600
    retries: int = 1
    gpu: int | None = None
    native_exit_debug: None = None
    policy_kind: str | None = None
    budget_version: str | None = None
    physics_ticks: int | None = None

    def __post_init__(self):
        from math import isfinite
        if not self.argv or type(self.retries) is not int or not 0 <= self.retries <= 2 or not isfinite(self.wall_s) or self.wall_s <= 0:
            raise ValueError("invalid episode job")
        if self.policy_kind is not None or self.budget_version is not None:
            from mind2act_eval.runner import EpisodeLimits
            EpisodeLimits(self.physics_ticks, self.wall_s, self.policy_kind, self.budget_version)
            if self.consumer_kind != ConsumerKind.MODEL or self.retries != 0:
                raise ValueError('versioned policy jobs require model consumer and retries=0')
        if self.native_exit_debug is not None:
            raise ValueError("native debugging tools are not part of the public runtime")




def check_gpu_idle(index):
    from mind2act_eval.cluster_profile import load_cluster_profile, check_pro_gpu
    profile = load_cluster_profile()
    if profile is not None:
        if profile.max_workers_per_gpu > 1 and profile.shared_gpu_admission_policy == 'warn':
            from mind2act_eval.cluster_profile import check_pro_shared_gpu
            return check_pro_shared_gpu(profile, index, admitted={})
        return check_pro_gpu(profile, index)
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    for row in query.stdout.splitlines():
        gpu, name, memory, utilization = [value.strip() for value in row.split(",")]
        if int(gpu) == index:
            if "PRO" in name.upper() and "5000" in name:
                raise ValueError("PRO5000 is excluded")
            if int(memory) >= 2048 or int(utilization) != 0:
                raise ValueError("selected GPU is not idle")
            return
    raise ValueError("selected GPU index does not exist")


def _stop_owned_group(child, *, grace_s=5):
    """A new_session child owns this group; never signal a discovered process."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    # Parent exit does not prove descendants exited. Kill residual owned group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def _invalid(job, reason, error):
    return Result(job.spec.task_id, job.spec.instance_id, None, reason, False,
                  job.consumer_kind, False, error=error, policy_kind=job.policy_kind,
                  budget_version=job.budget_version,
                  budget_limits=(dict(physics_ticks=job.physics_ticks, wall_s=job.wall_s,
                                      retries=job.retries) if job.policy_kind else {}))


def _validate_result_budget(job, result):
    if (result.policy_kind, result.budget_version) != (job.policy_kind, job.budget_version):
        raise ValueError('worker result policy budget version differs from parent job')
    if job.policy_kind and any(result.budget_limits.get(key) != value for key, value in
            dict(physics_ticks=job.physics_ticks, wall_s=job.wall_s, retries=job.retries).items()):
        raise ValueError('worker result execution budget differs from parent job')


def _sidecar_artifacts(directory):
    artifacts = {}
    if (Path(directory) / 'run-budget.json').exists():
        artifacts['run_budget'] = 'run-budget.json'
    if (Path(directory) / 'call-budget.json').exists():
        artifacts['call_budget'] = 'call-budget.json'
    for key, name in (('native_phase', 'native-phase.json'),
                      ('native_phase_traceback', 'native-phase-traceback.log')):
        if (Path(directory) / name).exists():
            artifacts[key] = name
    if (Path(directory) / "cleanup-started.json").exists():
        artifacts["cleanup_started"] = "cleanup-started.json"
    if (Path(directory) / "preclose-trace.json").exists():
        artifacts["preclose_trace"] = "preclose-trace.json"
    if (Path(directory) / "robot-receipt.json").exists():
        artifacts["native_robot_receipt"] = "robot-receipt.json"
    if (Path(directory) / "cleanup-traceback.log").exists():
        artifacts["cleanup_traceback"] = "cleanup-traceback.log"
    if (Path(directory) / "shutdown-diagnostic-signal.json").exists():
        artifacts["shutdown_diagnostic_signal"] = "shutdown-diagnostic-signal.json"
    if (Path(directory) / "worker-result-before-supervisor.json").exists():
        artifacts["worker_result_before_supervisor"] = "worker-result-before-supervisor.json"
    if (Path(directory) / "native-exit-debug-request.json").exists():
        artifacts["native_exit_debug_request"] = "native-exit-debug-request.json"
    debug_output = Path(directory) / "native-exit-debug"
    if (debug_output / "status.json").exists():
        artifacts["native_exit_debug_status"] = "native-exit-debug/status.json"
    if (debug_output / "native-stack.txt").exists():
        artifacts["native_exit_stack"] = "native-exit-debug/native-stack.txt"
    return artifacts


def _preserve_worker_result(directory):
    source = Path(directory) / "result.json"
    snapshot = Path(directory) / "worker-result-before-supervisor.json"
    if source.is_file() and not snapshot.exists():
        snapshot.write_bytes(source.read_bytes())


def _attach_sidecar_artifacts(result, directory):
    artifacts = _sidecar_artifacts(directory)
    if not artifacts:
        return result
    from dataclasses import replace
    if result.policy_kind in AGENT_POLICY_KINDS and result.calls_used is None:
        path = Path(directory) / 'call-budget.json'
        if path.is_file():
            try:
                recorded = json.loads(path.read_text())
                if (recorded.get('schema') != 'mind2act_eval.call_budget.v1'
                        or recorded.get('task_id') != result.task_id
                        or recorded.get('instance_id') != result.instance_id
                        or recorded.get('budget_version') != result.budget_version
                        or recorded.get('calls_limit') != 150
                        or recorded.get('decision_retry_limit') != 12):
                    raise ValueError('call budget receipt identity/caps differ')
                result = replace(result, **{key: recorded[key] for key in
                    ('calls_used', 'calls_limit', 'decision_retry_limit')})
            except (ValueError, KeyError, TypeError):
                # Preserve unknown counts rather than fabricating zero after a kill.
                pass
    return replace(result, artifacts={**artifacts, **result.artifacts})


def _worker_launch_argv(job, directory, argv=None):
    return [*(job.argv if argv is None else argv), "--output", str(directory)]


def _archive_attempt(directory: Path, archive: Path):
    """Archive durable evidence but skip Kit's regenerable runtime data tree."""
    directory, archive = Path(directory), Path(archive)
    temporary = archive.with_name(archive.name + ".tmp")
    try:
        with tarfile.open(temporary, "x:gz") as stream:
            def include(info):
                parts = Path(info.name).parts
                if any(part in ("cache", "tmp") for part in parts):
                    return None
                if any(parts[index:index + 2] == ("kit", "data")
                       for index in range(len(parts) - 1)):
                    return None
                return info
            stream.add(directory, arcname=directory.name, filter=include)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def _launcher_lock_root(output_root, *, require_shared=False, gpu_selection=None):
    root = Path(output_root).resolve()
    configured = os.environ.get("MIND2ACT_EVAL_LOCK_ROOT")
    if configured:
        lock_root = Path(configured).resolve()
    elif require_shared:
        raise ValueError("production batches require a stable MIND2ACT_EVAL_LOCK_ROOT")
    else:
        lock_root = root.parent / f".mind2act-eval-{os.getuid()}"
    from mind2act_eval.cluster_profile import load_cluster_profile
    profile = load_cluster_profile()
    if profile is not None and gpu_selection is not None:
        raise ValueError('PRO and Slurm storage profiles cannot be combined')
    if gpu_selection is not None:
        if not require_shared:
            raise ValueError('Slurm allocation cannot disable production storage/locking checks')
        gpu_selection.verify_storage(root, lock_root)
        if root.is_relative_to(lock_root):
            raise ValueError('Slurm output cannot be inside the canonical lock directory')
        return lock_root
    if profile is not None:
        if not require_shared:
            raise ValueError('PRO cannot disable production storage/locking checks')
        profile.verify_storage(root, lock_root)
        return lock_root
    if require_shared:
        shared_root = Path("/shared").resolve()
        if lock_root.name != f".mind2act-eval-{os.getuid()}":
            raise ValueError("production lock root must be the per-UID .mind2act-eval directory")
        if not lock_root.parent.is_relative_to(shared_root):
            raise ValueError("production lock root must be under /shared")
        if not root.is_relative_to(lock_root.parent):
            raise ValueError("production output root must be under the stable lock root's parent")
    return lock_root


@contextmanager
def _campaign_launcher_gates(lock_root, *, distributed_partition=None, legacy_cooldown_s=60., cancel_event=None,
                             clock=time, pro_admission=None):
    """Keep old exclusive campaigns compatible while allowing fixed node shards."""
    cancel_options = {} if cancel_event is None else {"cancel_event": cancel_event}
    with ExitStack() as gates:
        if pro_admission is not None:
            if Path(lock_root).resolve() != pro_admission.root.resolve():
                raise ValueError('capacity epoch requires the canonical launcher root')
            gates.enter_context(pro_admission)
            paths = startup_paths(lock_root)
            gates.enter_context(startup_compatibility(paths, cancel_event=cancel_event))
            # Legacy EX parents may die without a durable Kit timestamp.
            # Keep the queue transition wait in the same canonical Kit domain.
            with startup_gate(paths.kit, **cancel_options):
                from mind2act_eval.sim import _gate_wait
                _gate_wait(legacy_cooldown_s, clock=clock, cancel_event=cancel_event)
        elif distributed_partition is None:
            gates.enter_context(startup_gate(lock_root / "batch.lock", **cancel_options))
            paths = startup_paths(lock_root)
            gates.enter_context(startup_compatibility(paths, cancel_event=cancel_event))
        else:
            from mind2act_eval.distributed_campaign import node_lock_key
            gates.enter_context(startup_gate(lock_root / "batch.lock", shared=True, **cancel_options))
            node_lock = lock_root / f"batch-node-{node_lock_key(distributed_partition.hostname)}.lock"
            gates.enter_context(startup_gate(node_lock, **cancel_options))
            paths = startup_paths(lock_root)
            gates.enter_context(startup_compatibility(paths, cancel_event=cancel_event))
            # Older supervisors hold batch.lock exclusively but do not publish a
            # Kit timestamp. Cool down once after entering their compatibility
            # guard before this shard can launch any worker.
            with startup_gate(paths.kit, **cancel_options):
                from mind2act_eval.sim import _gate_wait
                _gate_wait(legacy_cooldown_s, clock=clock, cancel_event=cancel_event)
        yield


def _write_campaign_descriptor(lock_root, output_root, partition):
    from mind2act_eval.runner import write_json

    descriptor = {
        "schema": "mind2act_eval.distributed_campaign_receipt.v1",
        "campaign_id": partition.campaign_id,
        "output_root": str(partition.output_root),
        "lock_root_sha256": hashlib.sha256(str(partition.lock_root).encode()).hexdigest(),
        "campaign_manifest_sha256": partition.campaign_manifest_sha256,
        "partitions": [
            {"label": label, "hostname": hostname, "machine_id_sha256": machine_sha,
             "job_manifest_sha256": job_sha, "resource_config_sha256": resource_sha,
             "rss_limit_bytes": rss_limit, "max_parallel": max_parallel}
            for label, hostname, machine_sha, job_sha, resource_sha, rss_limit, max_parallel
            in partition.campaign_partitions
        ],
    }
    path = Path(output_root) / f"campaign-{partition.campaign_id}" / "campaign.json"
    with startup_gate(lock_root / f"campaign-{partition.campaign_id}.lock"):
        if path.is_symlink():
            raise ValueError("distributed campaign descriptor cannot be a symlink")
        if path.exists():
            if json.loads(path.read_text()) != descriptor:
                raise ValueError("existing campaign descriptor does not match manifest")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, descriptor)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _request_shutdown_trace(child, directory):
    directory = Path(directory)
    armed_path = directory / "cleanup-trace-armed.json"
    event = {
        "schema": "mind2act_eval.shutdown_diagnostic_signal.v1",
        "pid": child.pid,
        "signal": "SIGUSR1",
    }
    event_path = directory / "shutdown-diagnostic-signal.json"

    def record(status, reason=None, **details):
        event["status"] = status
        if reason is not None:
            event["reason"] = reason
        event.update(details)
        write_json(event_path, event)

    if not armed_path.exists():
        record("skipped", "handler_not_armed")
        return
    if child.poll() is not None:
        record("skipped", "worker_exited")
        return
    pidfd = None
    try:
        armed = json.loads(armed_path.read_text())
        if type(armed.get("pid")) is not int or armed["pid"] != child.pid:
            record("skipped", "armed_pid_mismatch", armed_pid=armed.get("pid"))
            return
        if armed.get("signal") != "SIGUSR1":
            record("skipped", "signal_not_allowed", armed_signal=armed.get("signal"))
            return
        trace_signal = signal.SIGUSR1
        if (not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")):
            record("unavailable", "pidfd_signal_not_supported")
            return
        pidfd = os.pidfd_open(child.pid)
        if child.poll() is not None:
            record("skipped", "worker_exited")
            return
        registered = _linux_sigcgt_has(child.pid, trace_signal)
        if registered is None:
            record("unavailable", "sigcgt_unreadable")
            return
        if not registered:
            record("skipped", "handler_missing")
            return
        signal.pidfd_send_signal(pidfd, trace_signal)
        record("sent", delivery="pidfd", sigcgt_verified=True)
    except ProcessLookupError:
        record("skipped", "worker_exited_before_signal")
    except Exception as exc:
        record("unavailable", "signal_verification_or_delivery_failed",
               error=f"{type(exc).__name__}: {exc}")
    finally:
        if pidfd is not None:
            os.close(pidfd)


def _linux_sigcgt_has(pid, signum):
    """Return whether Linux reports a caught signal, or None if unverifiable."""
    if sys.platform != "linux":
        return None
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("SigCgt:"):
            try:
                caught = int(line.split(":", 1)[1].strip(), 16)
            except ValueError:
                return None
            return bool(caught & (1 << (int(signum) - 1)))
    return None


@contextmanager
def parent_launch_gate(lock_root, *, interval_s=60., cancel_event=None, clock=time):
    """Wait on the actual node before Popen; retain its slot through child ready."""
    from mind2act_eval.sim import recover_startup_state
    paths = startup_paths(lock_root)
    with startup_compatibility(paths, cancel_event=cancel_event), startup_gate(paths.worker_launch, minimum_interval_s=interval_s,
            inherited_cooldown=paths.inherited_worker_cooldown, cancel_event=cancel_event, clock=clock,
            domain_check=lambda: verify_startup_domain(paths)) as mark:
        with startup_gate(paths.kit, cancel_event=cancel_event, clock=clock):
            with paths.kit.open('r+') as stream:
                recover_startup_state(stream, paths.kit, interval_s,
                    inherited_cooldown=paths.inherited_kit_cooldown, clock=clock, cancel_event=cancel_event)
        yield mark


class AttemptDeferred(Exception):
    """An intake guard refused a launch before creating any child."""


def run_attempt(job, directory: Path, *, rss_limit=RSS_LIMIT_BYTES, poll_s=.1, lock_root=None,
                cleanup_timeout_s=120., stop_grace_s=5., trace_signal_after_s=90.,
                claim_lock_fd=None, parent_start_gate=False, parent_start_interval_s=60., cancel_event=None,
                cluster_scope=None, launch_guard=None, capacity_admission=None):
    from math import isfinite
    if (not isfinite(cleanup_timeout_s) or cleanup_timeout_s <= 0
            or not isfinite(stop_grace_s) or stop_grace_s < 0
            or not isfinite(trace_signal_after_s) or trace_signal_after_s <= 0):
        raise ValueError("cleanup timeout/grace/trace signal threshold must be finite and in range")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "spec.json", job.spec)
    if job.policy_kind:
        write_json(directory / 'run-budget.json', dict(policy_kind=job.policy_kind,
            budget_version=job.budget_version, physics_ticks=job.physics_ticks,
            wall_s=job.wall_s, retries=job.retries))
    started = time.monotonic()
    peak_rss = 0
    child = None
    result = None
    exit_code = None
    seal_valid = False
    seal_receipt = None
    seal_error = None
    cleanup_started_at = None
    trace_signal_requested = False
    launch_gate = None
    initialization_gate = None
    initialization_mark = None
    initialization_handoff = None
    parallel_parent = False
    dispatch_deferred = False
    launch_wait_s = 0.
    last_gpu_monitor = float('-inf')
    peak_gpu_used_mib = None
    scratch_owner = None
    cache_guard = None
    stop_child = _stop_owned_group if cluster_scope is None else cluster_scope.stop_owned
    try:
        from mind2act_eval.cluster_profile import load_cluster_profile, check_pro_gpu
        profile = load_cluster_profile()
        if profile is not None:
            if cluster_scope is None or cluster_scope.profile.sha256 != profile.sha256:
                raise ValueError('PRO attempts require the registered canonical batch scope')
            cluster_scope.verify_job(job, dynamic_gpu=claim_lock_fd is not None and launch_guard is None)
            profile.verify_storage(directory, lock_root)
        elif cluster_scope is not None:
            raise ValueError('campaign scope requires its explicit PRO profile')
        from mind2act_eval.episode_storage import prepare_scratch, node_cache, cache_use_lock
        scratch_owner = prepare_scratch(directory)
        fallback_cache = Path(lock_root).absolute().parent / '.mind2act-cache' if lock_root else directory.parent / '.mind2act-cache'
        shared_cache = node_cache(directory, fallback_root=fallback_cache)
        cache_guard = cache_use_lock(shared_cache, cancel_event=cancel_event)
        cache_fd = cache_guard.__enter__()
        if parent_start_gate:
            if lock_root is None:
                raise ValueError("resident parent startup requires canonical lock_root")
            waiting = time.monotonic()
            parallel_parent = profile is not None and profile.startup_capacity == 2
            if parallel_parent:
                from mind2act_eval import pro_startup
                paths = startup_paths(lock_root)
                pending_initialization = pro_startup.parent_lease(profile, paths, job.gpu, directory,
                                                                  cancel_event=cancel_event)
                initialization_mark = pending_initialization.__enter__()
                initialization_gate = pending_initialization
                initialization_handoff = pro_startup.handoff_environment(profile, paths, job.gpu,
                                                                         directory, initialization_mark)
                pending_gate = pro_startup.dispatch_gate(paths, interval_s=parent_start_interval_s,
                                                         cancel_event=cancel_event, profile=profile)
            else:
                pending_gate = parent_launch_gate(lock_root, interval_s=parent_start_interval_s, cancel_event=cancel_event)
            mark_launch = pending_gate.__enter__()
            launch_gate = pending_gate
            launch_wait_s = time.monotonic() - waiting
            started = time.monotonic()
        peak_rss = user_rss_bytes() if cluster_scope is None else cluster_scope.snapshot()['rss_bytes']
        if peak_rss >= rss_limit:
            result = _invalid(job, Termination.RESOURCE_LIMIT, "user RSS budget unavailable before launch")
        else:
            if lock_root is None:
                if job.spec.task_id != "demo":
                    raise ValueError("native attempts must be launched by a batch with a shared lock root")
                lock_root = directory.parent / f".mind2act-eval-{os.getuid()}"
            lock_root = Path(lock_root).resolve()
            if job.gpu is not None and capacity_admission is None:
                if profile is not None:
                    admission = cluster_scope.check_gpu(job.gpu)
                else:
                    admission = check_gpu_idle(job.gpu)
                if profile is not None:
                    peak_gpu_used_mib = admission['device']['used_mib']
                    write_json(directory / 'pro-admission.json', dict(admission,
                        profile_sha256=profile.sha256, resources=cluster_scope.snapshot()))
            environment = dict(os.environ)
            if job.gpu is not None:
                from mind2act_eval.gpu_allocation import worker_cuda_mask
                environment["CUDA_VISIBLE_DEVICES"] = worker_cuda_mask(job.gpu, job.argv, environment)
            for key, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                                  ("CUDA_CACHE_PATH", "cache/cuda"), ("PYTHONPYCACHEPREFIX", "cache/pycache")):
                path = directory / relative
                path.mkdir(parents=True, exist_ok=True)
                environment[key] = str(path)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["CUDA_CACHE_PATH"] = str(shared_cache / "cuda")
            environment["MIND2ACT_EVAL_NODE_CACHE"] = str(shared_cache)
            environment["MIND2ACT_EVAL_CACHE_LOCK_FD"] = str(cache_fd)
            environment["MIND2ACT_EVAL_LOCK_ROOT"] = str(lock_root)
            if initialization_handoff is not None:
                environment[pro_startup.LEASE_ENV] = initialization_handoff
            argv = list(job.argv)
            if "--spec" in argv:
                argv[argv.index("--spec") + 1] = str(directory / "spec.json")
            with (directory / "worker.log").open("wb") as log:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("resident cancelled before Popen")
                if claim_lock_fd is not None and (type(claim_lock_fd) is not int or claim_lock_fd < 0):
                    raise ValueError("claim_lock_fd must be a nonnegative file descriptor")
                launch_options = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      env=environment, start_new_session=True, cwd=directory,
                                      pass_fds=(cache_fd,) if claim_lock_fd is None else (claim_lock_fd, cache_fd))
                if initialization_mark is not None:
                    launch_options['pass_fds'] += initialization_mark.lease_fds
                def prepare_launch():
                    if profile is not None:
                        cluster_scope.verify_job(job, dynamic_gpu=claim_lock_fd is not None and launch_guard is None)
                    if initialization_mark is not None:
                        # Commit close-only ownership before Popen: even a
                        # registration error must not unlock a spawned child.
                        initialization_mark.handoff()
                # Keep final selected validation, handoff and PID registration
                # inside the rolling guard, after any cooperative capacity wait.
                with nullcontext() if launch_guard is None else launch_guard():
                    if capacity_admission is not None:
                        from mind2act_eval.contracts import to_json
                        from mind2act_eval.pro_device_leases import CapacityUnavailable, CapacityLaunchError
                        try:
                            child, _identity = capacity_admission.launch(job.gpu, cluster_scope,
                                _worker_launch_argv(job, directory, argv), receipt_path=directory/'pro-admission.json',
                                case_key=dict(spec=to_json(job.spec), consumer=job.consumer_kind.value),
                                wait_for_slot=launch_guard is None, cancel_event=cancel_event,
                                before_launch=prepare_launch, **launch_options)
                        except CapacityLaunchError as exc:
                            child = exc.child  # Preserve real ownership even if durable commit failed.
                            raise
                        except CapacityUnavailable as exc:
                            raise AttemptDeferred(str(exc)) from exc
                        peak_gpu_used_mib = capacity_admission.receipts[child.pid]['device'].get('used_mib')
                    else:
                        prepare_launch()
                        if cluster_scope is None:
                            child = subprocess.Popen(_worker_launch_argv(job, directory, argv), **launch_options)
                        else:
                            child, _identity = cluster_scope.launch(_worker_launch_argv(job, directory, argv), **launch_options)
                started = time.monotonic()
                if launch_gate is not None:
                    mark_launch()
                    write_json(directory / "parent-launch.json", {"pid": child.pid, "parent_wait_s": launch_wait_s,
                                                                  "child_budget_started_at_monotonic": started})
                if parallel_parent:
                    initialization_gate.__exit__(None, None, None)
                    initialization_gate = None
                    launch_gate.__exit__(None, None, None)
                    launch_gate = None
                while True:
                    if cluster_scope is not None:
                        resource_snapshot = cluster_scope.snapshot()
                        if time.monotonic() - last_gpu_monitor >= 2:
                            monitored = (cluster_scope.check_gpu(job.gpu, require_idle=False, snapshot=resource_snapshot)
                                if capacity_admission is None else capacity_admission.monitor_gpu(job.gpu, cluster_scope))
                            peak_gpu_used_mib = max(peak_gpu_used_mib or 0, monitored['device']['used_mib'])
                            last_gpu_monitor = time.monotonic()
                    if cancel_event is not None and cancel_event.is_set():
                        result = _invalid(job, Termination.KILLED, "resident driver stopping owned worker")
                        break
                    if launch_gate is not None and (directory / "ready.json").is_file():
                        launch_gate.__exit__(None, None, None)
                        launch_gate = None
                    if (directory / SEAL_NAME).exists():
                        try:
                            # Early validation lets the supervisor react to a
                            # corrupt seal, but is provisional until the child
                            # and its output writes have stopped.
                            validate_worker_seal(directory, expected_consumer=job.consumer_kind)
                        except Exception as exc:
                            seal_error = f"{type(exc).__name__}: {exc}"
                            result = _invalid(job, Termination.ERROR, "worker seal validation failed: " + seal_error)
                            break
                        try:
                            child.wait(timeout=1.)
                        except subprocess.TimeoutExpired:
                            stop_child(child, grace_s=stop_grace_s)
                        exit_code = child.returncode
                        try:
                            result, seal_receipt = validate_worker_seal(directory, durable=True, expected_consumer=job.consumer_kind)
                            seal_valid = True
                        except Exception as exc:
                            seal_error = f"{type(exc).__name__}: {exc}"
                            result = _invalid(job, Termination.ERROR, "final worker seal validation failed: " + seal_error)
                        break
                    current_rss = user_rss_bytes() if cluster_scope is None else resource_snapshot['rss_bytes']
                    peak_rss = max(peak_rss, current_rss)
                    if peak_rss > rss_limit:
                        result = _invalid(job, Termination.RESOURCE_LIMIT, "user RSS exceeded budget")
                        break
                    from mind2act_eval.runner.observation import observation_wall_s
                    # Reserve the known environment allowance before its receipt
                    # exists, so polling cannot kill the actor-to-observer handoff.
                    # run_episode still enforces the original actor wall budget.
                    environment_wall = (observation_wall_s(job.spec)
                        if job.consumer_kind != ConsumerKind.DIAGNOSTIC else 0.)
                    if time.monotonic() - started > job.wall_s + environment_wall:
                        result = _invalid(job, Termination.WALL_TIMEOUT, "supervisor wall budget exhausted")
                        break
                    if child.poll() is not None:
                        break
                    if (directory / "cleanup-started.json").exists():
                        if cleanup_started_at is None:
                            cleanup_started_at = time.monotonic()
                        elapsed_cleanup = time.monotonic() - cleanup_started_at
                        if (not trace_signal_requested
                                and elapsed_cleanup >= trace_signal_after_s):
                            _request_shutdown_trace(child, directory)
                            trace_signal_requested = True
                        if elapsed_cleanup > cleanup_timeout_s:
                            _preserve_worker_result(directory)
                            detail = f"worker cleanup exceeded {cleanup_timeout_s:g}s"
                            checkpoint = directory / "preclose-trace.json"
                            if checkpoint.exists():
                                trace = json.loads(checkpoint.read_text())
                                preclose_error = trace.get("result", {}).get("error")
                                if preclose_error:
                                    detail += f"; pre-close error: {preclose_error}"
                            result = _invalid(job, Termination.ERROR, detail)
                            result = _attach_sidecar_artifacts(result, directory)
                            break
                    time.sleep(poll_s)
                if child.poll() is None:
                    stop_child(child, grace_s=stop_grace_s)
                exit_code = child.returncode
            # Shared filesystem metadata can lag the child's process-exit notice.
            if result is None:
                seal_deadline = time.monotonic() + 1.0
                while not (directory / SEAL_NAME).exists() and time.monotonic() < seal_deadline:
                    time.sleep(min(poll_s, .02))
                if (directory / SEAL_NAME).exists():
                    try:
                        result, seal_receipt = validate_worker_seal(directory, durable=True, expected_consumer=job.consumer_kind)
                        seal_valid = True
                    except Exception as exc:
                        seal_error = f"{type(exc).__name__}: {exc}"
                        result = _invalid(job, Termination.ERROR, "worker seal validation failed: " + seal_error)
            if result is None:
                if exit_code and exit_code > 0 and (directory / "result.json").exists():
                    result = Result(**json.loads((directory / "result.json").read_text()))
                    if result.infra_valid:
                        raise ValueError("worker returned valid score without a verified completion seal")
                elif exit_code != 0:
                    _preserve_worker_result(directory)
                    from mind2act_eval.runtime_diagnostics import native_phase_timeout
                    phase_timeout = native_phase_timeout(directory, child.pid)
                    result = (_invalid(job, Termination.WALL_TIMEOUT, phase_timeout) if phase_timeout else
                              _invalid(job, Termination.KILLED if exit_code < 0 else Termination.ERROR,
                                       f"worker exit {exit_code}"))
                else:
                    path = directory / "result.json"
                    if path.exists():
                        result = Result(**json.loads(path.read_text()))
                        if result.infra_valid:
                            raise ValueError("worker returned valid score without a verified completion seal")
                    else:
                        checkpoint = directory / "preclose-trace.json"
                        detail = "worker exited before writing final result.json"
                        if checkpoint.exists():
                            trace = json.loads(checkpoint.read_text())
                            detail += f"; pre-close stage={trace.get('stage')!r}, tick={trace.get('tick')!r}"
                        else:
                            detail += "; no pre-close trace was written"
                        result = _invalid(job, Termination.ERROR, detail)
            result = _attach_sidecar_artifacts(result, directory)
            if seal_valid:
                try:
                    from mind2act_eval.contracts import InstanceSpec
                    job.spec.assert_binding(InstanceSpec.from_dict(json.loads((directory / "spec.json").read_text())))
                    if (result.task_id, result.instance_id, result.consumer_kind) != (job.spec.task_id, job.spec.instance_id, job.consumer_kind):
                        raise ValueError("worker result identity/consumer mismatch")
                    _validate_result_budget(job, result)
                except Exception:
                    seal_valid = False
                    seal_receipt = None
                    raise
    except Exception as exc:
        dispatch_deferred = isinstance(exc, AttemptDeferred) and child is None
        _preserve_worker_result(directory)
        result = _invalid(job, Termination.ERROR, f"{type(exc).__name__}: {exc}")
        result = _attach_sidecar_artifacts(result, directory)
    finally:
        if child is not None and child.poll() is None:
            stop_child(child, grace_s=stop_grace_s)
        worker_exit_verified = child is None or child.poll() is not None
        if launch_gate is not None:
            launch_gate.__exit__(None, None, None)
        if initialization_gate is not None:
            initialization_gate.__exit__(None, None, None)
        if cluster_scope is not None and child is not None:
            peak_rss = max(peak_rss, cluster_scope.peak_rss_bytes)
            try:
                cluster_scope.assert_worker_exited(child.pid)
            except Exception as exc:
                worker_exit_verified = False
                seal_valid = False
                result = _invalid(job, Termination.RESOURCE_LIMIT, str(exc))
        elif child is not None and worker_exit_verified:
            from mind2act_eval.episode_storage import process_group_exited
            try:
                worker_exit_verified = process_group_exited(child.pid)
                if not worker_exit_verified:
                    stop_child(child, grace_s=stop_grace_s)
                    deadline = time.monotonic() + min(cleanup_timeout_s, stop_grace_s + 1.)
                    while not worker_exit_verified and time.monotonic() < deadline:
                        worker_exit_verified = process_group_exited(child.pid)
                        if not worker_exit_verified:
                            time.sleep(min(poll_s, .05))
            except Exception:
                worker_exit_verified = False
            if not worker_exit_verified:
                _preserve_worker_result(directory)
                seal_valid = False
                seal_receipt = None
                detail = 'owned worker process-group exit unknown; retain attempt without retry'
                if result is None or result.infra_valid:
                    result = _invalid(job, Termination.RESOURCE_LIMIT, detail)
                else:
                    from dataclasses import replace
                    result = replace(result, error=f'{result.error}; {detail}')
        if child is not None and worker_exit_verified and seal_valid:
            # A sealed leader can exit while a descendant is still writing.
            # Only the complete stopped tree makes this the final seal check.
            try:
                final_result, final_seal = validate_worker_seal(directory, durable=True,
                                                               expected_consumer=job.consumer_kind)
                from mind2act_eval.contracts import InstanceSpec
                job.spec.assert_binding(InstanceSpec.from_dict(json.loads((directory / "spec.json").read_text())))
                if (final_result.task_id, final_result.instance_id, final_result.consumer_kind) != (
                        job.spec.task_id, job.spec.instance_id, job.consumer_kind):
                    raise ValueError("worker result identity/consumer mismatch")
                _validate_result_budget(job, final_result)
                result = _attach_sidecar_artifacts(final_result, directory)
                seal_receipt = final_seal
            except Exception as exc:
                _preserve_worker_result(directory)
                seal_valid = False
                seal_receipt = None
                seal_error = f'{type(exc).__name__}: {exc}'
                result = _invalid(job, Termination.ERROR, 'post-exit worker seal validation failed: ' + seal_error)
        if result is not None and not seal_valid:
            write_json(directory / "result.json", result)
        scratch_cleanup = None
        if scratch_owner is not None and child is not None and worker_exit_verified and seal_valid:
            from mind2act_eval.episode_storage import cleanup_scratch
            try:
                scratch_cleanup = cleanup_scratch(directory, scratch_owner, worker_pid=child.pid,
                    exited=True, expected_consumer=job.consumer_kind)
            except Exception as exc:
                # Keep the original score/error/log intact; cleanup failure is a
                # separate retained supervisor diagnostic, never silent deletion.
                from mind2act_eval.episode_storage import durable_json, RECEIPT
                scratch_cleanup = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
                durable_json(directory / RECEIPT, scratch_cleanup)
        if cache_guard is not None:
            cache_guard.__exit__(None, None, None)
        capacity_release = None
        if capacity_admission is not None and child is not None and worker_exit_verified:
            try:
                capacity_release = capacity_admission.worker_exited(child)
            except Exception as exc:
                capacity_release = dict(status='retained', error=f'{type(exc).__name__}: {exc}')
        write_json(directory / "batch-exit.json", {
            "worker_exit_verified": worker_exit_verified,
            **({} if launch_guard is None else dict(worker_started=child is not None, dispatch_deferred=dispatch_deferred)),
            "exit_code": exit_code, "peak_user_rss_bytes": peak_rss,
            "rss_limit_bytes": rss_limit, "wall_s": time.monotonic() - started,
            "cleanup_timeout_s": cleanup_timeout_s,
            "worker_seal_valid": seal_valid,
            "worker_teardown": seal_receipt.get("teardown") if seal_receipt else None,
            "worker_seal_sha256": hashlib.sha256((directory / SEAL_NAME).read_bytes()).hexdigest()
            if (directory / SEAL_NAME).is_file() else None,
            "worker_seal_error": seal_error,
            "scratch_cleanup": scratch_cleanup,
            "parent_launch_wait_s": launch_wait_s,
            **({} if capacity_admission is None else {'capacity_lease_release':capacity_release}),
            **({} if cluster_scope is None else {
                'peak_sampled_gpu_used_mib': peak_gpu_used_mib,
                'rss_scope': 'registered campaign trees and supervisor; not host UID RSS',
                'profile_sha256': cluster_scope.profile.sha256}),
        })
    return result


def run_batch(jobs, output_root: Path, *, rss_limit=RSS_LIMIT_BYTES, require_shared=True,
              max_parallel=1, start_interval_s=60., cleanup_timeout_s=120., resource_config=None,
              distributed_partition=None):
    from concurrent.futures import ThreadPoolExecutor
    from mind2act_eval.cluster_profile import load_cluster_profile
    profile = load_cluster_profile()
    resource_options = (rss_limit, max_parallel, resource_config)
    if profile is None:
        rss_limit, max_parallel, _ = resolve_resources(*resource_options)
    else:
        if (not require_shared or distributed_partition is not None or resource_config is not None
                or type(rss_limit) is not int or rss_limit != RSS_LIMIT_BYTES
                or type(max_parallel) is not int or max_parallel not in (1, profile.max_parallel)):
            raise ValueError('PRO profile requires canonical serial admission or its pinned parallel manifest')
        rss_limit = profile.rss_limit_bytes
        max_parallel = profile.max_parallel
    jobs = list(jobs)
    protocols = {(job.policy_kind, job.budget_version) for job in jobs}
    if len(protocols) > 1:
        raise ValueError('separate batches are required for different policy budget versions')
    if profile is not None:
        profile.verify_jobs(jobs)  # Reject substitutions before storage/cgroup/worker side effects.
    from mind2act_eval.gpu_allocation import optional_gpu_selection
    selections = [optional_gpu_selection(job.gpu, job.argv, os.environ) for job in jobs]
    gpu_selection = next((selection for selection in selections if selection is not None), None)
    if gpu_selection is not None:
        if (not require_shared or distributed_partition is not None or max_parallel != 1
                or any(selection is None for selection in selections)
                or len({selection.receipt_sha256 for selection in selections}) != 1):
            raise ValueError('explicit Slurm allocation requires one serial production storage domain')
    root = Path(output_root).resolve()
    if profile is not None and gpu_selection is not None:
        raise ValueError('PRO and Slurm allocation profiles cannot be combined')
    shared_root = (profile.storage_root if profile is not None else
                   gpu_selection.storage_root if gpu_selection is not None else Path('/shared').resolve())
    if require_shared and not root.is_relative_to(shared_root):
        raise ValueError("production outputs must be under /shared")
    if require_shared and start_interval_s < 60:
        raise ValueError("production Kit starts require >=60s spacing")
    lock_root = _launcher_lock_root(root, require_shared=require_shared,
        **({} if gpu_selection is None else {'gpu_selection': gpu_selection}))
    if distributed_partition is not None:
        from mind2act_eval.distributed_campaign import CampaignPartition
        if not isinstance(distributed_partition, CampaignPartition):
            raise TypeError("distributed_partition must be a validated CampaignPartition")
        if not require_shared or not start_interval_s >= 60:
            raise ValueError("distributed campaigns require production shared storage and 60s starts")
        requested_output = Path(os.path.abspath(output_root))
        requested_lock = Path(os.path.abspath(os.environ.get("MIND2ACT_EVAL_LOCK_ROOT", "")))
        if (requested_output != distributed_partition.output_root
                or requested_lock != distributed_partition.lock_root):
            raise ValueError("distributed campaign output/lock roots differ from its pinned manifest")
        shared_root = Path("/shared").resolve()
        if not all(path.is_relative_to(shared_root) for path in (
                distributed_partition.campaign_manifest_path,
                distributed_partition.manifest_path,
                distributed_partition.resource_config_path)):
            raise ValueError("distributed campaign manifests must be under /shared")
        if resource_config is None:
            raise ValueError("distributed campaigns require an explicit per-node resource config")
        if (resource_config.source_sha256 != distributed_partition.resource_config_sha256
                or resource_config.rss_limit_bytes != distributed_partition.resource_config.rss_limit_bytes
                or resource_config.max_parallel != distributed_partition.resource_config.max_parallel):
            raise ValueError("distributed resource config differs from the campaign manifest")
        entries = json.loads(distributed_partition.manifest_bytes)
        expected_episodes = {
            (entry["episode_key"]["task_id"], entry["episode_key"]["instance_id"],
             entry["episode_key"]["difficulty"]) for entry in entries
        }
        actual_episodes = {(job.spec.task_id, job.spec.instance_id, job.spec.difficulty) for job in jobs}
        if len(jobs) != len(entries) or actual_episodes != expected_episodes:
            raise ValueError("distributed jobs do not match the validated campaign partition")
        campaign_root = root / f"campaign-{distributed_partition.campaign_id}"
        _write_campaign_descriptor(lock_root, root, distributed_partition)
        campaign = campaign_root / distributed_partition.label
    else:
        campaign = root / ("batch-" + uuid.uuid4().hex)
    summary = []
    from mind2act_eval.pro_device_leases import ProCampaignAdmission
    capacity_admission = ProCampaignAdmission.from_env(profile)
    # Legacy campaigns take LOCK_EX on batch.lock for their full duration.
    # Distributed shards take LOCK_SH there, then one host/UID lock each.
    with _campaign_launcher_gates(lock_root, distributed_partition=distributed_partition,
            pro_admission=capacity_admission), ExitStack() as scopes, \
            ThreadPoolExecutor(max_workers=max_parallel) as pool:
        campaign.mkdir(parents=True, exist_ok=False)
        cluster_scope = None
        if profile is not None:
            from mind2act_eval.campaign_resources import CampaignResources
            cluster_scope = scopes.enter_context(CampaignResources(profile, campaign / 'pro-resources.json'))
            if capacity_admission is not None:
                scopes.callback(capacity_admission.close)  # Release while the owned cgroup still exists.
        gpu_assignments = [{"episode_index": i, "gpu": job.gpu} for i, job in enumerate(jobs)]
        try:
            if cluster_scope is None:
                _, _, resource_receipt = resolve_resources(*resource_options)
            else:
                resource_receipt = dict(cluster_scope.snapshot(), profile_sha256=profile.sha256,
                    max_parallel=max_parallel, max_workers_per_gpu=profile.max_workers_per_gpu,
                    startup_capacity=profile.startup_capacity,
                    startup_host_capacity=getattr(profile, 'startup_host_capacity', 2),
                    rss_limit_bytes=rss_limit,
                    job_manifest=profile.job_manifest, device_pool=list(profile.devices),
                    budget_semantics='registered campaign trees plus supervisor RSS; kernel cgroup containment; host and ancestor hard gates')
        except ResourceAdmissionError as exc:
            write_json(campaign / "resource-receipt.json", dict(exc.receipt, gpu_assignments=gpu_assignments))
            raise
        resource_receipt["gpu_assignments"] = gpu_assignments
        if distributed_partition is not None:
            resource_receipt["distributed_campaign"] = {
                "campaign_id": distributed_partition.campaign_id,
                "partition": distributed_partition.label,
                "hostname": distributed_partition.hostname,
                "machine_id_sha256": distributed_partition.machine_id_sha256,
                "lock_root_sha256": hashlib.sha256(str(distributed_partition.lock_root).encode()).hexdigest(),
                "campaign_manifest_sha256": distributed_partition.campaign_manifest_sha256,
                "job_manifest_sha256": distributed_partition.manifest_sha256,
                "resource_config_sha256": distributed_partition.resource_config_sha256,
                "campaign_partitions": [
                    {"label": label, "hostname": hostname, "machine_id_sha256": machine_sha,
                     "job_manifest_sha256": job_sha, "resource_config_sha256": resource_sha,
                     "rss_limit_bytes": node_rss, "max_parallel": node_parallel}
                    for label, hostname, machine_sha, job_sha, resource_sha, node_rss, node_parallel
                    in distributed_partition.campaign_partitions
                ],
                "lock_scope": "uid-and-node",
            }
        write_json(campaign / "resource-receipt.json", resource_receipt)
        pending = list(enumerate(jobs))
        active = []
        attempt_counts = {}
        last_start = float("-inf")
        previous_ready = None
        while pending or active:
            for item in list(active):
                index, job, directory, future = item
                if not future.done():
                    continue
                result = future.result()
                exit_receipt = directory / 'batch-exit.json'
                if exit_receipt.is_file() and json.loads(exit_receipt.read_text()).get('worker_exit_verified') is not True:
                    raise RuntimeError('worker exit unknown; retain attempt without releasing a batch slot or retrying')
                archive = directory.with_suffix(".tar.gz")
                archive_error = None
                try:
                    _archive_attempt(directory, archive)
                except Exception as exc:
                    archive_error = f"{type(exc).__name__}: {exc}"
                summary.append({"episode_index": index, "directory": directory.name,
                                "archive": archive.name if archive_error is None else None,
                                "archive_error": archive_error, "result": result})
                write_json(campaign / "summary.json", summary)
                active.remove(item)
                if not result.infra_valid and result.termination in (Termination.KILLED, Termination.RESOURCE_LIMIT, Termination.WALL_TIMEOUT) and attempt_counts[index] <= job.retries:
                    pending.append((index, job))
            previous_stable = previous_ready is None or previous_ready.exists() or not any(item[2] == previous_ready.parent for item in active)
            # Legacy profiles keep one worker per card. Explicitly pinned
            # sharing allows at most three; startup gates and scope ownership
            # remain common to the entire campaign.
            gpu_counts = {}
            for item in active:
                gpu_counts[item[1].gpu] = gpu_counts.get(item[1].gpu, 0) + 1
            candidate = next((i for i, (_, job) in enumerate(pending)
                              if job.gpu is None or gpu_counts.get(job.gpu, 0) <
                              (profile.max_workers_per_gpu if profile is not None else 1)), None)
            if profile is not None and profile.startup_capacity > 1:
                starting = [item for item in active if not (item[2] / 'ready.json').is_file()]
                starting_gpus = {item[1].gpu for item in starting}
                # Scan past a same-card/unready candidate to the next free card.
                candidate = next((i for i, (_, job) in enumerate(pending)
                                  if job.gpu not in starting_gpus
                                  and gpu_counts.get(job.gpu, 0) < profile.max_workers_per_gpu), None)
                previous_stable = len(starting) < getattr(profile, 'startup_host_capacity', 2)
            if candidate is not None and len(active) < max_parallel and previous_stable and time.monotonic() - last_start >= start_interval_s:
                index, job = pending.pop(candidate)
                attempt_counts[index] = attempt_counts.get(index, 0) + 1
                directory = campaign / f"episode-{index:04d}-attempt-{attempt_counts[index]:02d}"
                active.append((index, job, directory, pool.submit(run_attempt, job, directory,
                                                                   rss_limit=rss_limit,
                                                                   lock_root=lock_root,
                                                                   cleanup_timeout_s=cleanup_timeout_s,
                                                                   **({} if cluster_scope is None else {'cluster_scope': cluster_scope}),
                                                                   **({} if capacity_admission is None else {'capacity_admission':capacity_admission}))))
                previous_ready = directory / "ready.json"
                last_start = time.monotonic()
            if pending or active:
                time.sleep(.1)
    return campaign, summary
