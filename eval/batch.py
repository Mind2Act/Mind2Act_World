"""Serial child supervisor with bounded retries and immutable attempts.

Only process groups created here are terminated. Never search command lines or
signal pre-existing user processes. All output/cache/temp files live externally.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import nullcontext

from mind2act_eval.contracts import ConsumerKind, Result, Termination
from mind2act_eval.runner import write_json
from mind2act_eval.budgets import AGENT_POLICY_KINDS
from mind2act_eval.sealing import SEAL_NAME, validate_worker_seal

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


class AttemptDeferred(Exception):
    """An intake guard refused a launch before creating any child."""


def run_attempt(job, directory: Path, *, poll_s=.1,
                cleanup_timeout_s=120., stop_grace_s=5., trace_signal_after_s=90.,
                claim_lock_fd=None, cancel_event=None,
                launch_guard=None):
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
    child = None
    result = None
    exit_code = None
    seal_valid = False
    seal_receipt = None
    seal_error = None
    cleanup_started_at = None
    trace_signal_requested = False
    dispatch_deferred = False
    scratch_owner = None
    cache_guard = None
    stop_child = _stop_owned_group
    try:
        from mind2act_eval.episode_storage import prepare_scratch, node_cache, cache_use_lock
        scratch_owner = prepare_scratch(directory)
        fallback_cache = directory.parent / '.mind2act-cache'
        shared_cache = node_cache(directory, fallback_root=fallback_cache)
        cache_guard = cache_use_lock(shared_cache, cancel_event=cancel_event)
        cache_fd = cache_guard.__enter__()
        environment = dict(os.environ)
        if job.gpu is not None:
            if type(job.gpu) is not int or job.gpu < 0:
                raise ValueError("explicit nonnegative physical GPU index required")
            environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        for key, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                              ("CUDA_CACHE_PATH", "cache/cuda"), ("PYTHONPYCACHEPREFIX", "cache/pycache")):
            path = directory / relative
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUDA_CACHE_PATH"] = str(shared_cache / "cuda")
        environment["MIND2ACT_EVAL_NODE_CACHE"] = str(shared_cache)
        environment["MIND2ACT_EVAL_CACHE_LOCK_FD"] = str(cache_fd)
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
            # Keep final selected validation, handoff and PID registration
            # inside the rolling guard, after any cooperative capacity wait.
            with nullcontext() if launch_guard is None else launch_guard():
                child = subprocess.Popen(_worker_launch_argv(job, directory, argv), **launch_options)
            started = time.monotonic()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    result = _invalid(job, Termination.KILLED, "resident driver stopping owned worker")
                    break
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
        if child is not None and worker_exit_verified:
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
        write_json(directory / "batch-exit.json", {
            "worker_exit_verified": worker_exit_verified,
            **({} if launch_guard is None else dict(worker_started=child is not None, dispatch_deferred=dispatch_deferred)),
            "exit_code": exit_code, "wall_s": time.monotonic() - started,
            "cleanup_timeout_s": cleanup_timeout_s,
            "worker_seal_valid": bool(seal_valid and seal_receipt and seal_receipt.get("status") == "sealed"),
            "worker_completion_valid": seal_valid,
            "worker_teardown": seal_receipt.get("teardown") if seal_receipt else None,
            "worker_seal_sha256": hashlib.sha256((directory / SEAL_NAME).read_bytes()).hexdigest()
            if (directory / SEAL_NAME).is_file() else None,
            "worker_seal_error": seal_error,
            "scratch_cleanup": scratch_cleanup,
        })
    return result
