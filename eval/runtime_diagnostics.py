"""Small shutdown diagnostics for fresh episode workers."""
from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path

_TRACE_HANDLES = []


def checkpoint_prerun_failure(output, error, *, phase, source_file, spec=None):
    """Keep the first setup exception durable before close can block or crash.

    This is an invalid diagnostic checkpoint, never a completed worker Result.
    Constructor callers recover case identity from the parent's original spec.
    Failure to persist is reported and cannot replace the original exception.
    """
    try:
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        sys.stderr.flush()
    except BaseException:
        pass
    try:
        from mind2act_eval.episode_storage import durable_json
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'preclose-trace.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if (prior.get('schema') != 'mind2act_eval.preclose_trace.v1'
                    or prior.get('pid') != os.getpid()
                    or prior.get('phase') not in ('constructor', 'worker_setup')):
                raise ValueError('refusing to replace an existing pre-close checkpoint')
            if spec is not None and (prior['result'].get('task_id'), prior['result'].get('instance_id')) != (spec.task_id, spec.instance_id):
                raise ValueError('existing pre-close checkpoint belongs to another case')
            identity = prior['result']
            # Re-entered CLI cleanup preserves the constructor's first error.
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        else:
            context = {'task_id': None, 'instance_id': None}
            provenance_error = None
            try:
                if spec is None:
                    context.update(json.loads((output / 'spec.json').read_text()))
                else:
                    context.update(task_id=spec.task_id, instance_id=spec.instance_id)
            except Exception as exc:
                provenance_error = f'{type(exc).__name__}: {exc}'
            source = Path(source_file).resolve()
            source_digest, source_error = None, None
            try:
                source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError as exc:
                source_error = f'{type(exc).__name__}: {exc}'
            identity = {'task_id': context['task_id'], 'instance_id': context['instance_id']}
            durable_json(path, {
                'schema': 'mind2act_eval.preclose_trace.v1', 'stage': 'awaiting_cleanup',
                'phase': phase, 'pid': os.getpid(), 'tick': 0,
                'source_file': str(source), 'source_sha256': source_digest, 'source_provenance_error': source_error,
                'spec_provenance_error': provenance_error,
                'exception': {'type': type(error).__name__, 'message': str(error),
                    'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__))},
                'result': {'task_id': context['task_id'], 'instance_id': context['instance_id'],
                    'error': f'{type(error).__name__}: {error}', 'infra_valid': False, 'score': None,
                    'model_tested': False, 'ticks': 0},
            })
        marker = output / 'cleanup-started.json'
        if marker.exists():
            prior_marker = json.loads(marker.read_text())
            if (prior_marker.get('schema') != 'mind2act_eval.cleanup_started.v1'
                    or prior_marker.get('pid') != os.getpid()
                    or prior_marker.get('phase') not in ('constructor', 'worker_setup')):
                raise ValueError('refusing to replace an existing cleanup marker')
        else:
            durable_json(marker, {'schema': 'mind2act_eval.cleanup_started.v1',
                'pid': os.getpid(), 'phase': phase, 'tick': 0,
                'task_id': identity['task_id'], 'instance_id': identity['instance_id']})
        # Also commits the first checkpoint's name when re-entering after a
        # previous directory-fsync failure. No saved-success claim precedes it.
        fd = os.open(output, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except BaseException as diagnostic_error:
        try:
            print(f'pre-run checkpoint failed: {type(diagnostic_error).__name__}: {diagnostic_error}', file=sys.stderr)
            sys.stderr.flush()
        except BaseException:
            pass
        return False


def arm_cleanup_trace(path: Path, *, timeout_s: int = 90) -> Path:
    """Keep fatal and delayed all-thread traces open through interpreter shutdown."""
    if type(timeout_s) is not int or timeout_s <= 0:
        raise ValueError("cleanup trace timeout must be a positive integer")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", buffering=1)
    _TRACE_HANDLES.append(stream)
    faulthandler.enable(file=stream, all_threads=True)
    faulthandler.dump_traceback_later(timeout_s, repeat=True, file=stream)
    trace_signal = getattr(signal, "SIGUSR1", None)
    if trace_signal is not None:
        faulthandler.register(trace_signal, file=stream, all_threads=True, chain=False)
    return path


def cleanup_trace_signal_name() -> str | None:
    """Return the signal registered for an out-of-process shutdown dump."""
    return "SIGUSR1" if getattr(signal, "SIGUSR1", None) is not None else None


def run_native_phase(action, output, *, phase, timeout_s):
    """Bound one native call in a fresh worker, even while its GIL is held.

    The C-level faulthandler watchdog dumps all Python threads and exits this
    worker on timeout. The supervisor retains the receipt and publishes an
    invalid/null-score timeout; no Python timer thread or signal handler is needed.
    Call before cleanup diagnostics, which arm their own watchdog afterwards.
    """
    import math
    import time
    from .episode_storage import durable_json
    if (not isinstance(phase, str) or not phase.strip() or isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError('native phase requires a name and positive finite timeout')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'native-phase.json'
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)
    started = time.monotonic()
    receipt = dict(schema='mind2act_eval.native_phase.v1', pid=os.getpid(), phase=phase,
                   timeout_s=timeout_s, status='running', started_monotonic=started)
    with (output / 'native-phase-traceback.log').open('x') as stream:
        durable_json(receipt_path, receipt)
        faulthandler.dump_traceback_later(timeout_s, file=stream, exit=True)
        try:
            result = action()
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        else:
            receipt['status'] = 'completed'
            return result
        finally:
            faulthandler.cancel_dump_traceback_later()
            receipt['elapsed_s'] = time.monotonic() - started
            try:
                durable_json(receipt_path, receipt)
            except Exception:
                if receipt['status'] != 'failed':
                    raise
                traceback.print_exc(file=sys.stderr)


def native_phase_timeout(output, worker_pid):
    """Identify only a timed-out phase owned by the supervised child that exited."""
    import math
    output = Path(output)
    try:
        receipt = json.loads((output / 'native-phase.json').read_text())
        if not isinstance(receipt, dict):
            return None
        timeout = receipt['timeout_s']
        if (receipt.get('schema') != 'mind2act_eval.native_phase.v1'
                or type(receipt.get('pid')) is not int or receipt['pid'] != worker_pid
                or receipt.get('status') != 'running'
                or not isinstance(receipt.get('phase'), str) or not receipt['phase'].strip()
                or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            return None
        with (output / 'native-phase-traceback.log').open('rb') as stream:
            if not stream.read(64).startswith(b'Timeout ('):
                return None
        return f"{receipt['phase']} exceeded {timeout:g}s native phase deadline"
    except (OSError, ValueError, KeyError, TypeError):
        return None
