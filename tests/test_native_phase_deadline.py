"""A fake native reset holds the GIL; the worker must exit with a durable timeout."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pytest

from mind2act_eval.batch import EpisodeJob, run_attempt
from mind2act_eval.contracts import ConsumerKind, Termination
from test_batch import demo_spec


def test_native_reset_deadline_survives_gil_hold_and_is_not_a_model_zero(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[1]/'src'))
    script = tmp_path/'blocked_reset.py'
    script.write_text('''import ctypes
from pathlib import Path
from mind2act_eval.runtime_diagnostics import run_native_phase
class World:
    def reset(self):
        libc=ctypes.PyDLL(None)
        libc.usleep.argtypes=[ctypes.c_uint]
        libc.usleep(10_000_000)
run_native_phase(World().reset, Path.cwd(), phase='CP02_RESET', timeout_s=.15)
''')
    job = EpisodeJob(replace(demo_spec(), task_id='CP02'), (sys.executable, str(script)),
                     ConsumerKind.DIAGNOSTIC, 5., 0)
    result = run_attempt(job, tmp_path/'attempt', poll_s=.01, stop_grace_s=.05,
                         lock_root=tmp_path/'locks')
    assert result.termination == Termination.WALL_TIMEOUT
    assert not result.infra_valid and result.score is None and not result.model_tested
    assert 'CP02_RESET exceeded 0.15s' in result.error
    assert result.artifacts['native_phase'] == 'native-phase.json'
    trace = (tmp_path/'attempt/native-phase-traceback.log').read_text()
    assert 'Timeout' in trace and 'reset' in trace
    receipt = json.loads((tmp_path/'attempt/native-phase.json').read_text())
    assert receipt['status'] == 'running' and receipt['phase'] == 'CP02_RESET'
    assert 0 < time.monotonic()-receipt['started_monotonic'] < 2.


@pytest.mark.parametrize('raises', [False, True])
def test_finished_phase_cancels_watchdog_before_future_work(tmp_path, monkeypatch, raises):
    import subprocess
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[1]/'src'))
    script = '''import json,time
from pathlib import Path
from mind2act_eval.runtime_diagnostics import run_native_phase
def action():
    if RAISES: raise RuntimeError('native reset failed')
    return 42
try:
    assert run_native_phase(action, Path.cwd(), phase='CP02_RESET', timeout_s=.15) == 42
except RuntimeError as error:
    assert str(error) == 'native reset failed'
time.sleep(.3)
Path('survived').touch()
'''.replace('RAISES', repr(raises))
    child = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, timeout=3,
                           capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert (tmp_path/'survived').exists()
    receipt = json.loads((tmp_path/'native-phase.json').read_text())
    assert receipt['status'] == ('failed' if raises else 'completed')
    assert (tmp_path/'native-phase-traceback.log').read_text() == ''


@pytest.mark.parametrize('timeout', [True, 0, -1, float('nan'), float('inf')])
def test_invalid_phase_budget_does_not_invoke_native_code(tmp_path, timeout):
    from mind2act_eval.runtime_diagnostics import run_native_phase
    with pytest.raises(ValueError, match='positive finite timeout'):
        run_native_phase(lambda: pytest.fail('must reject before reset'), tmp_path,
                         phase='CP02_RESET', timeout_s=timeout)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('change', ['foreign_pid', 'completed', 'failed', 'no_timeout_trace', 'corrupt', 'non_object'])
def test_supervisor_does_not_label_other_worker_failures_as_native_timeout(tmp_path, change):
    import os
    from mind2act_eval.runtime_diagnostics import native_phase_timeout
    receipt = dict(schema='mind2act_eval.native_phase.v1', pid=os.getpid(), phase='CP02_RESET',
                   timeout_s=300., status='running')
    if change == 'foreign_pid': receipt['pid'] += 1
    if change in ('completed', 'failed'): receipt['status'] = change
    (tmp_path/'native-phase.json').write_text('{' if change == 'corrupt' else json.dumps([] if change == 'non_object' else receipt))
    (tmp_path/'native-phase-traceback.log').write_text('Fatal native error' if change == 'no_timeout_trace' else 'Timeout (0:05:00)!')
    assert native_phase_timeout(tmp_path, os.getpid()) is None
