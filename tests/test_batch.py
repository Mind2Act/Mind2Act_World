from dataclasses import replace
from pathlib import Path
import json
import os
import sys
import pytest
from mind2act_eval.contracts import ConsumerKind, InstanceSpec, to_json
from mind2act_eval.batch import EpisodeJob, run_attempt


def demo_spec():
    return InstanceSpec("demo", "demo-001", "easy", "synthetic-public-v1", "v1", "none", "v1")


def test_child_uses_frozen_requested_spec_even_if_source_changes(tmp_path, monkeypatch):
    # pytest pythonpath affects the parent only; bind the child to this checkout.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    spec = demo_spec()
    source = tmp_path / "source.json"
    source.write_text(json.dumps(to_json(spec)))
    argv = (sys.executable, "-m", "mind2act_eval.cli", "_worker", "--spec", str(source),
            "--asset-root", str(tmp_path), "--backend", "demo", "--wall-s", "10")
    job = EpisodeJob(spec, argv, ConsumerKind.DIAGNOSTIC, 10, 0)
    source.write_text(json.dumps(to_json(replace(spec, difficulty="hard", config={"recipe": "wrong"}))))
    from mind2act_eval import batch
    captured = {}
    original_popen = batch.subprocess.Popen
    def capture_popen(*args, **kwargs):
        if "env" in kwargs:
            captured.update(kwargs["env"])
        return original_popen(*args, **kwargs)
    monkeypatch.setattr(batch.subprocess, "Popen", capture_popen)
    lock_root = tmp_path / "runs" / ".mind2act-eval-locks"
    result = run_attempt(job, tmp_path / "attempt", lock_root=lock_root)
    assert result.infra_valid, result.error
    assert json.loads((tmp_path / "attempt/worker-seal.json").read_text())["teardown"] == "os_exit"
    assert json.loads((tmp_path / "attempt/batch-exit.json").read_text())["worker_seal_valid"] is True
    assert captured["MIND2ACT_EVAL_LOCK_ROOT"] == str(lock_root.resolve())
    resolved = InstanceSpec.from_dict(json.loads((tmp_path / "attempt/spec.json").read_text()))
    spec.assert_binding(resolved)


def test_supervisor_times_out_real_child_without_publishing_zero(tmp_path):
    from mind2act_eval.contracts import Termination
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "import time; time.sleep(30)"),
                     ConsumerKind.DIAGNOSTIC, .1, 0)
    result = run_attempt(job, tmp_path / "attempt")
    assert result.termination == Termination.WALL_TIMEOUT
    assert result.infra_valid is False
    assert result.score is None


@pytest.mark.parametrize('environment_phase', [True, False])
def test_registered_window_gets_separate_supervisor_allowance(tmp_path, environment_phase):
    spec = replace(demo_spec(), task_id='PR4' if environment_phase else 'demo')
    receipt = dict(schema='mind2act_eval.observation_closeout.v1', task_id=spec.task_id,
        instance_id=spec.instance_id, requested_ticks=121, actor_budget_consumed=False)
    # This deliberately incomplete CPU child tests supervisor timing only.
    script = ('import sys,json,time; from pathlib import Path; '
              'out=Path(sys.argv[sys.argv.index("--output")+1]); '
              f'(out/"observation-closeout.json").write_text({json.dumps(json.dumps(receipt))}) '
              f'if {environment_phase!r} else None; '
              'time.sleep(.5); (out/"window-finished").touch()')
    job = EpisodeJob(spec, (sys.executable, '-c', script), ConsumerKind.MODEL, .25, 0)
    result = run_attempt(job, tmp_path/'attempt', poll_s=.01, stop_grace_s=.05,
                         lock_root=tmp_path/'locks')
    assert (tmp_path/'attempt/window-finished').exists() is environment_phase
    assert not result.infra_valid and result.score is None  # No fake native result.


def test_zero_exit_without_worker_result_is_archived_as_infra_failure(tmp_path):
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "pass"),
                     ConsumerKind.DIAGNOSTIC, 10, 0)
    result = run_attempt(job, tmp_path / "attempt")

    saved = json.loads((tmp_path / "attempt/result.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert "result.json" in result.error
    assert saved["infra_valid"] is False
    assert saved["score"] is None


def test_valid_result_without_seal_stays_invalid_after_preseal_exit_zero(tmp_path, monkeypatch):
    from mind2act_eval.contracts import ConsumerKind, Result, Termination, to_json

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    script = tmp_path / "preseal_exit.py"
    script.write_text(
        "import argparse,json,os; from pathlib import Path; "
        "from mind2act_eval.contracts import Result,Termination,ConsumerKind,to_json; "
        "p=argparse.ArgumentParser(); p.add_argument('--spec'); p.add_argument('--output'); "
        "a=p.parse_args(); out=Path(a.output); "
        "(out/'evidence.json').write_text('{}'); (out/'score.json').write_text('{\"value\":25}'); "
        "r=Result('demo','demo-001',25.,Termination.PHYSICS_BUDGET,True,ConsumerKind.DIAGNOSTIC,False,1); "
        "(out/'result.json').write_text(json.dumps(to_json(r))); os._exit(0)"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, str(script), "--spec", "placeholder"),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    result = run_attempt(job, tmp_path / "attempt")

    saved = json.loads((tmp_path / "attempt/result.json").read_text())
    exit_record = json.loads((tmp_path / "attempt/batch-exit.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert saved["infra_valid"] is False
    assert saved["score"] is None
    assert exit_record["worker_seal_valid"] is False
    assert exit_record["worker_teardown"] is None


def test_sealed_result_is_valid_even_when_process_exit_code_is_nonzero(tmp_path, monkeypatch):
    from mind2act_eval.contracts import ConsumerKind, Result, Termination, to_json

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    script = tmp_path / "postseal_exit.py"
    script.write_text(
        "import argparse,json,os; from pathlib import Path; "
        "from mind2act_eval.contracts import ConsumerKind,Result,Termination,to_json; "
        "from mind2act_eval.sealing import seal_worker_outputs; "
        "p=argparse.ArgumentParser(); p.add_argument('--spec'); p.add_argument('--output'); "
        "a=p.parse_args(); out=Path(a.output); "
        "(out/'spec.json').write_text(Path(a.spec).read_text()); "
        "(out/'evidence.json').write_text('{}'); (out/'score.json').write_text('{\"value\":25}'); "
        "r=Result('demo','demo-001',25.,Termination.PHYSICS_BUDGET,True,ConsumerKind.DIAGNOSTIC,False,1); "
        "(out/'result.json').write_text(json.dumps(to_json(r))); "
        "pid=os.getpid(); "
        "rows=[('run-episode-returned.json','run_episode_returned',{'infra_valid':True}),"
        "('worker-returned.json','worker_returned',{'exit_code':0}),"
        "('cli-returned.json','cli_returned',{'exit_code':0})]; "
        "[(out/name).write_text(json.dumps({'schema':'mind2act_eval.worker_lifecycle.v1','stage':stage,'pid':pid,**details})) for name,stage,details in rows]; "
        "seal_worker_outputs(out); os._exit(23)"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, str(script), "--spec", "placeholder"),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    result = run_attempt(job, tmp_path / "attempt")

    exit_record = json.loads((tmp_path / "attempt/batch-exit.json").read_text())
    saved = json.loads((tmp_path / "attempt/result.json").read_text())
    assert result.infra_valid is True
    assert result.score == 25.
    assert saved["infra_valid"] is True
    assert exit_record["exit_code"] == 23
    assert exit_record["worker_teardown"] == "os_exit"
    assert exit_record["worker_seal_valid"] is True


def test_postseal_child_mutation_is_rejected_by_final_validation(tmp_path, monkeypatch):
    from mind2act_eval import batch
    from mind2act_eval.contracts import ConsumerKind

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    script = tmp_path / "mutate_after_seal.py"
    release = tmp_path / "allow_mutation"
    script.write_text(f"""
import argparse, json, os, time
from pathlib import Path
from mind2act_eval.contracts import ConsumerKind, Result, Termination, to_json
from mind2act_eval.sealing import seal_worker_outputs
p = argparse.ArgumentParser(); p.add_argument('--spec'); p.add_argument('--output')
a = p.parse_args(); out = Path(a.output)
(out / 'spec.json').write_text(Path(a.spec).read_text())
(out / 'evidence.json').write_text('{{}}'); (out / 'score.json').write_text('{{\"value\":25}}')
r = Result('demo', 'demo-001', 25., Termination.COMPLETED, True, ConsumerKind.DIAGNOSTIC, False, 1)
(out / 'result.json').write_text(json.dumps(to_json(r))); pid = os.getpid()
rows = [('run-episode-returned.json', 'run_episode_returned', {{'infra_valid': True}}),
        ('worker-returned.json', 'worker_returned', {{'exit_code': 0}}),
        ('cli-returned.json', 'cli_returned', {{'exit_code': 0}})]
for name, stage, details in rows:
    (out / name).write_text(json.dumps({{'schema': 'mind2act_eval.worker_lifecycle.v1',
        'stage': stage, 'pid': pid, **details}}))
seal_worker_outputs(out)
release = Path({str(release)!r})
while not release.exists():
    time.sleep(.005)
(out / 'evidence.json').write_text('{{\"late\":true}}')
os._exit(0)
""")
    original_validate = batch.validate_worker_seal
    validation_calls = []

    def validate_then_release(directory, **kwargs):
        result = original_validate(directory, **kwargs)
        validation_calls.append(Path(directory))
        release.touch()
        return result

    monkeypatch.setattr(batch, "validate_worker_seal", validate_then_release)
    job = EpisodeJob(demo_spec(), (sys.executable, str(script), "--spec", "placeholder"),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    result = run_attempt(job, tmp_path / "attempt", poll_s=.005)

    exit_record = json.loads((tmp_path / "attempt/batch-exit.json").read_text())
    assert validation_calls
    assert result.infra_valid is False
    assert result.score is None
    assert exit_record["exit_code"] == 0
    assert exit_record["worker_seal_valid"] is False
    assert "missing or changed: evidence.json" in exit_record["worker_seal_error"]


def test_visible_seal_with_failed_supervisor_fsync_is_invalid(tmp_path, monkeypatch):
    import stat
    import mind2act_eval.sealing as sealing
    from mind2act_eval.contracts import ConsumerKind

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    script = tmp_path / "delay_final_directory_fsync.py"
    entered = tmp_path / "final-dir-fsync-entered"
    script.write_text(f"""
import argparse, json, os, stat, sys, time
from pathlib import Path
from mind2act_eval.contracts import ConsumerKind, Result, Termination, to_json
from mind2act_eval.sealing import seal_worker_outputs
p = argparse.ArgumentParser(); p.add_argument('--spec'); p.add_argument('--output')
a = p.parse_args(); out = Path(a.output)
(out / 'spec.json').write_text(Path(a.spec).read_text())
(out / 'evidence.json').write_text('{{}}')
(out / 'score.json').write_text('{{\"value\":25}}')
(out / 'video.fixture').write_bytes(b'cpu-only fixture')
r = Result('demo', 'demo-001', 25., Termination.COMPLETED, True,
           ConsumerKind.DIAGNOSTIC, False, 1, artifacts={{'video': 'video.fixture'}})
(out / 'result.json').write_text(json.dumps(to_json(r)))
pid = os.getpid()
rows = [('run-episode-returned.json', 'run_episode_returned', {{'infra_valid': True}}),
        ('worker-returned.json', 'worker_returned', {{'exit_code': 0}}),
        ('cli-returned.json', 'cli_returned', {{'exit_code': 0}})]
for name, stage, details in rows:
    (out / name).write_text(json.dumps({{'schema': 'mind2act_eval.worker_lifecycle.v1',
        'stage': stage, 'pid': pid, **details}}))
real_fsync = os.fsync
def delayed_final_dir_fsync(fd):
    current = os.fstat(fd)
    if stat.S_ISDIR(current.st_mode) and current.st_ino == out.stat().st_ino and (out / 'worker-seal.json').exists():
        Path({str(entered)!r}).write_text('entered')
        time.sleep(60)
    return real_fsync(fd)
os.fsync = delayed_final_dir_fsync
seal_worker_outputs(out)
os._exit(0)
""")

    output = tmp_path / "attempt"
    real_parent_fsync = sealing.os.fsync

    def fail_parent_root_fsync(fd):
        current = os.fstat(fd)
        if (output / "worker-seal.json").exists():
            root = output.stat()
            if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (root.st_dev, root.st_ino):
                raise OSError("injected final directory fsync failure")
        return real_parent_fsync(fd)

    monkeypatch.setattr(sealing.os, "fsync", fail_parent_root_fsync)
    job = EpisodeJob(demo_spec(), (sys.executable, str(script), "--spec", "placeholder"),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    result = run_attempt(job, output, poll_s=.005)

    assert entered.exists()
    exit_record = json.loads((output / "batch-exit.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert exit_record["worker_seal_valid"] is False
    assert "injected final directory fsync failure" in exit_record["worker_seal_error"]


def test_zero_exit_after_preclose_checkpoint_archives_invalid_worker_result(tmp_path):
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "import json,os; from pathlib import Path; Path('robot-receipt.json').write_text(json.dumps({'status':'verified'})); Path('preclose-trace.json').write_text(json.dumps({'stage':'awaiting_cleanup','tick':3,'result':{'task_id':'demo','instance_id':'demo-001','score':25,'infra_valid':True}})); os._exit(0)"),
                     ConsumerKind.DIAGNOSTIC, 10, 0)
    result = run_attempt(job, tmp_path / "attempt")

    saved = json.loads((tmp_path / "attempt/result.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert "before writing final result" in result.error
    assert "tick=3" in result.error
    assert result.artifacts["preclose_trace"] == "preclose-trace.json"
    assert result.artifacts["native_robot_receipt"] == "robot-receipt.json"


def test_attempt_bounds_owned_worker_hanging_after_preclose_trace(tmp_path):
    from mind2act_eval.contracts import Termination
    script = (
        "import json,time; from pathlib import Path; "
        "Path('robot-receipt.json').write_text(json.dumps({'status':'verified'})); "
        "Path('preclose-trace.json').write_text(json.dumps({'stage':'awaiting_cleanup','tick':0,'result':{"
        "'error':'ValueError: configured camera prim is missing: left_wrist'}})); "
        "Path('cleanup-started.json').write_text('{}'); time.sleep(30)"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", script),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    result = run_attempt(job, tmp_path / "attempt", poll_s=.01,
                         cleanup_timeout_s=.15, stop_grace_s=.05)

    saved = json.loads((tmp_path / "attempt/result.json").read_text())
    exit_record = json.loads((tmp_path / "attempt/batch-exit.json").read_text())
    assert result.infra_valid is False
    assert result.termination == Termination.ERROR
    assert "worker cleanup exceeded 0.15s" in result.error
    assert "configured camera prim is missing: left_wrist" in result.error
    assert result.artifacts["preclose_trace"] == "preclose-trace.json"
    assert result.artifacts["cleanup_started"] == "cleanup-started.json"
    assert result.artifacts["native_robot_receipt"] == "robot-receipt.json"
    assert saved["score"] is None
    assert exit_record["exit_code"] < 0
    assert exit_record["cleanup_timeout_s"] == .15
    assert saved["infra_valid"] is False


def test_supervisor_signals_registered_trace_during_python_exit_cleanup(tmp_path):
    from mind2act_eval import batch
    from mind2act_eval.contracts import Termination
    script = (
        "import atexit,faulthandler,json,os,signal,time; from pathlib import Path; "
        "stream=open('cleanup-traceback.log','a',1); "
        "faulthandler.enable(file=stream,all_threads=True); "
        "faulthandler.register(signal.SIGUSR1,file=stream,all_threads=True,chain=False); "
        "Path('cleanup-trace-armed.json').write_text(json.dumps({'pid':os.getpid(),'signal':'SIGUSR1'})); "
        "Path('cleanup-started.json').write_text('{}'); "
        "atexit.register(lambda: time.sleep(10))"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", script),
                     ConsumerKind.DIAGNOSTIC, 10, 0)
    monkeypatch = pytest.MonkeyPatch()
    _mock_pidfd_signalling(monkeypatch, batch)
    monkeypatch.setattr(batch, "_linux_sigcgt_has", lambda pid, signum: True)

    try:
        result = run_attempt(job, tmp_path / "attempt", poll_s=.01,
                             cleanup_timeout_s=.4, trace_signal_after_s=.08,
                             stop_grace_s=.05)
    finally:
        monkeypatch.undo()

    trace = (tmp_path / "attempt/cleanup-traceback.log").read_text()
    signal_event = json.loads((tmp_path / "attempt/shutdown-diagnostic-signal.json").read_text())
    assert result.termination == Termination.ERROR
    assert signal_event["signal"] == "SIGUSR1"
    assert signal_event["status"] == "sent"
    assert "lambda" in trace
    assert result.artifacts["cleanup_traceback"] == "cleanup-traceback.log"
    assert result.artifacts["shutdown_diagnostic_signal"] == "shutdown-diagnostic-signal.json"


def test_trace_artifacts_are_linked_when_worker_exits_after_signal(tmp_path):
    from mind2act_eval import batch
    result_json = (
        "{\"task_id\":\"demo\",\"instance_id\":\"demo-001\",\"score\":25,"
        "\"termination\":\"completed\",\"infra_valid\":true,"
        "\"consumer_kind\":\"diagnostic\",\"model_tested\":false,\"ticks\":1,"
        "\"timings\":{},\"artifacts\":{},\"policy_metadata\":{},"
        "\"score_components\":{},\"task_metadata\":{}}"
    )
    script = (
        "import atexit,faulthandler,json,os,signal,time; from pathlib import Path; "
        "stream=open('cleanup-traceback.log','a',1); "
        "faulthandler.enable(file=stream,all_threads=True); "
        "faulthandler.register(signal.SIGUSR1,file=stream,all_threads=True,chain=False); "
        "Path('cleanup-trace-armed.json').write_text(json.dumps({'pid':os.getpid(),'signal':'SIGUSR1'})); "
        "Path('cleanup-started.json').write_text('{}'); "
        "Path('result.json').write_text(" + repr(result_json) + "); "
        "atexit.register(lambda: time.sleep(.15))"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", script),
                     ConsumerKind.DIAGNOSTIC, 10, 0)
    monkeypatch = pytest.MonkeyPatch()
    _mock_pidfd_signalling(monkeypatch, batch)
    monkeypatch.setattr(batch, "_linux_sigcgt_has", lambda pid, signum: True)

    try:
        result = run_attempt(job, tmp_path / "attempt", poll_s=.01,
                             cleanup_timeout_s=.8, trace_signal_after_s=.08)
    finally:
        monkeypatch.undo()

    assert result.infra_valid is False
    assert result.score is None
    assert result.artifacts["cleanup_traceback"] == "cleanup-traceback.log"
    assert result.artifacts["shutdown_diagnostic_signal"] == "shutdown-diagnostic-signal.json"
    signal_event = json.loads((tmp_path / "attempt/shutdown-diagnostic-signal.json").read_text())
    assert signal_event["status"] == "sent"


def test_failure_after_worker_result_still_links_shutdown_sidecars(tmp_path):
    result_json = (
        "{\"task_id\":\"demo\",\"instance_id\":\"wrong-instance\",\"score\":25,"
        "\"termination\":\"completed\",\"infra_valid\":true,"
        "\"consumer_kind\":\"diagnostic\",\"model_tested\":false,\"ticks\":1,"
        "\"timings\":{},\"artifacts\":{},\"policy_metadata\":{},"
        "\"score_components\":{},\"task_metadata\":{}}"
    )
    script = (
        "import json; from pathlib import Path; "
        "Path('cleanup-started.json').write_text('{}'); "
        "Path('cleanup-traceback.log').write_text('trace'); "
        "Path('shutdown-diagnostic-signal.json').write_text('{}'); "
        "Path('result.json').write_text(" + repr(result_json) + ")"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", script),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    result = run_attempt(job, tmp_path / "attempt")

    assert result.infra_valid is False
    assert result.artifacts["cleanup_traceback"] == "cleanup-traceback.log"
    assert result.artifacts["shutdown_diagnostic_signal"] == "shutdown-diagnostic-signal.json"


def _native_debug_config(tmp_path, *, limit_s=8., wrapper_hash_override=None):
    import hashlib
    from mind2act_eval.batch import NativeExitDebugConfig
    tmp_path.mkdir(parents=True, exist_ok=True)
    wrapper = tmp_path / "native_stack.py"
    shim = tmp_path / "exec_argv.py"
    debugger = tmp_path / "cuda-gdb"
    wrapper.write_text("# fixed wrapper fixture\n")
    shim.write_text("# fixed exec shim fixture\n")
    debugger.write_text("debugger fixture\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "schema": "mind2act_eval.native_exit_debug.v1",
        "wrapper_python": sys.executable,
        "wrapper": str(wrapper),
        "wrapper_sha256": wrapper_hash_override or digest(wrapper),
        "shim_sha256": digest(shim),
        "debugger": str(debugger),
        "debugger_sha256": digest(debugger),
        "after_s": .2,
        "limit_s": limit_s,
    }
    config_path = tmp_path / "native-exit-debug.json"
    config_path.write_text(json.dumps(config))
    parsed = None if wrapper_hash_override else NativeExitDebugConfig.read(config_path)
    return parsed, config_path


def test_native_exit_debug_config_pins_wrapper_shim_debugger_and_timing(tmp_path):
    from mind2act_eval.batch import NativeExitDebugConfig

    config, config_path = _native_debug_config(tmp_path)

    assert isinstance(config, NativeExitDebugConfig)
    assert config.config_sha256
    assert config.wrapper == (tmp_path / "native_stack.py").resolve()
    assert config.shim == (tmp_path / "exec_argv.py").resolve()
    assert config.after_s == .2
    assert config.limit_s == 8.


def test_native_exit_debug_config_rejects_changed_wrapper_before_launch(tmp_path):
    from mind2act_eval.batch import NativeExitDebugConfig
    import pytest

    _, config_path = _native_debug_config(tmp_path, wrapper_hash_override="0" * 64)

    with pytest.raises(ValueError, match="wrapper SHA256 mismatch"):
        NativeExitDebugConfig.read(config_path)


def test_run_attempt_rechecks_debugger_files_before_spawning(tmp_path, monkeypatch):
    from mind2act_eval import batch
    config, _ = _native_debug_config(tmp_path / "tools")
    config.wrapper.write_text("changed after config load\n")
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "pass"),
                     ConsumerKind.DIAGNOSTIC, 10, 0, None, config)
    monkeypatch.setattr(batch, "user_rss_bytes", lambda: 0)
    monkeypatch.setattr(batch.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("must reject changed wrapper before launch"))

    result = batch.run_attempt(job, tmp_path / "attempt", lock_root=tmp_path / "locks")

    assert result.infra_valid is False
    assert result.error == "ValueError: native exit debug wrapper SHA256 mismatch"


def test_run_attempt_wraps_frozen_worker_argv_and_links_native_stack(tmp_path, monkeypatch):
    from mind2act_eval import batch
    from mind2act_eval.contracts import Result, Termination
    debug_config, _ = _native_debug_config(tmp_path / "tools")
    output = tmp_path / "attempt"
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "pass", "literal $() spaces"),
                     ConsumerKind.DIAGNOSTIC, 10, 0, None, debug_config)
    captured = {}

    class Child:
        pid = 2_000_000_000
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        attempt = Path(kwargs["cwd"])
        from mind2act_eval.contracts import to_json
        from mind2act_eval.sealing import seal_worker_outputs
        batch.write_json(attempt / "spec.json", to_json(demo_spec()))
        batch.write_json(attempt / "evidence.json", {"tick": 1})
        batch.write_json(attempt / "score.json", {"value": 25.})
        batch.write_json(attempt / "result.json", Result(
            "demo", "demo-001", 25., Termination.COMPLETED, True,
            ConsumerKind.DIAGNOSTIC, False, 1, {}))
        for name, stage, details in (
            ("run-episode-returned.json", "run_episode_returned", {"infra_valid": True}),
            ("worker-returned.json", "worker_returned", {"exit_code": 0}),
            ("cli-returned.json", "cli_returned", {"exit_code": 0}),
        ):
            batch.write_json(attempt / name, {
                "schema": "mind2act_eval.worker_lifecycle.v1", "stage": stage,
                "pid": 2_000_000_000, **details,
            })
        debug_output = attempt / "native-exit-debug"
        debug_output.mkdir()
        (debug_output / "status.json").write_text(json.dumps({"stack_captured": True}))
        (debug_output / "native-stack.txt").write_text("native fixture stack\n")
        seal_worker_outputs(attempt)
        return Child()

    monkeypatch.setattr(batch, "user_rss_bytes", lambda: 0)
    monkeypatch.setattr(batch.subprocess, "Popen", popen)

    result = batch.run_attempt(job, output, lock_root=tmp_path / "locks")

    argv = captured["argv"]
    separator = argv.index("--")
    assert argv[0] == str(debug_config.wrapper_python)
    assert argv[1] == str(debug_config.wrapper)
    assert argv[separator + 1:separator + 1 + len(job.argv)] == list(job.argv)
    assert argv[separator + 1 + len(job.argv):] == ["--output", str(output.resolve())]
    assert "literal $() spaces" in argv
    assert result.infra_valid is True
    assert result.artifacts["native_exit_debug_status"] == "native-exit-debug/status.json"
    assert result.artifacts["native_exit_stack"] == "native-exit-debug/native-stack.txt"
    request = json.loads((output / "native-exit-debug-request.json").read_text())
    assert request["terminate_after_stack"] is True
    assert request["wrapper_sha256"] == debug_config.wrapper_sha256


def test_native_stack_diagnostic_termination_stays_invalid_and_preserves_worker_result(tmp_path, monkeypatch):
    from mind2act_eval import batch
    from mind2act_eval.contracts import Result, Termination
    debug_config, _ = _native_debug_config(tmp_path / "tools")
    output = tmp_path / "attempt"
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "pass"),
                     ConsumerKind.DIAGNOSTIC, 10, 0, None, debug_config)

    class Child:
        pid = 2_000_000_001
        returncode = -15

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def popen(argv, **kwargs):
        attempt = Path(kwargs["cwd"])
        batch.write_json(attempt / "result.json", Result(
            "demo", "demo-001", 100., Termination.COMPLETED, True,
            ConsumerKind.DIAGNOSTIC, False, 12, {}))
        debug_output = attempt / "native-exit-debug"
        debug_output.mkdir()
        (debug_output / "status.json").write_text(json.dumps({
            "stack_captured": True, "diagnostic_termination": True,
        }))
        (debug_output / "native-stack.txt").write_text("captured stack\n")
        return Child()

    monkeypatch.setattr(batch, "user_rss_bytes", lambda: 0)
    monkeypatch.setattr(batch.subprocess, "Popen", popen)

    result = batch.run_attempt(job, output, lock_root=tmp_path / "locks")

    snapshot = json.loads((output / "worker-result-before-supervisor.json").read_text())
    final = json.loads((output / "result.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert result.termination == Termination.KILLED
    assert snapshot["infra_valid"] is True and snapshot["score"] == 100.
    assert final["infra_valid"] is False and final["score"] is None
    assert result.artifacts["native_exit_stack"] == "native-exit-debug/native-stack.txt"
    assert result.artifacts["worker_result_before_supervisor"] == "worker-result-before-supervisor.json"


def test_native_exit_debug_rejects_multi_job_batch(tmp_path):
    import pytest
    from mind2act_eval.batch import run_batch
    config, _ = _native_debug_config(tmp_path)
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "pass"),
                     ConsumerKind.DIAGNOSTIC, 10, 0, None, config)

    with pytest.raises(ValueError, match="single job and max_parallel=1"):
        run_batch([job, job], tmp_path / "out", require_shared=False,
                  max_parallel=2, start_interval_s=0)


def _mock_pidfd_signalling(monkeypatch, batch):
    delivered = []
    pidfds = {}

    def open_pidfd(pid):
        fd = os.open(os.devnull, os.O_RDONLY)
        pidfds[fd] = pid
        return fd

    def send_signal(fd, signum):
        delivered.append((pidfds[fd], signum))
        os.kill(pidfds[fd], signum)

    monkeypatch.setattr(batch.os, "pidfd_open", open_pidfd, raising=False)
    monkeypatch.setattr(batch.signal, "pidfd_send_signal", send_signal, raising=False)
    return delivered


@pytest.mark.parametrize(
    ("armed", "handler_registered", "expected_status"),
    [
        ({"pid": 4321, "signal": "SIGUSR1"}, True, "skipped"),
        ({"pid": 1234, "signal": "SIGTERM"}, True, "skipped"),
        ({"pid": 1234, "signal": "SIGUSR1"}, False, "skipped"),
    ],
    ids=("foreign-pid", "arbitrary-signal", "handler-missing"),
)
def test_shutdown_trace_gate_never_sends_for_unverified_worker(
        tmp_path, monkeypatch, armed, handler_registered, expected_status):
    from types import SimpleNamespace
    from mind2act_eval import batch

    output = tmp_path / "attempt"
    output.mkdir()
    (output / "cleanup-trace-armed.json").write_text(json.dumps({**armed, "pid": armed["pid"]}))
    child = SimpleNamespace(pid=1234, poll=lambda: None)
    sent = _mock_pidfd_signalling(monkeypatch, batch)
    monkeypatch.setattr(batch, "_linux_sigcgt_has",
                        lambda pid, signum: handler_registered)

    batch._request_shutdown_trace(child, output)

    event = json.loads((output / "shutdown-diagnostic-signal.json").read_text())
    assert event["status"] == expected_status
    assert sent == []
    if not handler_registered and armed["signal"] == "SIGUSR1":
        assert event["reason"] == "handler_missing"


def test_batch_archives_trace_and_releases_after_bounded_cleanup(tmp_path):
    import tarfile
    from mind2act_eval.batch import run_batch
    script = (
        "import json,time; from pathlib import Path; "
        "Path('robot-receipt.json').write_text(json.dumps({'status':'verified'})); "
        "Path('preclose-trace.json').write_text(json.dumps({'stage':'awaiting_cleanup','tick':0,'result':{"
        "'error':'ValueError: configured camera prim is missing: left_wrist'}})); "
        "Path('cleanup-started.json').write_text('{}'); time.sleep(30)"
    )
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", script),
                     ConsumerKind.DIAGNOSTIC, 10, 0)

    campaign, summary = run_batch([job], tmp_path / "outputs", require_shared=False,
                                  start_interval_s=0, cleanup_timeout_s=.15)

    assert summary[0]["result"].infra_valid is False
    assert summary[0]["result"].score is None
    with tarfile.open(campaign / summary[0]["archive"]) as stream:
        names = stream.getnames()
    assert any(name.endswith("result.json") for name in names)
    assert any(name.endswith("preclose-trace.json") for name in names)
    assert any(name.endswith("cleanup-started.json") for name in names)
    assert any(name.endswith("robot-receipt.json") for name in names)


def test_archive_skips_kit_data_tree_even_when_descendant_is_unreadable(tmp_path):
    import tarfile
    from mind2act_eval.batch import _archive_attempt
    attempt = tmp_path / "episode-0000-attempt-01"
    screenshots = attempt / "kit/data/documents/Kit/shared/screenshots"
    screenshots.mkdir(parents=True)
    (attempt / "worker.log").write_text("worker evidence")
    (screenshots / "protected.bin").write_text("kit private data")
    screenshots.chmod(0)
    try:
        archive = _archive_attempt(attempt, tmp_path / "attempt.tar.gz")
        with tarfile.open(archive) as stream:
            names = stream.getnames()
        assert any(name.endswith("worker.log") for name in names)
        assert not any("kit/data/" in name for name in names)
    finally:
        screenshots.chmod(0o700)


def test_rss_budget_blocks_launch_without_touching_other_processes(tmp_path):
    from mind2act_eval.contracts import Termination
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "raise RuntimeError('must not start')"),
                     ConsumerKind.DIAGNOSTIC, 1, 0)
    result = run_attempt(job, tmp_path / "attempt", rss_limit=1)
    assert result.termination == Termination.RESOURCE_LIMIT
    assert not (tmp_path / "attempt/worker.log").exists()


def test_batch_archives_all_attempts_and_creates_unique_campaigns(tmp_path):
    from mind2act_eval.batch import run_batch
    import tarfile
    job = EpisodeJob(demo_spec(), (sys.executable, "-c", "import time; time.sleep(10)"),
                     ConsumerKind.DIAGNOSTIC, .05, 1)
    campaign, summary = run_batch([job], tmp_path, require_shared=False, start_interval_s=0)
    assert len(summary) == 2
    assert summary[0]["directory"] != summary[1]["directory"]
    for attempt in summary:
        with tarfile.open(campaign / attempt["archive"]) as stream:
            assert any(name.endswith("result.json") for name in stream.getnames())
    assert json.loads((campaign / "summary.json").read_text())[1]["result"]["score"] is None


def test_production_root_accepts_resolved_shared_mount_and_stays_contained(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import mind2act_eval.batch as batch

    shared_mount = tmp_path / "resolved-shared"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == Path("/shared"):
            return shared_mount
        if path.is_relative_to(Path("/shared")):
            return shared_mount / path.relative_to(Path("/shared"))
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    lock_root = Path("/shared/evaluation-runs") / f".mind2act-eval-{os.getuid()}"
    monkeypatch.setenv("MIND2ACT_EVAL_LOCK_ROOT", str(lock_root))
    lock_paths = []
    def capture_gate(path):
        lock_paths.append(Path(path))
        return nullcontext()
    monkeypatch.setattr(batch, "startup_gate", capture_gate)

    campaign, summary = batch.run_batch(
        [], Path("/shared/evaluation-runs/results"),
        require_shared=True, max_parallel=1, start_interval_s=60,
    )
    batch.run_batch(
        [], Path("/shared/evaluation-runs/probes/PR1/easy"),
        require_shared=True, max_parallel=1, start_interval_s=60,
    )

    assert campaign.is_relative_to(shared_mount)
    assert campaign.parent == shared_mount / "users/yuzhi/workspace_jclin/mind2act-eval-runs/results"
    assert campaign.parent.is_dir()
    physical_lock_root = shared_mount / "users/yuzhi/workspace_jclin/mind2act-eval-runs" / f".mind2act-eval-{os.getuid()}"
    assert lock_paths == [physical_lock_root / "batch.lock", physical_lock_root / "batch.lock"]
    assert physical_lock_root.is_relative_to(shared_mount / "users/yuzhi/workspace_jclin/mind2act-eval-runs")
    sibling_root = campaign.parent.parent / "other-results"
    assert batch._launcher_lock_root(sibling_root, require_shared=True) == physical_lock_root
    assert summary == []


def test_production_batch_requires_stable_lock_root(tmp_path, monkeypatch):
    import mind2act_eval.batch as batch

    monkeypatch.delenv("MIND2ACT_EVAL_LOCK_ROOT", raising=False)
    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs: tmp_path / "shared"
                        if path == Path("/shared") else path)
    output_root = tmp_path / "shared" / "runs" / "results"
    with pytest.raises(ValueError, match="stable MIND2ACT_EVAL_LOCK_ROOT"):
        batch.run_batch([], output_root, require_shared=True)
    assert not output_root.exists()














def test_production_root_still_rejects_paths_outside_shared(tmp_path, monkeypatch):
    import mind2act_eval.batch as batch

    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs: tmp_path / "resolved-shared"
                        if path == Path("/shared") else path)
    with pytest.raises(ValueError, match="under /shared"):
        batch.run_batch([], tmp_path / "private-output", require_shared=True)
