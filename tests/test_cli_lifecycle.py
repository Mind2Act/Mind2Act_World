import argparse
import json

from mind2act_eval.contracts import ConsumerKind, InstanceSpec, Result, Termination
from mind2act_eval.runner import write_json


def test_worker_marks_run_episode_returned_after_result_is_available(tmp_path, monkeypatch):
    from mind2act_eval import cli

    spec = InstanceSpec("demo", "one", "easy", "public", "v1", "none", "v1")
    spec_path = tmp_path / "spec.json"
    write_json(spec_path, spec)
    expected = Result("demo", "one", 25., Termination.COMPLETED, True,
                      ConsumerKind.DIAGNOSTIC, False, 0, {})
    monkeypatch.setattr(cli, "run_episode", lambda *args, **kwargs: expected)
    args = argparse.Namespace(
        spec=str(spec_path), output=str(tmp_path / "attempt"), asset_root=str(tmp_path),
        backend="demo", gpu=None, h5=None, replay_metadata=None,
        replay_metadata_sha256=None, h5_sha256=None,
        adapter="mind2act_eval.adapters.demo:create", adapter_config=None,
        physics_ticks=4, wall_s=10.,
    )
    (tmp_path / "attempt").mkdir()

    assert cli._worker(args) == 0

    marker = json.loads((tmp_path / "attempt/run-episode-returned.json").read_text())
    assert marker["stage"] == "run_episode_returned"
    assert marker["infra_valid"] is True
    assert marker["termination"] == "completed"
    assert marker["ticks"] == 0


def test_worker_and_cli_return_markers_distinguish_interpreter_shutdown(tmp_path, monkeypatch):
    from mind2act_eval import cli

    output = tmp_path / "attempt"
    output.mkdir()
    monkeypatch.setattr(cli, "_worker", lambda args: 1)

    code = cli._run_cli_entrypoint([
        "_worker", "--spec", str(tmp_path / "unused-spec.json"),
        "--output", str(output),
    ])

    assert code == 1
    worker = json.loads((output / "worker-returned.json").read_text())
    cli_returned = json.loads((output / "cli-returned.json").read_text())
    assert worker["stage"] == "worker_returned"
    assert worker["exit_code"] == 1
    assert cli_returned["stage"] == "cli_returned"
    assert cli_returned["exit_code"] == 1


def test_buffered_worker_stdout_is_flushed_before_seal_in_real_child(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from mind2act_eval.sealing import validate_worker_seal

    output = tmp_path / "attempt"
    output.mkdir()
    source_root = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source_root))
    script = r"""
import json, os, sys
from pathlib import Path
from mind2act_eval import cli
from mind2act_eval.contracts import ConsumerKind, InstanceSpec, Result, Termination, to_json
out = Path(sys.argv[1])
spec = InstanceSpec('demo', 'one', 'easy', 'public', 'v1', 'none', 'v1')
(out / 'spec.json').write_text(json.dumps(to_json(spec)))
(out / 'evidence.json').write_text('{}')
(out / 'score.json').write_text('{"value":25}')
(out / 'result.json').write_text(json.dumps(to_json(Result(
    'demo', 'one', 25., Termination.COMPLETED, True,
    ConsumerKind.DIAGNOSTIC, False, 1, {}))))
cli._WORKER_LIFECYCLE_OUTPUT = out
cli._write_worker_stage(out, 'run_episode_returned', infra_valid=True)
cli._write_worker_stage(out, 'worker_returned', exit_code=0)
cli.main = lambda argv=None: 0
print('buffered-before-seal')
cli._run_cli_entrypoint([])
"""
    log_path = output / "worker.log"
    with log_path.open("wb") as log:
        child = subprocess.run([sys.executable, "-c", script, str(output)],
                               stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())

    assert child.returncode == 0
    assert log_path.read_text() == "buffered-before-seal\n"
    result, seal = validate_worker_seal(output)
    assert result.infra_valid is True
    assert any(row["path"] == "worker.log" for row in seal["files"])
