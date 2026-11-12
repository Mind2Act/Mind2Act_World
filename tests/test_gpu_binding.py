from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import pytest

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json


def test_checked_physical_gpu_reaches_simulation_app_with_logical_cuda_zero(tmp_path, monkeypatch):
    from mind2act_eval import sim
    received = []
    lock_paths = []
    app_argv = []
    class FakeApp:
        def __init__(self, config):
            received.append(config)
            app_argv.extend(sys.argv)
        def close(self):
            pass
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    lock_root = tmp_path / ".mind2act-eval-locks"
    monkeypatch.setenv("MIND2ACT_EVAL_LOCK_ROOT", str(lock_root))
    monkeypatch.setattr(sys, "argv", ["mind2act-eval", "_worker"])
    monkeypatch.setattr(sim, "find_spec", lambda _: object())
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=FakeApp))
    monkeypatch.setattr(sim, "startup_gate", lambda path, **kwargs: (lock_paths.append(path), nullcontext(lambda: None))[1])
    output_dir = tmp_path / "attempt"
    simulator = sim.IsaacSimulator(ClockSpec(), output_dir, physical_gpu=4)
    simulator.close()
    assert received[0]["headless"] is True
    assert received[0]["active_gpu"] == 4
    assert received[0]["physics_gpu"] == 0
    assert received[0]["multi_gpu"] is False
    assert received[0]["fast_shutdown"] is False
    runtime = output_dir / "kit"
    assert received[0]["extra_args"] == [
        "--/app/settings/persistent=false",
        f"--/app/userConfigPath={runtime / 'user.config.json'}",
        f"--/app/cachePath={runtime / 'cache'}",
        f"--/app/dataPath={runtime / 'data'}",
        f"--/app/logPath={runtime / 'logs'}",
        f"--/log/file={runtime / 'kit.log'}",
    ]
    assert app_argv == ["mind2act-eval", "_worker", "--portable-root", str(runtime)]
    assert sys.argv == ["mind2act-eval", "_worker"]
    assert lock_paths == [lock_root / "kit.lock"]


def test_native_simulator_fails_closed_without_supervisor_lock_root(tmp_path, monkeypatch):
    from mind2act_eval import sim

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("MIND2ACT_EVAL_LOCK_ROOT", raising=False)
    monkeypatch.setattr(sim, "startup_gate", lambda *args, **kwargs: pytest.fail("must not start without a lock root"))
    with pytest.raises(ValueError, match="writable lock root"):
        sim.IsaacSimulator(ClockSpec(), tmp_path, physical_gpu=4)


@pytest.mark.parametrize('ignored_cache_setting', [False, True])
def test_supervised_kit_cache_shares_only_regenerable_content(tmp_path, monkeypatch, ignored_cache_setting):
    from mind2act_eval import sim
    from mind2act_eval.episode_storage import node_cache, cache_use_lock
    monkeypatch.setenv('M2A_CACHE_ROOT', str(tmp_path / 'cluster-cache'))
    cache = node_cache(tmp_path / 'attempt')
    configs, settings = [], {}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setenv('MIND2ACT_EVAL_LOCK_ROOT', str(tmp_path / '.locks'))
    monkeypatch.setenv('MIND2ACT_EVAL_NODE_CACHE', str(cache))
    monkeypatch.setattr(sim, 'find_spec', lambda _: object())
    monkeypatch.setattr(sim, 'startup_gate', lambda *args, **kwargs: nullcontext(lambda: None))
    def app(config):
        configs.append(config)
        settings.update(arg[2:].split('=', 1) for arg in config['extra_args'])
        if ignored_cache_setting:
            settings.pop('/rtx/shaderDb/shaderCachePath')
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, 'isaacsim', SimpleNamespace(SimulationApp=app))
    fake_settings = SimpleNamespace(get_settings=lambda: SimpleNamespace(set=settings.__setitem__, get=settings.get))
    monkeypatch.setitem(sys.modules, 'carb', SimpleNamespace(settings=fake_settings))
    monkeypatch.setitem(sys.modules, 'carb.settings', fake_settings)
    with cache_use_lock(cache) as fd:
        monkeypatch.setenv('MIND2ACT_EVAL_CACHE_LOCK_FD', str(fd))
        if ignored_cache_setting:
            with pytest.raises(RuntimeError, match='cache setting readback'):
                sim.IsaacSimulator(ClockSpec(), tmp_path / 'attempt', physical_gpu=4)
            return
        instance = sim.IsaacSimulator(ClockSpec(), tmp_path / 'attempt', physical_gpu=4)
        instance.close()
    args = configs[0]['extra_args']
    assert f'--/app/cachePath={tmp_path / "attempt/kit/cache"}' in args
    assert f'--/rtx/shaderDb/shaderCachePath={cache / "shaders"}' in args
    assert f'--/rtx/shaderDb/driverShaderCachePath={cache / "driver-shaders"}' in args
    assert f'--/app/extensions/registryCache={cache / "extensions"}' in args
    assert f'--/rtx-transient/resourcemanager/localTextureCachePath={cache / "textures"}' in args
    assert f'--/app/dataPath={tmp_path / "attempt/kit/data"}' in args
    assert f'--/app/userConfigPath={tmp_path / "attempt/kit/user.config.json"}' in args
    assert settings['/log/fileLogLevel'] == 'Error'


def test_native_simulator_reports_missing_extension_dependencies(tmp_path, monkeypatch):
    from mind2act_eval import sim

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("MIND2ACT_EVAL_LOCK_ROOT", str(tmp_path / ".locks"))
    monkeypatch.setattr(sim, "find_spec", lambda module: None if module == "toml" else object())
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(
        SimulationApp=lambda config: pytest.fail("Kit must not start with missing extension dependencies")))
    monkeypatch.setattr(sim, "startup_gate", lambda *args, **kwargs: nullcontext(lambda: None))
    with pytest.raises(RuntimeError, match="dependency 'toml' is missing"):
        sim.IsaacSimulator(ClockSpec(), tmp_path / "attempt", physical_gpu=4)


def test_kit_start_gate_enforces_spacing_across_clean_processes(tmp_path):
    import subprocess
    import time

    gate = tmp_path / "kit.lock"
    script = (
        "from pathlib import Path; import sys; "
        "from mind2act_eval.sim import startup_gate; "
        "with_gate = startup_gate(Path(sys.argv[1]), minimum_interval_s=.3); "
        "with_gate.__enter__(); with_gate.__exit__(None,None,None)"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    first_started = time.monotonic()
    subprocess.run([sys.executable, "-c", script, str(gate)], env=env, check=True)
    second_started = time.monotonic()
    subprocess.run([sys.executable, "-c", script, str(gate)], env=env, check=True)
    assert second_started - first_started >= .25


def test_kit_start_gate_crash_requires_fresh_cooldown(tmp_path):
    import subprocess
    import time

    gate = tmp_path / "kit.lock"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    crashing = (
        "from pathlib import Path; import os,sys; "
        "from mind2act_eval.sim import startup_gate; "
        "gate=startup_gate(Path(sys.argv[1]), minimum_interval_s=.25); "
        "gate.__enter__(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", crashing, str(gate)], env=env, check=True)
    recover = (
        "from pathlib import Path; import sys; "
        "from mind2act_eval.sim import startup_gate; "
        "gate=startup_gate(Path(sys.argv[1]), minimum_interval_s=.25); "
        "gate.__enter__(); gate.__exit__(None,None,None)"
    )
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", recover, str(gate)], env=env, check=True)
    assert time.monotonic() - started >= .48


def test_kit_start_gate_rejects_corrupt_state_without_starting(tmp_path):
    from mind2act_eval.sim import startup_gate

    gate = tmp_path / "kit.lock"
    gate.write_text('{"schema":"wrong","state":"idle"}\n')
    with pytest.raises(RuntimeError, match="state is invalid"):
        with startup_gate(gate, minimum_interval_s=.01):
            pytest.fail("must not enter with corrupt state")


def test_simulator_close_converts_zero_system_exit_to_cleanup_error(tmp_path, monkeypatch):
    from mind2act_eval import sim

    class App:
        def __init__(self, config):
            pass
        def close(self):
            raise SystemExit(0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("MIND2ACT_EVAL_LOCK_ROOT", str(tmp_path / ".locks"))
    monkeypatch.setattr(sim, "find_spec", lambda _: object())
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=App))
    monkeypatch.setattr(sim, "startup_gate", lambda *args, **kwargs: nullcontext(lambda: None))
    simulator = sim.IsaacSimulator(ClockSpec(), tmp_path / "attempt", physical_gpu=4)

    with pytest.raises(RuntimeError, match=r"SystemExit from App.close \(code=0\)"):
        simulator.close()

    assert simulator.app is None


def test_worker_converts_pre_result_system_exit_zero_to_failure(tmp_path, monkeypatch):
    from mind2act_eval import cli, sim

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "ARX.usd").write_text("asset")
    spec = InstanceSpec("demo", "one", "easy", "public", "v1", "none", "v1",
                        assets={"x5": "ARX.usd"})
    class ExitingSimulator:
        def __init__(self, *args, **kwargs):
            raise SystemExit(0)
    monkeypatch.setattr(sim, "IsaacSimulator", ExitingSimulator)
    args = SimpleNamespace(spec="unused", output=tmp_path / "attempt", asset_root=asset_root,
                           backend="isaac", gpu=0, h5=None, replay_metadata=None,
                           replay_metadata_sha256=None, adapter="unused:create",
                           adapter_config=None, physics_ticks=4, wall_s=10)
    monkeypatch.setattr(cli, "read_spec", lambda _: spec)

    with pytest.raises(RuntimeError, match=r"SystemExit\(0\) before completing its result"):
        cli._worker(args)


def test_cli_job_forwards_assigned_physical_gpu_to_worker(tmp_path, monkeypatch):
    from mind2act_eval import cli, batch
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(to_json(InstanceSpec("demo", "one", "easy", "public", "v1", "none", "v1"))))
    received = []
    def capture(jobs, output_root, **options):
        received.extend(jobs)
        return tmp_path, []
    monkeypatch.setattr(batch, "run_batch", capture)
    assert cli.main(["run", "--spec", str(source), "--gpu", "4", "--adapter", "model:create"]) == 0
    argv = received[0].argv
    assert argv[argv.index("--gpu") + 1] == "4"


def test_worker_passes_physical_gpu_to_simulator(tmp_path, monkeypatch):
    from mind2act_eval import cli, sim, robot
    from mind2act_eval.tasks import catalog
    source = tmp_path / "spec.json"
    (tmp_path / "ARX.usd").write_text("synthetic asset")
    spec = InstanceSpec("demo", "one", "easy", "public", "v1", "none", "v1", assets={"x5": "ARX.usd"})
    source.write_text(json.dumps(to_json(spec)))
    received = []
    class FakeSimulator:
        def __init__(self, *args, **options):
            received.append(options)
        def rgb(self):
            return {}
        def close(self):
            pass
    monkeypatch.setattr(sim, "IsaacSimulator", FakeSimulator)
    monkeypatch.setattr(robot, "X5Robot", lambda **options: None)
    monkeypatch.setattr(catalog, "load_task", lambda task_id: None)
    monkeypatch.setattr(cli, "run_episode", lambda *args, **options: SimpleNamespace(infra_valid=True))
    assert cli.main(["_worker", "--spec", str(source), "--asset-root", str(tmp_path),
                     "--output", str(tmp_path), "--gpu", "4"]) == 0
    assert received[0]["physical_gpu"] == 4
