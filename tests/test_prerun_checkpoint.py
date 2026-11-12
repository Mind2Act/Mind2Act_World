"""CPU-only exception paths, real durable files and a virtual parent process boundary."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mind2act_eval.contracts import ClockSpec, ConsumerKind, InstanceSpec, Termination
from mind2act_eval.runner import write_json


def spec():
    return InstanceSpec('CP02', 'cpu-prerun', 'easy', 'cpu', 'v1', 'none', 'v1',
                        assets={'x5': 'fake.usda'})


def close_probe(output, synced, expected_phase):
    """A failing/blocking close can read the first error only after both fsyncs."""
    trace_path, marker_path = output / 'preclose-trace.json', output / 'cleanup-started.json'
    trace = json.loads(trace_path.read_text())
    marker = json.loads(marker_path.read_text())
    assert trace['phase'] == expected_phase
    assert trace['result']['infra_valid'] is False and trace['result']['score'] is None
    assert trace['result']['model_tested'] is False and trace['tick'] == 0
    assert trace['result']['task_id'] == 'CP02'
    assert trace['result']['instance_id'] == 'cpu-prerun'
    assert 'first setup error' in trace['exception']['traceback']
    assert marker['pid'] == trace['pid'] == os.getpid()
    source = Path(trace['source_file'])
    assert trace['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    for path in (trace_path, marker_path):
        st = path.stat()
        assert (st.st_dev, st.st_ino) in synced
    directory = output.stat()
    assert synced.count((directory.st_dev, directory.st_ino)) >= 2
    assert not (output / 'result.json').exists()
    assert not (output / 'worker-seal.json').exists()
    return trace_path.read_bytes()


@pytest.fixture
def synced(monkeypatch):
    calls, real = [], os.fsync
    def fsync(fd):
        st = os.fstat(fd)
        calls.append((st.st_dev, st.st_ino))
        return real(fd)
    monkeypatch.setattr(os, 'fsync', fsync)
    return calls


def constructor_seam(tmp_path, monkeypatch, close):
    from mind2act_eval import sim, episode_storage
    output = tmp_path / 'attempt'
    output.mkdir()
    write_json(output / 'spec.json', spec())  # Same original input the parent persists.
    original = ValueError('first setup error: constructor')
    @contextmanager
    def gate(*args, **kwargs):
        def mark():
            raise original
        yield mark
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('MIND2ACT_EVAL_LOCK_ROOT', str(tmp_path / 'locks'))
    monkeypatch.setattr(sim, '_require_isaac_extension_dependencies', lambda: None)
    monkeypatch.setattr(sim, 'startup_compatibility', lambda paths: nullcontext())
    monkeypatch.setattr(sim, 'startup_gate', gate)
    monkeypatch.setattr(episode_storage, 'worker_cache', lambda: None)
    monkeypatch.setitem(sys.modules, 'isaacsim', SimpleNamespace(
        SimulationApp=lambda config: SimpleNamespace(close=lambda: close(output))))
    simulator_type = sim.IsaacSimulator
    return lambda: simulator_type(ClockSpec(), output, physical_gpu=0), original, output


@pytest.mark.parametrize('secondary', [False, True])
def test_constructor_saves_original_before_partial_app_close_and_preserves_it(tmp_path, monkeypatch, synced, secondary, capsys):
    observed = []
    def close(output):
        observed.append(close_probe(output, synced, 'constructor'))
        if secondary:
            raise RuntimeError('secondary close failure')
    invoke, original, output = constructor_seam(tmp_path, monkeypatch, close)

    with pytest.raises(ValueError) as raised:
        invoke()

    assert raised.value is original
    assert observed == [(output / 'preclose-trace.json').read_bytes()]
    if secondary:
        assert 'secondary close failure' in capsys.readouterr().err


def worker_seam(output, monkeypatch, close):
    from mind2act_eval import cli, robot, runtime_release, sim
    from mind2act_eval.tasks import catalog
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fake.usda').write_text('#usda 1.0\n')
    write_json(output / 'spec.json', spec())
    original = ValueError('first setup error: adapter factory')
    def factory(config):
        raise original
    simulator = SimpleNamespace(rgb=lambda: {}, close=lambda: close(output))
    monkeypatch.setattr(runtime_release, 'validate_release_inputs', lambda *a, **k: None)
    monkeypatch.setattr(sim, 'IsaacSimulator', lambda *a, **k: simulator)
    monkeypatch.setattr(robot, 'X5Robot', lambda **k: object())
    monkeypatch.setattr(catalog, 'load_task', lambda task_id: object())
    monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(create=factory))
    monkeypatch.setattr(cli, 'run_episode', lambda *a, **k: pytest.fail('pre-run failure cannot execute physics/model'))
    args = argparse.Namespace(spec=str(output / 'spec.json'), output=str(output), asset_root=str(output),
        backend='isaac', gpu=0, h5=None, replay_metadata=None, replay_metadata_sha256=None,
        h5_sha256=None, adapter='cpu_factory:create', adapter_config=None, physics_ticks=40320, wall_s=43200.)
    return lambda: cli._worker(args), original


@pytest.mark.parametrize('secondary', [False, True])
def test_worker_setup_saves_first_error_before_close_without_minting_result(tmp_path, monkeypatch, synced, secondary):
    from mind2act_eval import cli
    observed = []
    def close(output):
        observed.append(close_probe(output, synced, 'worker_setup'))
        if secondary:
            raise RuntimeError('secondary close failure')
    invoke, original = worker_seam(tmp_path, monkeypatch, close)

    with pytest.raises(ValueError) as raised:
        invoke()

    assert raised.value is original and len(observed) == 1
    assert cli._WORKER_FAILURE_EXIT_READY is False
    assert not (tmp_path / 'run-episode-returned.json').exists()


def test_failed_checkpoint_is_not_reported_as_saved_or_allowed_to_mask_original(tmp_path, monkeypatch, capsys):
    def close(output):
        assert not (output / 'cleanup-started.json').exists()
        raise RuntimeError('secondary close failure')
    invoke, original = worker_seam(tmp_path, monkeypatch, close)
    def fail(fd):
        raise OSError('checkpoint disk unavailable')
    monkeypatch.setattr(os, 'fsync', fail)

    with pytest.raises(ValueError) as raised:
        invoke()

    assert raised.value is original
    assert 'checkpoint disk unavailable' in capsys.readouterr().err
    assert not (tmp_path / 'result.json').exists()
    assert not (tmp_path / 'worker-seal.json').exists()


def test_constructor_checkpoint_stays_first_when_cli_also_handles_failure(tmp_path, monkeypatch, synced):
    from mind2act_eval import cli, sim
    def close(output):
        close_probe(output, synced, 'constructor')
        raise RuntimeError('secondary close failure')
    constructor, original, output = constructor_seam(tmp_path, monkeypatch, close)
    # The wrapper still uses the real constructor body and its failure handling.
    real_constructor = constructor
    invoke, _ = worker_seam(output, monkeypatch, close)
    monkeypatch.setattr(sim, 'IsaacSimulator', lambda *a, **k: real_constructor())

    with pytest.raises(ValueError) as raised:
        invoke()

    assert raised.value is original
    trace = json.loads((output / 'preclose-trace.json').read_text())
    assert trace['phase'] == 'constructor'
    assert trace['exception']['message'] == 'first setup error: constructor'


@pytest.mark.parametrize('receipt_failure', [False, True])
def test_runner_exception_keeps_its_existing_progress_checkpoint(tmp_path, monkeypatch, receipt_failure):
    from mind2act_eval import cli
    from mind2act_eval.sim import source_assets
    checkpoint = tmp_path / 'preclose-trace.json'
    observed = []
    def close(output):
        observed.append(checkpoint.read_bytes())
    invoke, _ = worker_seam(tmp_path, monkeypatch, close)
    monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(create=lambda config: None))
    original = RuntimeError('inside runner')
    secondary = OSError('post-run source receipt failure')
    if receipt_failure:
        def fail_receipt(*args):
            raise secondary
        monkeypatch.setattr(source_assets, 'write_source_projection_receipt', fail_receipt)
    saved = b'{"tick":17,"result":{"error":"inside runner","model_tested":true}}\n'
    def run(*args, **kwargs):
        checkpoint.write_bytes(saved)
        raise original
    monkeypatch.setattr(cli, 'run_episode', run)

    with pytest.raises((RuntimeError, OSError)) as raised:
        invoke()

    assert raised.value is (secondary if receipt_failure else original) and observed == [saved]
    assert checkpoint.read_bytes() == saved
    assert not (tmp_path / 'cleanup-started.json').exists()


def test_limit_validation_before_runner_is_still_a_setup_failure(tmp_path, monkeypatch):
    from mind2act_eval import cli
    observed = []
    def close(output):
        trace = json.loads((output / 'preclose-trace.json').read_text())
        assert trace['phase'] == 'worker_setup'
        assert 'limits rejected' in trace['exception']['message']
        assert (output / 'cleanup-started.json').exists()
        observed.append(True)
    invoke, _ = worker_seam(tmp_path, monkeypatch, close)
    monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(create=lambda config: None))
    original = ValueError('limits rejected before runner')
    def reject(*args):
        raise original
    monkeypatch.setattr(cli, 'EpisodeLimits', reject)

    with pytest.raises(ValueError) as raised:
        invoke()

    assert raised.value is original and observed == [True]
    assert not (tmp_path / 'result.json').exists()


@pytest.mark.parametrize('argument', ['BuildContext', 'EpisodeLimits'])
def test_first_setup_error_survives_secondary_projection_failure_before_close(tmp_path, monkeypatch, synced, capsys, argument):
    from mind2act_eval import cli
    from mind2act_eval.sim import source_assets
    events, snapshots = [], []
    def snapshot(label):
        path = tmp_path / 'preclose-trace.json'
        snapshots.append((label, path.read_bytes() if path.exists() else None))
    def close(output):
        events.append('sim.close')
        snapshot('close')
    invoke, _ = worker_seam(tmp_path, monkeypatch, close)
    adapter = SimpleNamespace(close=lambda: events.append('adapter.close'))
    monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(create=lambda config: adapter))
    original = ValueError('first setup error: ' + argument)
    secondary = OSError('secondary source projection receipt failure')
    def reject(*args):
        events.append('setup.error')
        raise original
    def fail_receipt(*args):
        events.append('projection.error')
        snapshot('projection')
        raise secondary
    monkeypatch.setattr(cli, argument, reject)
    monkeypatch.setattr(source_assets, 'write_source_projection_receipt', fail_receipt)

    with pytest.raises((ValueError, OSError)) as raised:
        invoke()

    assert raised.value is original
    assert events == ['setup.error', 'projection.error', 'adapter.close', 'sim.close']
    assert [label for label, _ in snapshots] == ['projection', 'close']
    for _, raw in snapshots:
        trace = json.loads(raw)
        assert trace['exception']['type'] == 'ValueError'
        assert trace['exception']['message'] == str(original)
        assert 'raise original' in trace['exception']['traceback']
        assert 'raise secondary' not in trace['exception']['traceback']
        assert trace['result']['error'] == 'ValueError: ' + str(original)
    assert snapshots[0][1] == snapshots[1][1] == (tmp_path / 'preclose-trace.json').read_bytes()
    for name in ('preclose-trace.json', 'cleanup-started.json'):
        st = (tmp_path / name).stat()
        assert (st.st_dev, st.st_ino) in synced
    assert 'secondary source projection receipt failure' in capsys.readouterr().err
    assert cli._WORKER_FAILURE_EXIT_READY is False
    assert not any((tmp_path / name).exists() for name in ('result.json', 'score.json', 'worker-seal.json', 'run-episode-returned.json'))


def test_marker_write_failure_keeps_first_trace_but_never_claims_persistence(tmp_path, monkeypatch, capsys):
    from mind2act_eval import episode_storage
    real = episode_storage.durable_json
    def persist(path, value):
        if Path(path).name == 'cleanup-started.json':
            raise OSError('marker persistence failed')
        return real(path, value)
    monkeypatch.setattr(episode_storage, 'durable_json', persist)
    def close(output):
        assert 'first setup error' in (output / 'preclose-trace.json').read_text()
        assert not (output / 'cleanup-started.json').exists()
    invoke, original = worker_seam(tmp_path, monkeypatch, close)

    with pytest.raises(ValueError) as raised:
        invoke()

    assert raised.value is original
    assert 'marker persistence failed' in capsys.readouterr().err
    assert not (tmp_path / 'worker-seal.json').exists()


@pytest.mark.parametrize('two_faults', [False, True])
def test_real_parent_watchdog_consumes_setup_checkpoint_at_unchanged_120_seconds(tmp_path, monkeypatch, synced, two_faults):
    from mind2act_eval import batch, episode_storage
    output = tmp_path / 'attempt'
    class Child:
        pid = 2_000_000_000
        returncode = None
        def poll(self): return self.returncode
    class Clock:
        now = 0.
        def monotonic(self): return self.now
        def sleep(self, seconds): self.now += seconds
    clock, child, stops, traces = Clock(), Child(), [], []
    class BlockedClose(BaseException): pass
    def close(output):
        close_probe(output, synced, 'worker_setup')
        raise BlockedClose('fake blocked native teardown')
    def popen(argv, **kwargs):
        invoke, original = worker_seam(Path(kwargs['cwd']), monkeypatch, close)
        if two_faults:
            from mind2act_eval import cli
            from mind2act_eval.sim import source_assets
            original = ValueError('first setup error: limits construction')
            def reject(*args): raise original
            def fail_receipt(*args): raise OSError('secondary source receipt error')
            monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(create=lambda config: None))
            monkeypatch.setattr(cli, 'EpisodeLimits', reject)
            monkeypatch.setattr(source_assets, 'write_source_projection_receipt', fail_receipt)
        with pytest.raises(ValueError) as raised:
            invoke()
        assert raised.value is original
        return child  # Virtual process remains in blocked teardown; no OS signals.
    def stop(proc, **kwargs):
        stops.append(clock.now)
        proc.returncode = 1
    monkeypatch.setattr(batch.subprocess, 'Popen', popen)
    monkeypatch.setattr(batch, 'time', clock)
    monkeypatch.setattr(batch, 'user_rss_bytes', lambda: 0)
    monkeypatch.setattr(batch, '_stop_owned_group', stop)
    monkeypatch.setattr(batch, '_request_shutdown_trace', lambda *args: traces.append(clock.now))
    monkeypatch.setattr(episode_storage, 'process_group_exited', lambda pid: True)
    job = batch.EpisodeJob(spec(), ('cpu-only',), ConsumerKind.MODEL, 43200., 0)

    result = batch.run_attempt(job, output, lock_root=tmp_path / 'locks', poll_s=30.)

    assert result.termination == Termination.ERROR and result.infra_valid is False
    assert result.score is None and result.model_tested is False and result.ticks == 0
    assert 'worker cleanup exceeded 120s' in result.error
    assert ('first setup error: limits construction' if two_faults else 'first setup error: adapter factory') in result.error
    assert stops == [150.] and traces == [90.]
    assert job.wall_s == 43200. and job.retries == 0
    assert not (output / 'worker-seal.json').exists()
    assert json.loads((output / 'batch-exit.json').read_text())['worker_exit_verified'] is True
