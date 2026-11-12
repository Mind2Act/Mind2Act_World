"""CPU-only real-child counterexample for an isolated native worker's build failure."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CPU_WORKER = r'''
import atexit, builtins, json, sys
from pathlib import Path
from unittest.mock import patch
from mind2act_eval import cli, robot, runtime_release, sim
from mind2act_eval.adapters.demo import DemoAdapter
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.runner import write_json
from mind2act_eval.sim.asset_resolution import AssetUnresolved
from mind2act_eval.sim.demo import DemoRobot
from mind2act_eval.tasks import TaskDefinition, catalog

out, mode = Path(sys.argv[1]), sys.argv[2]
def event(name):
    with (out / 'events.log').open('a') as f:
        f.write(name + '\n')
atexit.register(lambda: event('python.atexit'))
original_import = builtins.__import__
def cpu_import(name, *args, **kwargs):
    if name.split('.')[0] in {'isaacsim', 'omni', 'carb', 'torch'}:
        raise AssertionError('CPU counterexample cannot import native SDK/model: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_import

class Recorder:
    artifacts = {}
    def close(self):
        event('recorder.close')
        if mode == 'cleanup':
            raise RuntimeError('encoder flush failed')
class App:
    def close(self):
        event('SDK.app.close')
        raise RuntimeError('unsafe framework release reached')
class World:
    def stop(self): event('world.stop')
class Camera:
    def destroy(self): event('camera.destroy')
class Simulator(sim.IsaacSimulator):
    def __init__(self, clock, output, **kwargs):
        self._output_dir = output
        self._camera_receipt = None
        self._world = World()
        self._contacts = None
        self._cameras = {'head': Camera()}
        self.app = App()
        self.recorder = Recorder()
        self.timings = {}
    def bind_task_instance(self, spec, output): pass
    def arm_cleanup_diagnostics(self): return {}
    def _release_gate(self): event('release_gate')
    def ready(self): fail()
    def prepare_for_os_exit(self):
        if mode == 'unprepared':
            # A stale disk receipt must never stand in for this call completing.
            write_json(out / 'shutdown-receipt.json', {'status': 'os_exit_prepared'})
            return
        super().prepare_for_os_exit()
class Scene:
    def close(self):
        event('scene.restore')
        if mode == 'scene_cleanup': raise RuntimeError('scene restore failed')
def fail():
    report = {'missing_paths': ['./Aluminum_Anodized/Aluminum_Anodized_BaseColor.png']}
    write_json(out / 'asset-resolution-report.json', report)
    raise AssetUnresolved(report)
def build(spec, ctx):
    if mode == 'build': fail()
    return Scene()
def projection(assets, output):
    if mode == 'projection': raise OSError('projection receipt write failed')
    event('source.receipt')
def adapter_close(self): event('adapter.close')
def forbidden(*args, **kwargs): raise AssertionError('failed build must not infer or score')
spec = InstanceSpec('E1', 'cpu-failure', 'easy', 'public', 'v1', 'none', 'v1',
                    assets={'x5': 'fake.usda'})
write_json(out / 'input-spec.json', spec)
(out / 'fake.usda').write_text('#usda 1.0\n')
task = TaskDefinition('E1', build, forbidden)
from mind2act_eval.sim import source_assets
if mode == 'flush':
    from mind2act_eval import sealing
    def fail_fsync(path): raise OSError('disk flush failed')
    sealing._fsync_file = fail_fsync
with patch.object(runtime_release, 'validate_release_inputs', lambda *a, **k: None), \
     patch.object(sim, 'IsaacSimulator', Simulator), \
     patch.object(robot, 'X5Robot', lambda **kwargs: DemoRobot()), \
     patch.object(catalog, 'load_task', lambda task_id: task), \
     patch.object(source_assets, 'write_source_projection_receipt', projection), \
     patch.object(DemoAdapter, 'close', adapter_close), \
     patch.object(DemoAdapter, 'act', forbidden):
    print('buffered-failure-stdout')
    print('buffered-failure-stderr', file=sys.stderr)
    code = cli._run_cli_entrypoint(['_worker', '--backend', 'isaac', '--gpu', '0',
        '--spec', str(out / 'input-spec.json'), '--asset-root', str(out), '--output', str(out)])
    event('entrypoint.returned:' + str(code))
    raise SystemExit(code)
'''


def run_cpu_worker(tmp_path, mode):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / '.'))
    child = subprocess.run([sys.executable, '-c', CPU_WORKER, str(tmp_path), mode],
                           capture_output=True, text=True, env=env, timeout=20)
    events = (tmp_path / 'events.log').read_text().splitlines()
    return child, events


@pytest.mark.parametrize('mode', ['build', 'scene'])
def test_invalid_native_worker_flushes_then_exits_nonzero_without_sdk_or_success_seal(tmp_path, mode):
    child, events = run_cpu_worker(tmp_path, mode)
    assert child.returncode == 1, child.stderr
    assert 'SDK.app.close' not in events
    assert 'python.atexit' not in events
    assert not any(e.startswith('entrypoint.returned') for e in events)
    assert events == (['adapter.close'] if mode == 'build' else ['adapter.close', 'scene.restore']) + [
        'recorder.close', 'world.stop', 'camera.destroy', 'release_gate', 'source.receipt']
    assert 'buffered-failure-stdout' in child.stdout
    assert 'buffered-failure-stderr' in child.stderr
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['infra_valid'] is False and result['score'] is None
    assert result['model_tested'] is False and result['ticks'] == 0
    assert result['error'].startswith('AssetUnresolved: asset_unresolved: ./Aluminum_Anodized/')
    assert result['task_metadata']['invalid_reason'] == 'asset_unresolved'
    assert result['timings']['total_s'] > 0
    assert result['timings']['episode_s'] == 0
    assert not (tmp_path / 'ready.json').exists()
    assert not (tmp_path / 'worker-seal.json').exists()
    assert json.loads((tmp_path / 'cli-returned.json').read_text())['exit_code'] == 1
    from mind2act_eval.sealing import seal_worker_outputs
    with pytest.raises(ValueError, match='infrastructure-valid'):
        seal_worker_outputs(tmp_path)


@pytest.mark.parametrize('mode', ['cleanup', 'scene_cleanup', 'projection', 'unprepared', 'flush'])
def test_incomplete_cleanup_or_projection_never_claims_controlled_failure_exit(tmp_path, mode):
    child, events = run_cpu_worker(tmp_path, mode)
    assert child.returncode > 0
    assert 'python.atexit' in events
    assert any(e.startswith('entrypoint.returned:') for e in events)
    assert not (tmp_path / 'worker-seal.json').exists()
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['infra_valid'] is False and result['score'] is None
    assert 'AssetUnresolved' in result['error']
    if mode == 'cleanup':
        assert 'encoder flush failed' in result['error']
    if mode == 'scene_cleanup':
        assert 'scene restore failed' in result['error']
    if mode == 'flush':
        assert 'disk flush failed' in child.stderr


def test_new_cli_invocation_does_not_inherit_failure_exit_permission(tmp_path, monkeypatch):
    from mind2act_eval import cli
    monkeypatch.setattr(cli, '_WORKER_FAILURE_EXIT_READY', True)
    monkeypatch.setattr(cli, '_worker', lambda args: 1)
    monkeypatch.setattr(os, '_exit', lambda code: pytest.fail('stale permission must not exit this process'))
    assert cli._run_cli_entrypoint(['_worker', '--output', str(tmp_path), '--backend', 'demo']) == 1
    assert cli._WORKER_FAILURE_EXIT_READY is False
