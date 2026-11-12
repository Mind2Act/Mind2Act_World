import json

import pytest


def test_monotonic_phases_count_nested_calls_and_exception_without_reordering(monkeypatch):
    from mind2act_eval.timing import PhaseTimings
    import mind2act_eval.timing as timing
    now = [0.]
    monkeypatch.setattr(timing.time, 'monotonic', lambda: now[0])
    phases = PhaseTimings()
    events = []
    with phases.measure('outer'):
        events.append('before')
        now[0] = 1.
        with phases.measure('inner'):
            events.append('operation')
            now[0] = 3.
        now[0] = 4.
        events.append('after')
    with pytest.raises(ValueError, match='original'):
        with phases.measure('inner'):
            now[0] = 5.
            raise ValueError('original')
    assert events == ['before', 'operation', 'after']
    assert phases.snapshot() == {
        'outer': {'calls': 1, 'wall_s': 4., 'max_s': 4.},
        'inner': {'calls': 2, 'wall_s': 3., 'max_s': 2.},
    }


def test_runner_records_action_phases_with_original_tick_order(tmp_path):
    from test_runner import fixture
    from mind2act_eval.runner import EpisodeLimits, run_episode
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.contracts import ConsumerKind
    spec, task, ctx, events = fixture(tmp_path)
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid, result.error
    assert result.ticks == 4 and result.score == 25.
    assert events == [(name, tick) for tick in range(1, 5)
                      for name in ('before', 'step', 'after', 'evidence')]
    receipt = json.loads((tmp_path / 'timing-breakdown.json').read_text())
    for name in ('scene.before_step', 'robot.apply', 'sim.step', 'scene.after_step', 'scene.evidence'):
        assert receipt['runner'][name]['calls'] == 4
    assert receipt['runner']['evidence.validate']['calls'] == 5  # Includes final scoring validation.
    assert receipt['clock'] == 'monotonic'
    assert receipt['nested_not_additive'] is True


def _run_and_mark_demo_worker(tmp_path):
    from test_runner import fixture
    from mind2act_eval.runner import EpisodeLimits, run_episode, write_json
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.contracts import ConsumerKind
    spec, task, ctx, _ = fixture(tmp_path)
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid and result.score == 25. and result.ticks == 4
    for name, stage, details in (
        ('run-episode-returned.json', 'run_episode_returned', {'infra_valid': True}),
        ('worker-returned.json', 'worker_returned', {'exit_code': 0}),
        ('cli-returned.json', 'cli_returned', {'exit_code': 0}),
    ):
        write_json(tmp_path / name, dict(schema='mind2act_eval.worker_lifecycle.v1',
                                       stage=stage, pid=12345, **details))
    return result


@pytest.mark.parametrize('failure', ['replace', 'partial_write'])
def test_optional_timing_write_failure_still_allows_real_worker_seal(tmp_path, monkeypatch, failure):
    from pathlib import Path
    from mind2act_eval.sealing import seal_worker_outputs, validate_worker_seal
    original_open, original_replace = Path.open, Path.replace
    injected = []

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, value):
            self.stream.write(value[:8])
            self.stream.flush()
            injected.append('partial_write')
            raise OSError('injected timing partial write')

    def failing_open(path, mode='r', *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if failure == 'partial_write' and path.name == 'timing-breakdown.json.tmp' and mode in ('w', 'x'):
            return PartialWriter(stream)
        return stream

    def failing_replace(path, target):
        if failure == 'replace' and path.name == 'timing-breakdown.json.tmp':
            injected.append('replace')
            raise OSError('injected timing replace')
        return original_replace(path, target)

    monkeypatch.setattr(Path, 'open', failing_open)
    monkeypatch.setattr(Path, 'replace', failing_replace)
    result = _run_and_mark_demo_worker(tmp_path)
    assert injected == [failure]
    assert 'timing_breakdown' not in result.artifacts
    seal = seal_worker_outputs(tmp_path)
    sealed_result, checked = validate_worker_seal(tmp_path)
    assert checked == seal and sealed_result == result
    assert not (tmp_path / 'timing-breakdown.json.tmp').exists()


@pytest.mark.parametrize('name', ['timing-breakdown.json.tmp', 'required-output.json.tmp'])
def test_optional_timing_preserves_preexisting_temporary_and_strict_sealing(tmp_path, name):
    from mind2act_eval.sealing import seal_worker_outputs
    temporary = tmp_path / name
    temporary.write_bytes(b'preexisting unfinished output')
    result = _run_and_mark_demo_worker(tmp_path)
    assert temporary.read_bytes() == b'preexisting unfinished output'
    if name == 'timing-breakdown.json.tmp':
        assert 'timing_breakdown' not in result.artifacts
    with pytest.raises(ValueError, match='unfinished temporary'):
        seal_worker_outputs(tmp_path)


def test_render_rgba_and_pipe_phases_keep_order_bytes_and_zero_delta(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace as NS
    import numpy as np
    from mind2act_eval.sim import IsaacSimulator
    from mind2act_eval.sim.video import RGBRecorder
    from mind2act_eval.timing import phases_for
    import mind2act_eval.timing as timing
    now, events, payloads = [0.], [], []
    monkeypatch.setattr(timing.time, 'monotonic', lambda: now[0])
    def operation(name, seconds):
        events.append(name)
        now[0] += seconds
    world = NS(current_time=0., current_time_step_index=0, is_playing=lambda: True)
    def step(*, render):
        assert render is False
        operation('physics', 2.)
        world.current_time, world.current_time_step_index = 1/120, 1
    world.step, world.render = step, lambda: operation('render', 3.)
    monkeypatch.setattr(IsaacSimulator, 'world', property(lambda self: world))
    monkeypatch.setattr(IsaacSimulator, '_render_settings_snapshot', lambda self: {})
    def replicate(**kwargs):
        assert kwargs == dict(rt_subframes=2, delta_time=0., pause_timeline=False, wait_for_render=True)
        operation('rep', 5.)
    for name in ('omni', 'omni.replicator', 'omni.replicator.core'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['omni.replicator.core'].orchestrator = NS(step=replicate)
    pixels = np.full((2, 2, 4), 17, dtype=np.uint8)
    def rgba():
        operation('rgba', 1.)
        return pixels
    def write(value):
        payloads.append(value)
        operation('pipe', 7.)
    recorder = RGBRecorder(tmp_path, 30)
    recorder._writers['head'] = (NS(stdin=NS(write=write)), None, (2, 2))
    sim = object.__new__(IsaacSimulator)
    sim.timings = dict(physics_s=0., render_s=0., video_s=0.)
    sim._ready, sim._initial_rgb = True, None
    sim._cameras = {'head': NS(get_rgba=rgba)}
    sim._camera_config = {'head': {'resolution': [2, 2]}}
    sim.recorder = recorder
    sim.step(render=True)
    sim.capture()
    assert events == ['physics', 'render', 'rep', 'rgba', 'pipe']
    assert payloads == [pixels[..., :3].tobytes()]
    assert world.current_time_step_index == 1 and world.current_time == 1/120
    assert sim.timings == dict(physics_s=2., render_s=8., video_s=8.)
    detail = phases_for(sim).snapshot()
    assert detail['world.render'] == dict(calls=1, wall_s=3., max_s=3.)
    assert detail['rep.step'] == dict(calls=1, wall_s=5., max_s=5.)
    for name in ('capture.audit', 'get_rgba.head', 'rgba.validate.head', 'rgb.copy.head'):
        assert detail[name]['calls'] == 1
    assert phases_for(recorder).snapshot()['pipe.write.head'] == dict(calls=1, wall_s=7., max_s=7.)
