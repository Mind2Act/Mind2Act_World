"""CPU seams: real runner/Isaac capture/adapter code, synthetic physics and pixels."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mind2act_eval.contracts import BuildContext, ConsumerKind, Health, InstanceSpec, NativeEvidence, Observation, Score
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.sim import IsaacSimulator
from mind2act_eval.sim.demo import DemoRobot
from mind2act_eval.sim.video import video_interval
from mind2act_eval.tasks import TaskDefinition

CAMERAS = ('head', 'left_wrist', 'right_wrist')


def episode(tmp_path, monkeypatch, video_hz=6, evidence_rgb_ticks=()):
    world = NS(current_time=0., current_time_step_index=0)
    renders, captures, samples, evidence_ticks, observed_rgb = [], [], [], [], []
    def step(*, render):
        assert render is False
        world.current_time_step_index += 1
        world.current_time = world.current_time_step_index / 120
    world.step = step
    world.render = lambda: renders.append(world.current_time_step_index)
    monkeypatch.setattr(IsaacSimulator, 'world', property(lambda self: world))
    sim = object.__new__(IsaacSimulator)
    sim._output_dir = tmp_path
    sim.bind_task_instance = lambda *args: None
    sim.arm_cleanup_diagnostics = None
    sim.prepare_for_os_exit = None
    sim._camera_receipt = None
    sim.video_hz = video_hz
    sim.timings = dict(physics_s=0., render_s=0., video_s=0.)
    sim._ready, sim._initial_rgb = True, None
    sim._record_runtime = lambda *args: None
    sim._synchronize_capture = lambda: captures.append(world.current_time_step_index)
    def sample():
        tick = world.current_time_step_index
        samples.append(tick)
        return ({name: np.full((2, 2, 3), tick % 256, dtype=np.uint8) for name in CAMERAS},
                {name: dict(valid=True) for name in CAMERAS})
    sim._sample_cameras = sample
    frame_ticks = []
    def record(rgb):
        assert set(rgb) == set(CAMERAS)
        assert all(int(frame[0, 0, 0]) == world.current_time_step_index % 256 for frame in rgb.values())
        frame_ticks.append(world.current_time_step_index)
        for name in rgb:
            sim.recorder.frame_counts[name] = sim.recorder.frame_counts.get(name, 0) + 1
    sim.recorder = NS(capture=record, frame_counts={}, artifacts={})
    def ready():
        # Native ready() already captures and records the render-only tick-zero image.
        sim._initial_rgb = sample()[0]
        sim.recorder.capture(sim._initial_rgb)
    sim.ready, sim.close = ready, lambda: None
    robot = DemoRobot()
    original_observe = robot.observe
    def observe():
        return {**original_observe(), 'rgb': sim.rgb()}
    robot.observe = observe
    spec = InstanceSpec('demo', 'one', 'easy', 'public', 'v1', 'none', 'v1')
    class Scene:
        def health(self): return Health(True)
        def is_complete(self): return False
        def observe(self, tick):
            obs = Observation(tick, tick / 120, instruction='move', **robot.observe())
            observed_rgb.append((tick, bool(obs.rgb)))
            return obs
        def evidence(self, tick):
            evidence_ticks.append(tick)
            if tick in evidence_rgb_ticks:
                assert int(sim.rgb()['head'][0, 0, 0]) == tick % 256
            return NativeEvidence('demo', 'one', tick, dict(tick=tick))
        def requires_rgb(self, tick): return tick in evidence_rgb_ticks
        def close(self): pass
    task = TaskDefinition('demo', lambda spec, ctx: Scene(), lambda e: Score(25.))
    ctx = BuildContext(sim, robot, None, tmp_path)
    return spec, task, ctx, NS(renders=renders, captures=captures, samples=samples,
                              frame_ticks=frame_ticks, evidence_ticks=evidence_ticks, observed_rgb=observed_rgb)


class ChunkTransport:
    capabilities = {}
    def __init__(self, chunk): self.chunk = chunk
    def reset(self, seed):
        self.needs_observation, self.row, self.image_ticks = True, 0, []
    def action(self, payload):
        if self.needs_observation:
            assert payload['frequency'] == 30
            images = payload['images']
            assert len(images) == 3
            self.image_ticks.append(int(images['cam_high'][0, 0, 0]))
        else:
            assert payload == {}
        self.row += 1
        self.needs_observation = self.row % self.chunk == 0
        return dict(left_arm=[0.] * 6, right_arm=[0.] * 6, left_gripper=[1.], right_gripper=[1.])
    def close(self): pass


@pytest.mark.parametrize('chunk', [1, 2, 8, 32])
def test_chunk_observation_is_fresh_and_shared_with_video(tmp_path, monkeypatch, chunk):
    from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
    spec, task, ctx, seen = episode(tmp_path, monkeypatch)
    transport = ChunkTransport(chunk)
    result = run_episode(spec, task, ctx, XPolicyLabAdapter(transport), limits=EpisodeLimits(160, 60))
    assert result.infra_valid, result.error
    requested = set(range(0, 160, chunk * 4))
    video_ticks = set(range(20, 161, 20))
    expected = sorted((requested | video_ticks) - {0})
    assert seen.renders == seen.captures == expected
    assert seen.samples == [0, *expected]
    assert seen.frame_ticks == [0, *sorted(video_ticks)]
    assert transport.image_ticks == sorted(requested)
    assert seen.observed_rgb == [(tick, tick in requested) for tick in range(0, 160, 4)]
    assert seen.evidence_ticks == list(range(1, 161))
    assert dict(result.video_metadata['frame_counts']) == {name: 9 for name in CAMERAS}
    assert result.video_metadata['video_hz'] == 6
    assert result.policy_metadata['clock']['validated'] is True
    assert dict(result.policy_metadata['clock']['required']) == dict(control_hz=30, rgb_hz=30)
    assert json.loads((tmp_path/'spec.json').read_text())['clock']['rgb_hz'] == 30


def test_scorer_can_request_an_off_cadence_image_without_extra_video(tmp_path, monkeypatch):
    from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
    spec, task, ctx, seen = episode(tmp_path, monkeypatch, evidence_rgb_ticks=(7, 20))
    result = run_episode(spec, task, ctx, XPolicyLabAdapter(ChunkTransport(32)), limits=EpisodeLimits(24, 60))
    assert result.infra_valid, result.error
    assert seen.renders == seen.captures == [7, 20]
    assert seen.frame_ticks == [0, 20]
    assert seen.evidence_ticks == list(range(1, 25))


@pytest.mark.parametrize('value', [0, -1, 7, 121, True, 6., None])
def test_video_rate_rejects_invalid_physics_cadence(value):
    with pytest.raises(ValueError, match='video_hz'):
        video_interval(120, value)


def test_unknown_adapter_retains_every_requested_observation(tmp_path, monkeypatch):
    from mind2act_eval.adapters.demo import DemoAdapter
    spec, task, ctx, seen = episode(tmp_path, monkeypatch)
    result = run_episode(spec, task, ctx, DemoAdapter(), limits=EpisodeLimits(24, 60))
    assert result.infra_valid, result.error
    assert seen.renders == seen.captures == [4, 8, 12, 16, 20]
    assert seen.frame_ticks == [0, 20]
    assert all(present for _, present in seen.observed_rgb)


def test_configured_video_rate_keeps_clock_and_physics(tmp_path, monkeypatch):
    from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
    spec, task, ctx, seen = episode(tmp_path, monkeypatch, video_hz=10)
    result = run_episode(spec, task, ctx, XPolicyLabAdapter(ChunkTransport(32)), limits=EpisodeLimits(120, 60))
    assert result.infra_valid, result.error
    assert seen.renders == seen.frame_ticks[1:] == list(range(12, 121, 12))
    assert ctx.sim.world.current_time_step_index == result.ticks == 120
    assert spec.clock.rgb_hz == spec.clock.control_hz == 30
    assert result.video_metadata['video_hz'] == 10


def test_three_saved_mp4s_have_6hz_timestamps_and_actual_frame_counts(tmp_path, monkeypatch):
    import shutil
    import subprocess
    from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
    from mind2act_eval.sim.video import RGBRecorder
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('explicit local ffmpeg and ffprobe required')
    spec, task, ctx, seen = episode(tmp_path, monkeypatch)
    ctx.sim.recorder = RGBRecorder(tmp_path, 6)
    ctx.sim.close = ctx.sim.recorder.close
    result = run_episode(spec, task, ctx, XPolicyLabAdapter(ChunkTransport(32)), limits=EpisodeLimits(120, 60))
    assert result.infra_valid, result.error
    assert dict(result.video_metadata['frame_counts']) == {name: 7 for name in CAMERAS}
    for name in CAMERAS:
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                                    '-of', 'json', str(tmp_path / (name + '.mp4'))]))['streams'][0]
        assert info['r_frame_rate'] == info['avg_frame_rate'] == '6/1'
        assert int(info['nb_frames']) == 7
        assert float(info['duration']) == pytest.approx(7 / 6, abs=1e-5)
        frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_frames',
                                                      '-of', 'json', str(tmp_path / (name + '.mp4'))]))['frames']
        assert [float(frame['pts_time']) for frame in frames] == pytest.approx([i / 6 for i in range(7)], abs=1e-5)


def test_native_coding_agent_decisions_get_current_three_camera_frames(tmp_path, monkeypatch):
    from mind2act_eval.adapters.coding_agent_native import NativeCodingAgentAdapter
    from test_coding_agent_native import Integration, chunk, row
    spec, task, ctx, seen = episode(tmp_path, monkeypatch)
    integration = Integration([chunk(*[row() for _ in range(8)]) for _ in range(3)])
    result = run_episode(spec, task, ctx, NativeCodingAgentAdapter(integration), limits=EpisodeLimits(96, 60))
    assert result.infra_valid, result.error
    assert integration.calls == [0, 32, 64]
    for obs in integration.observations:
        assert set(obs.rgb) == set(CAMERAS)
        assert all(int(image[0, 0, 0]) == obs.tick for image in obs.rgb.values())
    assert seen.renders == seen.captures == [20, 32, 40, 60, 64, 80]
    assert seen.frame_ticks == [0, 20, 40, 60, 80]
    assert integration.confirmed == [8, 8, 8]
    assert dict(result.video_metadata['frame_counts']) == {name: 5 for name in CAMERAS}


def test_single_action_coding_bridge_gets_current_frames_at_each_decision(tmp_path, monkeypatch):
    from mind2act_eval.adapters.coding_agent import CodingAgentAdapter
    from mind2act_eval.contracts import Action
    spec, task, ctx, seen = episode(tmp_path, monkeypatch)
    decisions = []
    def call_tool(name, *, observation):
        decisions.append(observation.tick)
        assert name == 'act' and set(observation.rgb) == set(CAMERAS)
        assert all(int(frame[0, 0, 0]) == observation.tick for frame in observation.rgb.values())
        return Action((0.,) * 6, (0.,) * 6, 1., 1.)
    transport = NS(reset=lambda **kwargs: None, call_tool=call_tool, close=lambda: None)
    result = run_episode(spec, task, ctx, CodingAgentAdapter(transport), limits=EpisodeLimits(24, 60))
    assert result.infra_valid, result.error
    assert decisions == list(range(0, 24, 4))
    assert seen.frame_ticks == [0, 20]
    assert result.video_metadata['video_hz'] == 6


@pytest.mark.parametrize('video_hz', [6, 10, 30])
def test_batch_manifest_forwards_recording_rate_to_isolated_worker(video_hz):
    from pathlib import Path
    from mind2act_eval.cli import _batch_jobs, _job_defaults
    defaults = _job_defaults()
    defaults.update(spec=str(Path(__file__).resolve().parents[1] / 'manifests' / 'demo.json'),
                    asset_root='.', backend='demo', physics_ticks=8, wall_s=10)
    job, = _batch_jobs([dict(video_hz=video_hz)], defaults)
    assert job.argv[job.argv.index('--video-hz') + 1] == str(video_hz)
    assert job.spec.clock.rgb_hz == 30
    with pytest.raises(ValueError, match='video_hz'):
        _batch_jobs([dict(video_hz=7)], defaults)
