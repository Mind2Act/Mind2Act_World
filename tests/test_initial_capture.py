"""CPU call-chain regression; SDK boundaries model unsynchronized link transforms.

This exercises production reset/ready/RGB/recording, not native visual proof.
Native clean/random frame-0/H5 comparison remains a separately scheduled gate.
"""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.robot import X5Robot
from mind2act_eval.sim import IsaacSimulator


@pytest.mark.parametrize("chain", ["reset_ready", "runner"])
def test_reset_pose_reaches_all_initial_cameras_without_a_physics_step(tmp_path, monkeypatch, chain):
    events = []
    state = {"actual": np.zeros(8), "links": np.zeros(8), "render": np.zeros(8)}
    rest = np.array([0., 1.1344640137963142, 1.7453292519943295,
                     -1.1344640137963142, 0., 0., .044, .044])

    def module(name, **attrs):
        value = ModuleType(name)
        value.__path__ = []
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.sensors",
                 "omni", "omni.replicator"):
        module(name)
    module("isaacsim.core.utils.types", ArticulationAction=lambda **kw: SimpleNamespace(**kw))
    stage = SimpleNamespace(GetPrimAtPath=lambda path: object())
    module("isaacsim.core.utils.stage", get_current_stage=lambda: stage)

    class Arm:
        def set_joint_positions(self, positions):
            state["actual"] = positions.copy()
            events.append("actual_pose")
        def set_joint_velocities(self, velocities):
            assert not velocities.any()
        def apply_action(self, action):
            np.testing.assert_array_equal(action.joint_positions, rest)
        def get_joint_positions(self):
            return state["actual"].copy()

    def forward_links():
        events.append("forward_links")
        state["links"] = state["actual"].copy()

    def publish_transforms(*args):
        assert args == (True, True, True, False)
        events.append("publish_transforms")
        state["render"] = state["links"].copy()

    module("omni.physx", get_physx_interface=lambda: SimpleNamespace(update_transformations=publish_transforms))

    class World:
        current_time = 0.
        current_time_step_index = 0
        physics_sim_view = SimpleNamespace(update_articulations_kinematic=forward_links)
        def reset(self):
            events.append("reset")
        def render(self):
            events.append("render")
        def step(self, **kwargs):
            pytest.fail("initial capture must not step physics")

    world = World()
    warmups = []
    def warmup(**kw):
        warmups.append(kw)
        events.append("warmup")
    module("omni.replicator.core", orchestrator=SimpleNamespace(
        set_capture_on_play=lambda enabled: None, step=warmup),
        settings=SimpleNamespace(set_render_rtx_realtime=lambda **kw: None))
    module("carb", settings=SimpleNamespace(get_settings=lambda: SimpleNamespace(get=lambda key: None)))

    class Camera:
        def __init__(self, **kwargs): self.reads = 0
        def initialize(self): pass
        def get_rgba(self):
            # Shape-valid USD default image stays raised until link transforms
            # are published. Rest-state image also includes the two finger DOFs.
            self.reads += 1
            value = 180 if np.array_equal(state["render"], rest) else 20
            if self.reads > 1:
                value += 1  # An asynchronous sensor refresh must not replace frame0.
            return np.full((4, 4, 4), value, dtype=np.uint8)

    module("isaacsim.sensors.camera", Camera=Camera)
    monkeypatch.setattr(IsaacSimulator, "world", property(lambda self: world))
    monkeypatch.setattr(IsaacSimulator, "stage", property(lambda self: stage))
    sim = object.__new__(IsaacSimulator)
    sim._ready = False
    sim._gate = None
    sim._output_dir = tmp_path
    sim._camera_config = {name: {"path": "/World/" + name, "resolution": [4, 4]}
                          for name in ("head", "left_wrist", "right_wrist")}
    sim._cameras = {}
    sim._camera_receipt = None
    sim.timings = {"render_s": 0., "video_s": 0.}
    encoded = []
    sim.recorder = SimpleNamespace(capture=lambda frames: encoded.append(frames), artifacts={})
    robot = X5Robot(rest=rest, rgb_reader=sim.rgb)
    robot._arms = {"left": Arm(), "right": Arm()}

    if chain == "reset_ready":
        robot.reset_to_rest()
        sim.ready()
        observed = sim.rgb()
    else:
        from mind2act_eval.contracts import (BuildContext, Health, InstanceSpec,
                                            NativeEvidence, Observation, PolicyStop, Score)
        from mind2act_eval.runner import run_episode, EpisodeLimits
        from mind2act_eval.tasks import TaskDefinition
        spec = InstanceSpec("demo", "initial", "easy", "fixture", "v1", "v1", "v1")
        sim._world = None
        sim._active_task_spec = None
        sim._clock = spec.clock
        sim.prepare_for_os_exit = lambda: None
        sim.arm_cleanup_diagnostics = lambda: {}
        robot._arms = {}
        def articulations(world):
            robot._arms = {"left": Arm(), "right": Arm()}
        robot.articulations = articulations
        robot.attach_cameras = lambda stage: None
        robot.verify_native = lambda: None  # Native drive audit is outside this CPU seam.
        robot.native_receipt = lambda: {}
        scene = SimpleNamespace(
            observe=lambda tick: Observation(tick, 0., **robot.observe()),
            health=lambda: Health(True), is_complete=lambda: False,
            evidence=lambda tick: NativeEvidence("demo", "initial", tick, {}), close=lambda: None)
        def build(spec, ctx):
            robot._configured = robot._mounted = True
            return scene
        observations = []
        def act(observation):
            observations.append(observation)
            raise PolicyStop("done", "initial observation inspected")
        adapter = SimpleNamespace(reset=lambda seed: None, act=act, close=lambda: None)
        task = TaskDefinition("demo", build, lambda evidence: Score(0.))
        result = run_episode(spec, task, BuildContext(sim, robot, None, tmp_path), adapter,
                             limits=EpisodeLimits(4, 10))
        assert result.infra_valid, result.error
        assert result.ticks == 0 and result.model_tested
        assert len(observations) == 1
        np.testing.assert_array_equal(observations[0].left_arm, rest[:6])
        assert observations[0].left_gripper == observations[0].right_gripper == 1.
        observed = observations[0].rgb

    np.testing.assert_array_equal(state["actual"], rest)
    print("trace:", events, "frame0:", {name: int(rgb[0, 0, 0]) for name, rgb in encoded[0].items()})
    assert len(encoded) == 1
    assert all((rgb == 180).all() for rgb in encoded[0].values()), \
        "frame0 still shows USD default pose although actual joints/fingers are at training rest"
    assert len(warmups) == 2
    assert all(kw == dict(rt_subframes=8, delta_time=0., pause_timeline=False,
                          wait_for_render=True) for kw in warmups)
    assert world.current_time_step_index == 0
    for name in observed:
        np.testing.assert_array_equal(observed[name], encoded[0][name])
    # Caller mutation cannot corrupt the retained tick-zero observation either.
    observed["head"][:] = 0
    assert (sim.rgb()["head"] == 180).all()
    assert sim._initial_pose_publication["world_clock_before"] == sim._initial_pose_publication["world_clock_after"]


def test_initial_transform_publication_rejects_clock_advance(monkeypatch):
    world = SimpleNamespace(current_time=0., current_time_step_index=0)
    def forward():
        world.current_time_step_index += 1
    world.physics_sim_view = SimpleNamespace(update_articulations_kinematic=forward)
    monkeypatch.setattr(IsaacSimulator, "world", property(lambda self: world))
    module = ModuleType("omni.physx")
    module.get_physx_interface = lambda: SimpleNamespace(update_transformations=lambda *args: None)
    monkeypatch.setitem(sys.modules, "omni.physx", module)
    sim = object.__new__(IsaacSimulator)
    with pytest.raises(RuntimeError, match="initial pose transform publication advanced"):
        sim._publish_reset_transforms()
    assert sim._initial_pose_publication["status"] == "failed"


def test_first_physics_step_releases_initial_rgb(monkeypatch):
    world = SimpleNamespace(current_time=0., current_time_step_index=0)
    def step(**kwargs):
        world.current_time_step_index += 1
    world.step = step
    monkeypatch.setattr(IsaacSimulator, "world", property(lambda self: world))
    sim = object.__new__(IsaacSimulator)
    sim._initial_rgb = {"head": np.zeros((4, 4, 3), dtype=np.uint8)}
    sim.timings = {"physics_s": 0.}
    sim.step(render=False)
    assert sim._initial_rgb is None
    assert world.current_time_step_index == 1
