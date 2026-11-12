from pathlib import Path
import pytest
from mind2act_eval.contracts import (
    BuildContext, ClockSpec, ConsumerKind, Health, InstanceSpec,
    NativeEvidence, Observation, Score, Termination,
)
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.tasks import TaskDefinition
from mind2act_eval.sim.demo import DemoSimulator, DemoRobot
from mind2act_eval.adapters.demo import DemoAdapter


def fixture(tmp_path, *, health=Health(True)):
    events = []
    spec = InstanceSpec("demo", "one", "easy", "public", "v1", "none", "v1")
    robot = DemoRobot()
    class Simulator(DemoSimulator):
        def step(self, *, render):
            events.append(("step", self.ticks + 1))
            super().step(render=render)
    class Scene:
        def before_step(self, tick):
            events.append(("before", tick))
        def after_step(self, tick):
            events.append(("after", tick))
        def observe(self, tick):
            return Observation(tick, tick / 120, **robot.observe())
        def evidence(self, tick):
            events.append(("evidence", tick))
            return NativeEvidence("demo", "one", tick, {"ticks": tick})
        def health(self):
            return health
        def is_complete(self):
            return False
        def close(self):
            pass
    ctx = BuildContext(Simulator(), robot, None, tmp_path)
    task = TaskDefinition("demo", lambda spec, ctx: Scene(), lambda e: Score(25.))
    return spec, task, ctx, events


def test_every_physics_tick_samples_once_after_hooks(tmp_path):
    spec, task, ctx, events = fixture(tmp_path)
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid, result.error
    assert events == [(name, tick) for tick in range(1, 5)
                      for name in ("before", "step", "after", "evidence")]
    assert result.score == 25.


def test_random_resolution_failure_retains_requested_variation_without_claiming_apply(tmp_path):
    import json
    from dataclasses import replace
    from mind2act_eval.contracts import VariationSpec, to_json
    from mind2act_eval.sim import LocalAssets
    from mind2act_eval.variations import base_spec_digest
    spec, task, ctx, events = fixture(tmp_path)
    spec = replace(spec, variation=VariationSpec('random', 'missing-layout', '1'*64,
                   base_spec_digest(spec), '2'*64))
    ctx = replace(ctx, assets=LocalAssets(tmp_path, {'variation_manifest':'missing-manifest.json'}))
    result = run_episode(spec, task, ctx, DemoAdapter(), limits=EpisodeLimits(4,10))
    assert not result.infra_valid and 'FileNotFoundError' in result.error
    assert events == []
    variation = json.loads((tmp_path/'result.json').read_text())['task_metadata']['variation']
    assert all(variation[k] == v for k,v in to_json(spec.variation).items())
    assert variation['authored'] is False and variation['applied'] is False
    assert variation['plan_sha256'] is None
    assert variation['receipt'] is None and variation['receipt_sha256'] is None
    assert 'variation_receipt' not in result.artifacts


@pytest.mark.parametrize("consumer", [ConsumerKind.MODEL, ConsumerKind.EXPERT_REPLAY])
def test_runner_interpolates_same_chunk_rows_and_holds_terminal_row(tmp_path, consumer):
    from mind2act_eval.contracts import Action
    spec, task, ctx, _ = fixture(tmp_path)
    rows = [Action((v,) * 6, (-v,) * 6, v / 4, 1 - v / 4) for v in (0., 4.)]
    class Adapter(DemoAdapter):
        index = 0
        def act(self, obs):
            assert obs.tick == self.index * 4
            if self.index == len(rows):
                raise StopIteration
            action = rows[self.index]
            self.index += 1
            return action
        def peek_next_action(self):
            return rows[self.index] if self.index < len(rows) else None
    applied = []
    ctx.robot.apply = applied.append
    result = run_episode(spec, task, ctx, Adapter(), consumer_kind=consumer,
                         limits=EpisodeLimits(8, 10))
    assert result.infra_valid, result.error
    assert [a.left_arm[0] for a in applied] == [0., 1., 2., 3., 4., 4., 4., 4.]
    assert [a.right_arm[0] for a in applied] == [0., -1., -2., -3., -4., -4., -4., -4.]
    assert [a.left_gripper for a in applied] == [0., .25, .5, .75, 1., 1., 1., 1.]
    assert [a.right_gripper for a in applied] == [1., .75, .5, .25, 0., 0., 0., 0.]


@pytest.mark.parametrize("budget", [6, 16])
def test_h5_interpolation_preserves_rows_eof_and_full_final_period(tmp_path, budget):
    import h5py
    import numpy as np
    import json
    from mind2act_eval.replay import H5Replay, ReplayMetadata
    from mind2act_eval.contracts import to_json
    spec, task, ctx, _ = fixture(tmp_path)
    path = tmp_path / "linear.h5"
    with h5py.File(path, "w") as h5:
        for side in ("left", "right"):
            h5.create_dataset(f"action/{side}_arm_joint_states", data=np.repeat([[0.], [4.], [8.]], 6, axis=1))
            h5.create_dataset(f"action/{side}_ee_joint_states", data=[[0.], [.5], [1.]])
    adapter = H5Replay(path, ReplayMetadata("episode", spec.source_id, 120, 30, 30,
                                           "rad", "normalized", 1., 0., instance_spec=to_json(spec)))
    applied = []
    ctx.robot.apply = applied.append
    result = run_episode(spec, task, ctx, adapter, consumer_kind=ConsumerKind.EXPERT_REPLAY,
                         limits=EpisodeLimits(budget, 10))
    assert result.infra_valid, result.error
    assert [a.left_arm[0] for a in applied] == (list(range(8)) + [8.] * 4)[:budget]
    assert [a.left_gripper for a in applied] == ([i / 8 for i in range(8)] + [1.] * 4)[:budget]
    coverage = json.loads((tmp_path / "replay-coverage.json").read_text())
    assert coverage["adapter_returned_rows"] == coverage["dispatched_rows"] == (2 if budget == 6 else 3)
    assert coverage["clock_coverage_complete"] is (budget == 16)
    assert coverage['executed_physics_ticks'] == min(budget, 12)
    assert result.ticks == min(budget, 12)
    if budget == 16:
        assert 'post_eof_settle_ticks' not in coverage
        assert result.termination == Termination.TRAJECTORY_EXHAUSTED


def test_render_preparation_time_is_outside_episode_wall_budget(tmp_path, monkeypatch):
    import mind2act_eval.runner as runner
    spec, task, ctx, _ = fixture(tmp_path)
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    def cold_renderer_ready():
        clock[0] = 100.
    ctx.sim.ready = cold_renderer_ready
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid, result.error
    assert result.ticks == 4
    assert result.timings["build_s"] == 100.


def test_successful_episode_prepares_for_sealed_exit_instead_of_sim_close(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path)
    calls = []
    ctx.sim.prepare_for_os_exit = lambda: calls.append("prepare")
    ctx.sim.close = lambda: pytest.fail("successful native worker must skip app.close")

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))

    assert result.infra_valid
    assert calls == ["prepare"]


def test_none_health_is_invalid_instead_of_none_get_crash(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path, health=None)
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC)
    assert result.infra_valid is False
    assert result.score is None
    assert result.termination == Termination.ERROR
    assert "Health" in result.error


def test_failed_nonisolated_episode_retains_normal_sim_close(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path, health=None)
    calls = []
    ctx.sim.prepare_for_os_exit = lambda: calls.append('prepare')
    ctx.sim.close = lambda: calls.append('close')

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC)

    assert result.infra_valid is False and result.score is None
    assert calls == ['close']


def test_adapter_metadata_survives_close_and_is_written_to_result(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    class Adapter(DemoAdapter):
        def __init__(self):
            self.closed = False
        def result_metadata(self):
            assert not self.closed
            return {"configured_checkpoint": "configured-only", "requested_episode_seed": 0,
                    "seed_confirmed": False, "remote_model_identity_verified": False}
        def close(self):
            self.closed = True
    result = run_episode(spec, task, ctx, Adapter(), consumer_kind=ConsumerKind.MODEL,
                         limits=EpisodeLimits(4, 10))
    saved = json.loads((tmp_path / "result.json").read_text())
    assert result.model_tested is True
    assert saved["policy_metadata"]["seed_confirmed"] is False
    assert saved["policy_metadata"]["configured_checkpoint"] == "configured-only"


def test_robot_native_readback_receipt_is_archived_and_linked_from_result(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    expected = {"schema": "mind2act_eval.native_robot_receipt.v1",
                "status": "verified", "robots": {"left": {"joint7_max_effort_N": 100.0}}}
    ctx.robot.native_receipt = lambda: expected

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))

    stored = json.loads((tmp_path / "robot-receipt.json").read_text())
    result_json = json.loads((tmp_path / "result.json").read_text())
    assert {key: stored[key] for key in expected} == expected
    assert stored["task_id"] == "demo"
    assert stored["instance_id"] == "one"
    assert result_json["artifacts"]["native_robot_receipt"] == "robot-receipt.json"
    assert result.infra_valid


def test_robot_native_readback_failure_still_archives_explicit_unsupported_receipt(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    task = TaskDefinition("demo", lambda spec, ctx: (_ for _ in ()).throw(RuntimeError("build failed")),
                          task.scorer)
    ctx.robot.native_receipt = lambda: {"schema": "mind2act_eval.native_robot_receipt.v1",
                                       "status": "unavailable",
                                       "unsupported": ["native initialization did not complete"]}

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC)

    receipt = json.loads((tmp_path / "robot-receipt.json").read_text())
    stored = json.loads((tmp_path / "result.json").read_text())
    assert receipt["status"] == "unavailable"
    assert result.infra_valid is False
    assert stored["artifacts"]["native_robot_receipt"] == "robot-receipt.json"


def test_preclose_trace_is_persisted_before_simulator_close(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    ctx.sim.artifacts = {"native_camera_receipt": "camera-receipt.json",
                         "native_shutdown_receipt": "shutdown-receipt.json"}
    def close_with_early_exit():
        assert (tmp_path / "preclose-trace.json").exists()
        assert (tmp_path / "cleanup-started.json").exists()
        raise SystemExit(0)
    ctx.sim.close = close_with_early_exit

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))

    trace = json.loads((tmp_path / "preclose-trace.json").read_text())
    assert trace["stage"] == "awaiting_cleanup"
    assert trace["result"]["score"] == 25.
    assert trace["sim_artifacts"]["native_camera_receipt"] == "camera-receipt.json"
    assert trace["sim_artifacts"]["native_shutdown_receipt"] == "shutdown-receipt.json"
    assert result.artifacts["cleanup_started"] == "cleanup-started.json"
    assert result.infra_valid is False


def test_task_metadata_and_score_components_are_written_into_result(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    scene = task.build(spec, ctx)
    scene.result_metadata = lambda: {"cue_duration_s": 1.0, "direction_gate_enabled": False}
    task = TaskDefinition(
        "demo", lambda spec, ctx: scene,
        lambda evidence: Score(7.5, {"task_success": False,
                                    "completion": {"k": 3, "n": 8, "ratio": 3 / 8}}),
    )
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))
    saved = json.loads((tmp_path / "result.json").read_text())
    assert result.task_metadata["cue_duration_s"] == 1.0
    assert saved["task_metadata"]["direction_gate_enabled"] is False
    assert saved["score_components"]["completion"] == {"k": 3, "n": 8, "ratio": 3 / 8}


def test_nonmapping_score_components_produce_archived_invalid_result(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    task = TaskDefinition("demo", task.build, lambda evidence: Score(25., ["not-a-mapping"]))
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))
    saved = json.loads((tmp_path / "result.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert "components must be a mapping" in result.error
    assert saved["infra_valid"] is False
    assert saved["score"] is None


def test_expert_replay_exhaustion_retains_score_without_claiming_model_test(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path)
    class Exhausted(DemoAdapter):
        def act(self, observation):
            raise StopIteration
    result = run_episode(spec, task, ctx, Exhausted(), consumer_kind=ConsumerKind.EXPERT_REPLAY)
    assert result.score == 25.
    assert result.model_tested is False
    assert result.termination == Termination.TRAJECTORY_EXHAUSTED


def test_expert_replay_runs_every_h5_row_and_full_hold_after_task_completes(tmp_path):
    import h5py
    import json
    import numpy as np
    from mind2act_eval.replay import H5Replay, ReplayMetadata
    from mind2act_eval.contracts import to_json

    spec = InstanceSpec("demo", "one", "easy", "public", "v1", "none", "v1")
    path = tmp_path / "three-rows.h5"
    with h5py.File(path, "w") as h5:
        for side in ("left", "right"):
            h5.create_dataset(f"action/{side}_arm_joint_states", data=np.zeros((3, 6)))
            h5.create_dataset(f"action/{side}_ee_joint_states", data=np.ones((3, 1)))
    replay = H5Replay(path, ReplayMetadata(
        "episode-141", spec.source_id, 120, 30, 30, "rad", "normalized", 1., 0.,
        instance_spec=to_json(spec),
    ))
    _, task, ctx, _ = fixture(tmp_path)
    # Bind a three-row replay to the real runner seam and complete on physical tick 1.
    ctx.robot.applied = []
    original_apply = ctx.robot.apply
    def apply(action):
        ctx.robot.applied.append(action)
        original_apply(action)
    ctx.robot.apply = apply
    task = TaskDefinition("demo", lambda spec, ctx: _complete_after_one_tick_scene(ctx), task.scorer)

    result = run_episode(spec, task, ctx, replay, consumer_kind=ConsumerKind.EXPERT_REPLAY,
                         limits=EpisodeLimits(20, 10))

    assert result.infra_valid, result.error
    assert result.ticks == 12  # Three actions, each held for all four physics ticks.
    assert result.termination == Termination.TRAJECTORY_EXHAUSTED
    assert result.model_tested is False
    assert len(ctx.robot.applied) == 12  # One command applied on every physics tick in each hold.
    assert result.artifacts["replay_coverage"] == "replay-coverage.json"
    receipt = json.loads((tmp_path / "replay-coverage.json").read_text())
    assert receipt == {
        "schema": "mind2act_eval.replay_coverage.v1",
        "rows_total": 3,
        "adapter_returned_rows": 3,
        "dispatched_rows": 3,
        "adapter_eof": True,
        "ticks_per_action": 4,
        "expected_physics_ticks": 12,
        "executed_physics_ticks": 12,
        "last_hold_expected_ticks": 4,
        "last_hold_executed_ticks": 4,
        "task_first_complete_tick": 1,
        "clock_coverage_complete": True,
        "truncated": False,
    }


def _complete_after_one_tick_scene(ctx):
    class Scene:
        def before_step(self, tick): pass
        def after_step(self, tick): pass
        def observe(self, tick):
            return Observation(tick, tick / 120, **ctx.robot.observe())
        def evidence(self, tick):
            return NativeEvidence("demo", "one", tick, {"ticks": tick})
        def health(self): return Health(True)
        def is_complete(self): return ctx.sim.ticks >= 1
        def close(self): pass
    return Scene()


def test_model_runs_to_budget_despite_transient_task_completion(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path)
    task = TaskDefinition("demo", lambda spec, ctx: _complete_after_one_tick_scene(ctx), task.scorer)

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.MODEL,
                         limits=EpisodeLimits(20, 10))

    assert result.infra_valid
    assert result.ticks == 20
    assert result.termination == Termination.PHYSICS_BUDGET


def _three_row_replay(tmp_path, spec):
    import h5py
    import json
    import numpy as np
    from mind2act_eval.replay import H5Replay, ReplayMetadata
    from mind2act_eval.contracts import to_json

    path = tmp_path / "budget-rows.h5"
    with h5py.File(path, "w") as h5:
        for side in ("left", "right"):
            h5.create_dataset(f"action/{side}_arm_joint_states", data=np.zeros((3, 6)))
            h5.create_dataset(f"action/{side}_ee_joint_states", data=np.ones((3, 1)))
    return H5Replay(path, ReplayMetadata(
        "episode-budget", spec.source_id, 120, 30, 30, "rad", "normalized", 1., 0.,
        instance_spec=to_json(spec),
    ))


@pytest.mark.parametrize("budget, rows, last_hold", [(8, 2, 4), (6, 2, 2), (1, 1, 1), (12, 3, 4)])
def test_expert_replay_budget_cut_scores_and_records_actual_coverage(tmp_path, budget, rows, last_hold):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    replay = _three_row_replay(tmp_path, spec)
    task = TaskDefinition("demo", lambda spec, ctx: _complete_after_one_tick_scene(ctx), task.scorer)

    result = run_episode(spec, task, ctx, replay, consumer_kind=ConsumerKind.EXPERT_REPLAY,
                         limits=EpisodeLimits(budget, 10))

    assert result.infra_valid, result.error
    assert result.score == 25.
    assert result.termination == Termination.PHYSICS_BUDGET
    assert result.ticks == ctx.sim.ticks == budget
    assert result.model_tested is False
    assert replay._file.id.valid == 0
    coverage = json.loads((tmp_path / "replay-coverage.json").read_text())
    assert coverage["rows_total"] == 3
    assert coverage["adapter_returned_rows"] == coverage["dispatched_rows"] == rows
    assert coverage["executed_physics_ticks"] == budget
    assert coverage["expected_physics_ticks"] == 12
    assert coverage["last_hold_expected_ticks"] == 4
    assert coverage["last_hold_executed_ticks"] == last_hold
    assert coverage["adapter_eof"] is False
    assert coverage["clock_coverage_complete"] is False
    assert coverage["truncated"] is True
    assert json.loads((tmp_path / "evidence.json").read_text())["tick"] == budget
    assert json.loads((tmp_path / "score.json").read_text())["value"] == 25.


@pytest.mark.parametrize("fault", ["native", "health", "watchdog"])
def test_expert_replay_fault_at_budget_is_still_invalid(tmp_path, monkeypatch, fault):
    from mind2act_eval import runner
    spec, task, ctx, _ = fixture(tmp_path)
    replay = _three_row_replay(tmp_path, spec)
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original_step = ctx.sim.step
    def step(*, render):
        original_step(render=render)
        if ctx.sim.ticks == 6:
            if fault == "native":
                raise RuntimeError("synthetic native failure at budget")
            if fault == "watchdog":
                clock[0] = 20.
    ctx.sim.step = step
    original_build = task.build
    def build(spec, ctx):
        scene = original_build(spec, ctx)
        if fault == "health":
            scene.health = lambda: Health(ctx.sim.ticks < 6, "synthetic bad native health")
        return scene
    task = TaskDefinition("demo", build, lambda e: pytest.fail("fault must not be scored"))

    result = run_episode(spec, task, ctx, replay, consumer_kind=ConsumerKind.EXPERT_REPLAY,
                         limits=EpisodeLimits(6, 10))

    assert not result.infra_valid
    assert result.score is None
    assert result.ticks == 6
    assert result.termination == (Termination.WALL_TIMEOUT if fault == "watchdog" else Termination.ERROR)
    assert not (tmp_path / "score.json").exists()
    assert replay._file.id.valid == 0


def test_model_physics_budget_keeps_partial_hold_behavior(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path)
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.MODEL,
                         limits=EpisodeLimits(6, 10))
    assert result.infra_valid
    assert result.score == 25.
    assert result.ticks == ctx.sim.ticks == 6
    assert result.termination == Termination.PHYSICS_BUDGET
    assert result.model_tested
    assert not (tmp_path / "replay-coverage.json").exists()


def test_budget_replay_closes_three_cpu_video_producers_before_sealing(tmp_path):
    import shutil
    import numpy as np
    from mind2act_eval.runner import write_json
    from mind2act_eval.sim.video import RGBRecorder
    from mind2act_eval.sealing import seal_worker_outputs, validate_worker_seal

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("explicit local ffmpeg required")
    # Entirely synthetic public frames; this is a producer-closure seam, not Isaac media.
    spec, task, ctx, _ = fixture(tmp_path)
    replay = _three_row_replay(tmp_path, spec)
    recorder = RGBRecorder(tmp_path, 30, ffmpeg=ffmpeg)
    frames = {name: np.zeros((16, 16, 3), dtype=np.uint8)
              for name in ("head", "left_wrist", "right_wrist")}
    ctx.sim.ready = lambda: recorder.capture(frames)
    ctx.sim.capture = lambda: recorder.capture(frames)
    ctx.sim.artifacts = recorder.artifacts
    producers = []
    def prepare():
        producers.extend(child for child, _, _ in recorder._writers.values())
        recorder.close()
        assert len(producers) == 3 and all(child.returncode == 0 for child in producers)
        assert not recorder._writers
    ctx.sim.prepare_for_os_exit = prepare
    ctx.sim.close = recorder.close
    try:
        result = run_episode(spec, task, ctx, replay, consumer_kind=ConsumerKind.EXPERT_REPLAY,
                             limits=EpisodeLimits(6, 10))
    finally:
        recorder.close()
    assert result.infra_valid, result.error
    assert len(producers) == 3
    assert {"video_head", "video_left_wrist", "video_right_wrist"} <= set(result.artifacts)
    # Supply synthetic worker lifecycle markers after the real runner returned.
    for filename, stage, details in [
        ("run-episode-returned.json", "run_episode_returned", {"infra_valid": True}),
        ("worker-returned.json", "worker_returned", {"exit_code": 0}),
        ("cli-returned.json", "cli_returned", {"exit_code": 0}),
    ]:
        write_json(tmp_path / filename, {"schema": "mind2act_eval.worker_lifecycle.v1",
                                       "stage": stage, "pid": 12345, **details})
    seal_worker_outputs(tmp_path)
    checked, _ = validate_worker_seal(tmp_path)
    assert checked == result
    assert all(child.poll() == 0 for child in producers)


def test_bad_adapter_metadata_still_produces_invalid_result_artifact(tmp_path):
    spec, task, ctx, _ = fixture(tmp_path)
    class BadMetadata(DemoAdapter):
        def result_metadata(self):
            return ["not a mapping"]
    result = run_episode(spec, task, ctx, BadMetadata(), consumer_kind=ConsumerKind.MODEL,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid is False
    assert result.score is None
    assert (tmp_path / "result.json").exists()


def test_adapter_clock_validation_rejects_before_first_action_and_preserves_reset_metadata(tmp_path):
    from dataclasses import replace
    spec, task, ctx, _ = fixture(tmp_path)
    spec = replace(spec, clock=ClockSpec(120, 120, 30), seed=17)
    class Fixed30Hz(DemoAdapter):
        def reset(self, seed):
            self.seed = seed
        def validate_clock(self, clock):
            assert self.seed == 17
            if clock != ClockSpec(120, 30, 30):
                raise ValueError("adapter clock mismatch")
        def act(self, observation):
            raise AssertionError("must reject before any action")
        def result_metadata(self):
            return {"requested_episode_seed": self.seed}
    result = run_episode(spec, task, ctx, Fixed30Hz(), consumer_kind=ConsumerKind.MODEL,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid is False
    assert result.score is None
    assert "adapter clock mismatch" in result.error
    assert result.policy_metadata["requested_episode_seed"] == 17
    assert result.model_tested is False


def test_failed_inference_and_reset_keep_wall_time_separate_from_observation(tmp_path, monkeypatch):
    from mind2act_eval import runner
    spec, task, ctx, _ = fixture(tmp_path)
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original_build = task.build
    def build(spec, ctx):
        scene = original_build(spec, ctx)
        observe = scene.observe
        def delayed_observe(tick):
            clock[0] += 5.
            return observe(tick)
        scene.observe = delayed_observe
        return scene
    task = TaskDefinition(task.task_id, build, task.scorer)
    class FailingAdapter(DemoAdapter):
        def reset(self, seed):
            clock[0] += 2.
        def act(self, observation):
            clock[0] += 3.
            raise RuntimeError("failed inference")
    result = run_episode(spec, task, ctx, FailingAdapter(), consumer_kind=ConsumerKind.MODEL,
                         limits=EpisodeLimits(4, 20))
    assert result.infra_valid is False
    assert result.timings["adapter_s"] == 5.
    assert result.timings["observation_s"] == 5.
    assert result.timings["overhead_s"] == 0.


def test_failed_reset_also_retains_its_wall_time(tmp_path, monkeypatch):
    from mind2act_eval import runner
    spec, task, ctx, _ = fixture(tmp_path)
    clock = [0.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    class FailingReset(DemoAdapter):
        def reset(self, seed):
            clock[0] += 2.
            raise RuntimeError("failed reset")
    result = run_episode(spec, task, ctx, FailingReset(), consumer_kind=ConsumerKind.MODEL)
    assert result.timings["adapter_s"] == 2.
    assert result.timings["overhead_s"] == 0.


def test_cleanup_trace_is_armed_before_resources_close(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    calls = []
    def arm_diagnostics():
        calls.append("arm")
        (tmp_path / "cleanup-traceback.log").touch()
        return {"signal_name": "SIGUSR1"}
    ctx.sim.arm_cleanup_diagnostics = arm_diagnostics
    ctx.sim.close = lambda: calls.append("sim.close")

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))

    assert result.infra_valid
    assert calls == ["arm", "sim.close"]
    marker = json.loads((tmp_path / "cleanup-trace-armed.json").read_text())
    assert marker["timeout_s"] == 90
    assert marker["repeat"] is True
    assert marker["signal"] == "SIGUSR1"
    assert result.artifacts["cleanup_traceback"] == "cleanup-traceback.log"


def test_system_exit_from_sim_close_does_not_skip_invalid_result_artifact(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    task = TaskDefinition("demo", lambda spec, ctx: (_ for _ in ()).throw(RuntimeError("build failed")),
                          task.scorer)
    def close_with_zero_exit():
        raise SystemExit(0)
    ctx.sim.close = close_with_zero_exit

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC)

    saved = json.loads((tmp_path / "result.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert "build failed" in result.error
    assert "SystemExit" in result.error
    assert saved["infra_valid"] is False
    assert saved["score"] is None


def test_system_exit_zero_during_build_is_archived_as_invalid_result(tmp_path):
    import json
    spec, task, ctx, _ = fixture(tmp_path)
    task = TaskDefinition("demo", lambda spec, ctx: (_ for _ in ()).throw(SystemExit(0)),
                          task.scorer)

    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC)

    saved = json.loads((tmp_path / "result.json").read_text())
    assert result.infra_valid is False
    assert result.score is None
    assert "SystemExit(0) during episode" in result.error
    assert saved["infra_valid"] is False
    assert saved["score"] is None


def initial_hold_fixture(tmp_path, hold_ticks):
    from dataclasses import replace
    spec, task, ctx, events = fixture(tmp_path)
    build = task.build
    def build_with_hold(spec, ctx):
        scene = build(spec, ctx)
        scene.initial_model_hold_ticks = lambda: hold_ticks
        observe = scene.observe
        def observed(tick):
            events.append(('observe', tick))
            return observe(tick)
        scene.observe = observed
        return scene
    return spec, replace(task, build=build_with_hold), ctx, events


def test_retired_initial_hold_does_not_hide_observations_or_actions(tmp_path):
    from mind2act_eval.contracts import Action
    spec, task, ctx, events = initial_hold_fixture(tmp_path, 600)
    applied, calls, captures, confirmed = [], [], [], []
    ctx.robot.apply = applied.append
    ctx.sim.capture = lambda: captures.append(ctx.sim.ticks)
    action = Action((.4,) * 6, (-.5,) * 6, .1, .2)
    class Adapter(DemoAdapter):
        def act(self, observation):
            calls.append(observation.tick)
            return action
        def confirm_action_executed(self, ticks):
            confirmed.append(ticks)
    result = run_episode(spec, task, ctx, Adapter(), limits=EpisodeLimits(604, 10))
    assert result.infra_valid, result.error
    assert result.ticks == ctx.sim.ticks == 604
    assert calls == list(range(0, 604, 4)) and confirmed == [4] * 151
    assert applied == [action] * 604
    assert captures == list(range(20, 601, 20))
    assert [tick for name, tick in events if name == 'evidence'] == list(range(1, 605))
    assert not (tmp_path / 'initial-environment-hold.json').exists()


def test_retired_initial_hold_does_not_consume_control_budget(tmp_path):
    spec, task, ctx, events = initial_hold_fixture(tmp_path, 600)
    result = run_episode(spec, task, ctx, DemoAdapter(), limits=EpisodeLimits(13, 10))
    assert result.infra_valid, result.error
    assert result.ticks == ctx.sim.ticks == 13 and result.model_tested is True
    assert [tick for name, tick in events if name == 'observe'] == [0, 4, 8, 12]
    assert not (tmp_path / 'initial-environment-hold.json').exists()


def test_model_actions_still_obey_wall_budget_without_initial_hold(tmp_path, monkeypatch):
    import mind2act_eval.runner as runner
    spec, task, ctx, _ = initial_hold_fixture(tmp_path, 600)
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    step = ctx.sim.step
    def timed_step(*, render):
        clock[0] += 1.
        step(render=render)
    ctx.sim.step = timed_step
    result = run_episode(spec, task, ctx, DemoAdapter(), limits=EpisodeLimits(604, 4))
    assert result.ticks == ctx.sim.ticks == 4
    assert result.timings['action_s'] == 4.
    assert not result.infra_valid and 'wall budget exhausted' in result.error
    assert not (tmp_path / 'initial-environment-hold.json').exists()


@pytest.mark.parametrize('consumer', [ConsumerKind.EXPERT_REPLAY, ConsumerKind.DIAGNOSTIC])
def test_initial_environment_hold_does_not_shift_source_replay_or_diagnostics(tmp_path, consumer):
    spec, task, ctx, events = initial_hold_fixture(tmp_path, 600)
    observed = []
    class Adapter(DemoAdapter):
        def act(self, observation):
            observed.append(observation.tick)
            return super().act(observation)
    result = run_episode(spec, task, ctx, Adapter(), consumer_kind=consumer,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid, result.error
    assert observed == [0] and result.ticks == 4
    assert not (tmp_path / 'initial-environment-hold.json').exists()


def test_final_scoring_model_starts_at_tick_zero_and_scores_t0(tmp_path):
    from dataclasses import replace
    from mind2act_eval.contracts import PolicyStop
    spec, task, ctx, events = fixture(tmp_path)
    # A task that formerly requested both a prompt hold and post-actor window.
    spec = replace(spec, task_id='PR6')
    old_scene = task.build(spec, ctx)
    class Scene:
        initial_model_hold_ticks = lambda self: 600
        health = old_scene.health
        observe = old_scene.observe
        close = old_scene.close
        def is_complete(self):
            return ctx.sim.ticks >= 1  # transient old success must not stop model
        def is_success(self):
            return False  # Legacy completion is not the new final success.
        def evidence(self, tick):
            return NativeEvidence('PR6', 'one', tick, {'ticks': tick})
    observed = []
    class Adapter(DemoAdapter):
        def act(self, observation):
            observed.append(observation.tick)
            if observation.tick == 4:
                raise PolicyStop("done")
            return super().act(observation)
    task = TaskDefinition('PR6', lambda spec, ctx: Scene(), lambda e: Score(float(e.tick)))
    result = run_episode(spec, task, ctx, Adapter(), limits=EpisodeLimits(8, 10))
    assert result.infra_valid, result.error
    assert observed == [0, 4]
    assert result.ticks == result.score == 4
    assert result.termination == Termination.POLICY_STOP
    import json
    saved = json.loads((tmp_path / 'result.json').read_text())
    assert saved['scoring_version'] == 'mind2act-final-scoring-20261105'
    assert not (tmp_path / 'initial-environment-hold.json').exists()
    assert not (tmp_path / 'observation-closeout.json').exists()
