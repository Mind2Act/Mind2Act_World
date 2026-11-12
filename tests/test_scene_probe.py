import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.cli import _job, _job_defaults, _worker
from mind2act_eval.contracts import BuildContext, ConsumerKind, Health, InstanceSpec, Result, Termination, to_json
from mind2act_eval.scene_probe import PROJECTION, ProbeInputs, sha256, spec_digest, stability
from mind2act_eval.sealing import seal_worker_outputs, validate_worker_seal
from mind2act_eval.worker import run_probe_loop, write_png


def write(path, value):
    path.write_text(json.dumps(to_json(value)))
    return sha256(path)


@pytest.fixture
def intake(tmp_path):
    spec = InstanceSpec("F2", "original-source-instance", "easy", "native-source", "native-v1", "asset-v1", "v1",
                        seed=5001, config={"source_case": {"seed": 5001}}, assets={"station": "native.usd", "x5": "x5.usd"})
    material = {"id": "wood", "sha256": "a"*64, "mdl_name": "Wood"}
    rows = [{"category_idx": i, "clutter_idx": 0, "default_pos": [i/10, 0, .8],
             "default_ori": [1, 0, 0, 0], "settled_pos": [i/10, 0, .8], "settled_ori": [1, 0, 0, 0],
             "scale": [1, 1, 1], "physics": {"type": "rigid"}, "relative_plane": "Table",
             "type": "cluttered", "yaml_path": "Clutter/clutter.yml"} for i in range(5)]
    layout = {"schema": "mind2act.new15.layout.v1", "task": "F2", "difficulty": "easy", "logical_slot": "F2_easy_clean_016",
              "status": "ok", "Rigid": {"block": rows}, "ground": material,
              "Appearance": {"schema": "mind2act.visual-randomization.v2", "task": "F2", "difficulty": "easy",
                             "mode": "random", "table_enabled": True, "light_enabled": True, "material": material,
                             "light": {"type": "Sphere", "color_rgb": [1, 1, 1], "intensity_scale": 1},
                             "catalog_sha256": "c"*64, "profile_sha256": "d"*64}}
    catalog = {"schema": "mind2act_eval.scene_probe_catalog.v2", "source_spec_sha256": spec_digest(spec), "asset_root": str(tmp_path),
               "clutter": {f"block/{i:05d}": {"path": f"clutter/{i}.usdz", "sha256": "a"*64} for i in range(5)},
               "materials": {"wood": {"path": "wood.mdl", "sha256": "a"*64, "mdl_name": "Wood"}},
               "table_targets": ["/World/Worktable/Top"], "floor_targets": ["/World/Floor"], "key_light": "/World/Light",
               "dome_light": "/World/Studio", "floor_contact_targets": ["/World/Floor", "/World/Ground"],
               "support_z_m": .768, "source_provenance": {"revision": "trusted-baseline"}}
    config = {"schema": "mind2act_eval.scene_probe.v1", "projection": PROJECTION, "task_id": "F2", "difficulty": "easy",
              "logical_slot": "F2_easy_clean_016", "source_spec_sha256": spec_digest(spec),
              "layout": str(tmp_path/"delivery.json"), "layout_sha256": write(tmp_path/"delivery.json", layout),
              "catalog": str(tmp_path/"catalog.json"), "catalog_sha256": write(tmp_path/"catalog.json", catalog),
              "generation_seed": None, "seed_provenance": "not present in delivery; source seed is separate"}
    path = tmp_path/"probe.json"
    digest = write(path, config)
    return SimpleNamespace(spec=spec, layout=layout, catalog=catalog, config=config, path=path, digest=digest)


def read_inputs(intake):
    return ProbeInputs.read(intake.path, intake.digest, intake.spec)


def repin(intake):
    intake.config["layout_sha256"] = write(Path(intake.config["layout"]), intake.layout)
    intake.config["catalog_sha256"] = write(Path(intake.config["catalog"]), intake.catalog)
    intake.digest = write(intake.path, intake.config)


def test_delivery_projection_preserves_source_seed_and_identity(intake):
    before = to_json(intake.spec)
    probe = read_inputs(intake)
    assert len(probe.clutter) == 5
    assert probe.clutter[0].position == (0, 0, .8)
    assert probe.config["generation_seed"] is None
    assert to_json(intake.spec) == before


@pytest.mark.parametrize("change", ["robot", "path", "bad_quaternion", "unknown_asset", "nan", "wrong_spec"])
def test_untrusted_delivery_or_projection_cannot_change_protected_state(intake, change):
    if change == "robot":
        intake.layout["Robot"] = {"stiffness": 0}
    elif change == "path":
        intake.catalog["clutter"]["block/00000"]["path"] = "../../outside.usdz"
    elif change == "bad_quaternion":
        intake.layout["Rigid"]["block"][0]["settled_ori"] = [2, 0, 0, 0]
    elif change == "unknown_asset":
        intake.layout["Rigid"]["block"][0]["category_idx"] = 999
    elif change == "nan":
        intake.layout["Rigid"]["block"][0]["settled_pos"] = [float("nan"), 0, .8]
    else:
        intake.config["source_spec_sha256"] = "0"*64
    repin(intake)
    with pytest.raises(ValueError):
        read_inputs(intake)


def test_catalog_cannot_bind_robot_material_even_when_pinned(intake):
    intake.catalog["table_targets"] = ["/World/RobotLeft/Asset/link1"]
    repin(intake)
    with pytest.raises(ValueError, match="protected"):
        read_inputs(intake)


def test_pinned_layout_substitution_is_rejected(intake):
    Path(intake.config["layout"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        read_inputs(intake)


def args_for(intake):
    path = intake.path.parent/"source-spec.json"
    write(path, intake.spec)
    values = _job_defaults()
    values.update(spec=str(path), asset_root=str(intake.path.parent), backend="isaac", consumer="scene_probe", gpu=0, physics_ticks=600,
                  wall_s=120, probe_config=str(intake.path), probe_config_sha256=intake.digest)
    return argparse.Namespace(**values)


def test_job_uses_existing_supervisor_and_unscored_consumer(intake):
    job = _job(args_for(intake))
    assert job.consumer_kind is ConsumerKind.SCENE_PROBE
    assert job.spec == intake.spec and job.retries == 0
    assert "--consumer" in job.argv and "--h5" not in job.argv


@pytest.mark.parametrize("child_retries,expected", [(None, 0), (1, 2)])
def test_probe_child_argv_parser_preserves_no_retries_admission(intake, tmp_path, monkeypatch, child_retries, expected):
    import mind2act_eval.cli as cli
    job = _job(args_for(intake))
    output = tmp_path / "parsed-child"
    output.mkdir()
    argv = [*job.argv[3:], "--output", str(output)]
    if child_retries is not None:
        argv.extend(["--retries", str(child_retries)])
    seen = []
    def admit_only(args):
        seen.append(args.retries)
        cli._probe_inputs(args, cli.read_spec(args.spec))
        return 0  # CPU admission only; never enter Kit or native execution.
    monkeypatch.setattr(cli, "_worker", admit_only)
    assert cli.main(argv) == expected
    assert seen == [0 if child_retries is None else child_retries]


@pytest.mark.parametrize("key,value", [("h5", "invented.h5"), ("adapter", "arbitrary:run"), ("retries", 1),
                                      ("physics_ticks", 599), ("gpu", -1), ("backend", "demo")])
def test_job_and_direct_worker_reject_probe_bypasses(intake, key, value):
    args = args_for(intake)
    setattr(args, key, value)
    with pytest.raises(ValueError, match="scene_probe requires"):
        _job(args)
    with pytest.raises(ValueError, match="scene_probe requires"):
        _worker(args)


class FakeSim:
    def __init__(self, output, slip=False):
        self._world, self.require_binding, self.bound = None, False, False
        self.stage, self.output, self.slip, self.closed = object(), output, slip, False
        self.artifacts = {"video_head": "head.mp4", "native_camera_receipt": "camera-receipt.json", "native_shutdown_receipt": "shutdown-receipt.json"}
    @property
    def world(self):
        if self.require_binding and not self.bound:
            raise AssertionError("World accessed before source binding")
        if self._world is None:
            self._world = SimpleNamespace(current_time_step_index=17, current_time=17/120)
        return self._world
    def ready(self):
        write(self.output/"camera-receipt.json", {"schema": "mind2act_eval.native_camera_receipt.v1", "status": "verified",
              "initial_frame_encoded": True, "cameras": {"head": {"valid": True}}})
    def rgb(self):
        return {"head": np.zeros((4, 6, 3), dtype=np.uint8)}
    def step(self, render):
        self.world.current_time_step_index += 1
        self.world.current_time += (2 if self.slip else 1)/120
    def capture(self):
        pass
    def prepare_for_os_exit(self):
        self.closed = True
        (self.output/"head.mp4").write_bytes(b"fake-encoded-video-for-CPU-closure-only")
        write(self.output/"shutdown-receipt.json", {"schema": "mind2act_eval.shutdown_receipt.v1", "status": "os_exit_prepared",
              "teardown": "os_exit", "phases": [{"name": "recorder", "status": "complete"}, {"name": "app.close", "status": "skipped"}]})


class FakeScene:
    def __init__(self):
        self.before_ticks, self.steps, self.closed = [], [], False
    def health(self):
        return Health(True)
    def before_step(self, tick):
        self.before_ticks.append(tick)
    def after_step(self, tick):
        self.steps.append(tick)
    def is_complete(self):
        raise AssertionError("diagnostic must not consult task completion")
    def evidence(self, tick):
        raise AssertionError("diagnostic must not invoke task score evidence")
    def close(self):
        self.closed = True


class FakeNative:
    @staticmethod
    def spawn_clutter(stage, inputs, root):
        return [{"name": r.name} for r in inputs.clutter]
    @staticmethod
    def physics_modes(stage):
        return {"/World/Task/Spring": {"joint_enabled": True, "dynamic": True}}
    @staticmethod
    def apply_appearance(stage, inputs, root):
        return []
    class NativeMonitor:
        def __init__(self, ctx, clutter, *, floor_targets):
            assert floor_targets == ["/World/Floor", "/World/Ground"]
            self.clutter = clutter
        def sample(self):
            return {r["name"]: {"position": [0, 0, .8], "orientation": [1, 0, 0, 0], "bottom_z": .768,
                                "floor_contact": False, "contacts": []} for r in self.clutter}


def run_fake(intake, tmp_path, *, slip=False, bind_events=None, bind_error=False, initialization_ticks=0):
    output = tmp_path/"attempt"
    output.mkdir()
    sim, scene = FakeSim(output, slip), FakeScene()
    robot = SimpleNamespace(initialize=lambda world: None, apply=lambda action: pytest.fail("robot action forbidden"))
    if bind_events is not None:
        sim.require_binding = True
        def bind_sim(spec, owned_output):
            assert spec is intake.spec and owned_output == output and sim._world is None
            sim.bound = True
            bind_events.append("sim-bind")
        def bind_robot(spec, owned_output):
            assert spec is intake.spec and owned_output == output
            bind_events.append("robot-bind")
            if bind_error:
                raise ValueError("source binding failed")
        sim.bind_task_instance, robot.bind_task_instance = bind_sim, bind_robot
    ctx = BuildContext(sim, robot, SimpleNamespace(root=tmp_path), output)
    def build(spec, ctx):
        if bind_events is not None:
            assert bind_events == ["sim-bind", "robot-bind"]
            bind_events.append("build")
        for _ in range(initialization_ticks):
            sim.step(render=False)
        return scene
    task = SimpleNamespace(build=build)
    inputs = read_inputs(intake)
    archived = [("spec.json", intake.spec), ("probe-config.json", intake.config)]
    if not inputs.is_source_clean:
        archived.extend([("layout.json", intake.layout), ("probe-catalog.json", intake.catalog)])
    for path, value in archived:
        write(output/path, value)
    result = run_probe_loop(intake.spec, task, ctx, inputs, wall_s=120, native=FakeNative)
    return output, result, sim, scene


def success_markers(output):
    for filename, stage, details in [("scene-probe-returned.json", "scene_probe_returned", {"infra_valid": True}),
            ("worker-returned.json", "worker_returned", {"exit_code": 0}), ("cli-returned.json", "cli_returned", {"exit_code": 0})]:
        write(output/filename, {"schema": "mind2act_eval.worker_lifecycle.v1", "stage": stage, "pid": 1234, **details})


def test_loop_ignores_terminal_runs_exactly_600_live_ticks_and_keeps_source(intake, tmp_path):
    output, result, sim, scene = run_fake(intake, tmp_path)
    assert result.infra_valid and result.score is None and not result.model_tested
    assert result.ticks == 600 and scene.steps == list(range(1, 601))
    assert scene.before_ticks == scene.steps
    assert sim.closed and scene.closed
    receipt = json.loads((output/"scene-probe-receipt.json").read_text())
    assert receipt["exit_clock"]["tick"] - receipt["entry_clock"]["tick"] == 600
    assert receipt["source_spec_sha256"] == spec_digest(intake.spec)
    assert (output/"before-head.png").read_bytes().startswith(b"\x89PNG")
    assert not (output/"score.json").exists()


def test_real_e2_before_hook_keeps_source_tick_zero_at_first_probe_step(intake, tmp_path, monkeypatch):
    from mind2act_eval.tasks.E2.scene import ArithmeticScene
    source_ticks=[]
    display=SimpleNamespace(_display=source_ticks.append)
    original_before=FakeScene.before_step
    def before(scene,tick):
        original_before(scene,tick)
        ArithmeticScene.before_step(display,tick)
    monkeypatch.setattr(FakeScene,'before_step',before)
    _,result,_,scene=run_fake(intake,tmp_path)
    assert result.infra_valid
    assert scene.before_ticks == scene.steps == list(range(1,601))
    assert source_ticks == list(range(600))


def test_actual_clock_slip_is_infra_failure_and_cleans_up(intake, tmp_path):
    output, result, sim, scene = run_fake(intake, tmp_path, slip=True)
    assert not result.infra_valid and result.score is None and result.ticks == 1
    assert "actual World clock" in result.error and sim.closed and scene.closed


def test_source_settling_precedes_clutter_insertion_and_is_not_diagnostic_ticks(intake, tmp_path, monkeypatch):
    events=[]
    original_spawn=FakeNative.spawn_clutter
    def spawn(stage, inputs, root):
        assert events == ['sim-bind', 'robot-bind', 'build']
        events.append('clutter')
        return original_spawn(stage, inputs, root)
    monkeypatch.setattr(FakeNative, 'spawn_clutter', spawn)
    output, result, sim, scene = run_fake(intake, tmp_path, bind_events=events, initialization_ticks=2400)
    assert result.infra_valid and result.ticks == 600
    assert events == ['sim-bind', 'robot-bind', 'build', 'clutter']
    receipt=json.loads((output/'scene-probe-receipt.json').read_text())
    assert receipt['entry_clock']['tick'] == 2417
    assert receipt['exit_clock']['tick'] == 3017
    assert receipt['provenance']['source_initialization_exit_clock'] == receipt['entry_clock']


def test_projection_warmup_cannot_hide_a_physics_step_before_entry(intake, tmp_path, monkeypatch):
    original_ready=FakeSim.ready
    def ready(sim):
        original_ready(sim)
        sim.step(render=False)
    monkeypatch.setattr(FakeSim, 'ready', ready)
    _, result, sim, scene = run_fake(intake, tmp_path)
    assert not result.infra_valid and result.ticks == 0
    assert 'actual World clock' in result.error and sim.closed and scene.closed


def test_unregistered_actor_cannot_pass_from_static_usd_pose_fallback():
    from mind2act_eval.scene_probe_native import live_body_state
    body=SimpleNamespace(is_physics_handle_valid=lambda:False,
                         get_world_pose=lambda:pytest.fail('must reject before USD fallback'))
    with pytest.raises(RuntimeError,match='USD-only poses'):
        live_body_state(body)
    body=SimpleNamespace(is_physics_handle_valid=lambda:True,
                         get_world_pose=lambda:([0.,0.,.8],[1.,0.,0.,0.]),
                         get_linear_velocity=lambda:[.01,0.,0.], get_angular_velocity=lambda:None)
    with pytest.raises(RuntimeError,match='readback is missing'):
        live_body_state(body)
    body.get_angular_velocity=lambda:np.zeros(3,dtype=np.float32)
    body.get_linear_velocity=lambda:np.zeros(3,dtype=np.float32)
    body.get_world_pose=lambda:(np.array([0,0,.8],np.float32),np.array([1,0,0,0],np.float32))
    assert live_body_state(body)[0] == pytest.approx((0.,0.,.8))


def test_probe_binds_exact_source_spec_to_both_services_before_build(intake, tmp_path):
    events = []
    _, result, _, _ = run_fake(intake, tmp_path, bind_events=events)
    assert result.infra_valid and events == ["sim-bind", "robot-bind", "build"]


def test_probe_source_binding_failure_never_builds_task(intake, tmp_path):
    events = []
    _, result, sim, scene = run_fake(intake, tmp_path, bind_events=events, bind_error=True)
    assert not result.infra_valid and result.ticks == 0
    assert events == ["sim-bind", "robot-bind"] and not scene.steps and not scene.closed
    assert sim.closed and sim._world is None


def test_actual_sim_source_binding_rejects_late_replacement_and_wrong_output(intake, tmp_path):
    from mind2act_eval.sim import IsaacSimulator
    sim = object.__new__(IsaacSimulator)
    sim._output_dir, sim._world, sim._clock = tmp_path.resolve(), object(), intake.spec.clock
    with pytest.raises(RuntimeError, match="before the first World"):
        sim.bind_task_instance(intake.spec, tmp_path)
    assert sim.active_task_spec is None
    sim._world = None
    sim.bind_task_instance(intake.spec, tmp_path)
    assert sim.active_task_spec is intake.spec
    with pytest.raises(ValueError, match="cannot be replaced"):
        sim.bind_task_instance(replace(intake.spec), tmp_path)
    with pytest.raises(ValueError, match="owned output"):
        sim.bind_task_instance(intake.spec, tmp_path/"other")


def test_explicit_consumer_and_probe_inputs_cannot_be_silently_ignored(intake):
    args = args_for(intake)
    args.consumer = "expert_replay"
    with pytest.raises(ValueError, match="unsupported explicit consumer"):
        _job(args)
    with pytest.raises(ValueError, match="unsupported explicit consumer"):
        _worker(args)
    args.consumer = None
    with pytest.raises(ValueError, match="requires explicit"):
        _worker(args)


def test_unscored_diagnostic_seals_without_score_or_task_evidence(intake, tmp_path):
    output, result, _, _ = run_fake(intake, tmp_path)
    success_markers(output)
    with pytest.raises(ValueError, match="trusted expected consumer"):
        seal_worker_outputs(output)
    seal_worker_outputs(output, expected_consumer=ConsumerKind.SCENE_PROBE)
    with pytest.raises(ValueError, match="trusted parent job"):
        validate_worker_seal(output, expected_consumer=ConsumerKind.MODEL)
    actual, seal = validate_worker_seal(output, durable=True, expected_consumer=ConsumerKind.SCENE_PROBE)
    assert actual == result and "score.json" not in {r["path"] for r in seal["files"]}
    (output/"after-head.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed|digest mismatch"):
        validate_worker_seal(output, expected_consumer=ConsumerKind.SCENE_PROBE)


@pytest.mark.parametrize("defect", ["missing_receipt", "external_frame", "missing_samples"])
def test_probe_seal_refuses_incomplete_or_external_evidence(intake, tmp_path, defect):
    output, _, _, _ = run_fake(intake, tmp_path)
    success_markers(output)
    if defect == "missing_receipt":
        (output/"scene-probe-receipt.json").unlink()
    elif defect == "missing_samples":
        (output/"scene-probe-samples.jsonl").write_text("")
    else:
        (output/"after-head.png").unlink()
        outside = tmp_path/"external.png"
        write_png(outside, np.zeros((4, 6, 3), dtype=np.uint8))
        (output/"after-head.png").symlink_to(outside)
    with pytest.raises((ValueError, FileNotFoundError)):
        seal_worker_outputs(output, expected_consumer=ConsumerKind.SCENE_PROBE)


def test_supervisor_binds_unscored_seal_to_parent_consumer(intake, tmp_path, monkeypatch):
    """Synthetic child exercises parent seal admission; it never starts Kit."""
    import sys
    import mind2act_eval.batch as batch
    from mind2act_eval.batch import EpisodeJob, run_attempt
    fixture, _, _, _ = run_fake(intake, tmp_path)
    script = tmp_path/"cpu-seal-child.py"
    source = str(Path(__file__).parents[1]/".")
    script.write_text("import sys, pathlib, shutil, json, os\n"
        + "sys.path.insert(0, " + repr(source) + ")\n"
        + "from mind2act_eval.sealing import seal_worker_outputs\n"
        + "root=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
        + "fixture=pathlib.Path(" + repr(str(fixture)) + ")\n"
        + "for f in fixture.iterdir():\n"
        + " if f.name != 'spec.json': shutil.copy2(f, root/f.name)\n"
        + "for name,stage,details in [('scene-probe-returned','scene_probe_returned',{'infra_valid':True}),('worker-returned','worker_returned',{'exit_code':0}),('cli-returned','cli_returned',{'exit_code':0})]:\n"
        + " (root/(name+'.json')).write_text(json.dumps(dict(schema='mind2act_eval.worker_lifecycle.v1',stage=stage,pid=os.getpid(),**details)))\n"
        + "seal_worker_outputs(root, expected_consumer='scene_probe')\n")
    monkeypatch.setattr(batch, "user_rss_bytes", lambda: 1)
    # These synthetic writers exit immediately after sealing; no signals or
    # native teardown are needed to test the parent's consumer admission.
    monkeypatch.setattr(batch, "_stop_owned_group", lambda child, **kwargs: child.wait(timeout=5))
    job = EpisodeJob(intake.spec, (sys.executable, str(script)), ConsumerKind.SCENE_PROBE, wall_s=20, retries=0)
    result = run_attempt(job, tmp_path/"valid-probe", lock_root=tmp_path/"locks", poll_s=.01)
    assert result.infra_valid and result.score is None
    forged = replace(job, consumer_kind=ConsumerKind.MODEL)
    result = run_attempt(forged, tmp_path/"forged-type", lock_root=tmp_path/"locks", poll_s=.01)
    assert not result.infra_valid and result.score is None
    assert "trusted parent job" in result.error


def test_scene_probe_cannot_publish_fake_model_score():
    with pytest.raises(ValueError, match="cannot publish scores"):
        Result("F2", "id", 0., Termination.COMPLETED, True, ConsumerKind.SCENE_PROBE, False)
    with pytest.raises(ValueError, match="valid episode requires numeric score"):
        Result("F2", "id", None, Termination.COMPLETED, True, ConsumerKind.MODEL, True)


def test_draft_v2_drift_gate_but_contact_penetration_only_recorded():
    samples = [{"x": {"position": [.041*i/600, 0, .8], "orientation": [1, 0, 0, 0], "bottom_z": .768,
                       "floor_contact": False, "penetration_m": .1, "contact_gap_m": .2}} for i in range(601)]
    checks = stability(samples, support_z=.768)
    assert checks["bodies"][0]["failures"] == ["total_shift"]
    assert "penetration" in checks["diagnostic_only"]


def test_usd_loader_keeps_task_joint_and_applies_fixed_clutter_source_defaults(intake):
    from pxr import Usd, UsdGeom, UsdPhysics
    from mind2act_eval.scene_probe_native import physics_modes, spawn_clutter
    root = intake.path.parent
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/World/Physics")
    task = UsdGeom.Xform.Define(stage, "/World/Task/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(task).CreateRigidBodyEnabledAttr(True)
    spring = UsdPhysics.PrismaticJoint.Define(stage, "/World/Task/Spring")
    spring.CreateJointEnabledAttr(True)
    requested = [None, .37, 0., .85, -1.]
    for i, pin in enumerate(intake.catalog["clutter"].values()):
        if requested[i] is not None:
            intake.layout["Rigid"]["block"][i]["physics"]["mass"] = requested[i]
        path = root/pin["path"]
        path.parent.mkdir(exist_ok=True)
        asset = Usd.Stage.CreateNew(str(path.with_suffix(".usda")))
        obj = UsdGeom.Xform.Define(asset, "/Object").GetPrim()
        asset.SetDefaultPrim(obj)
        if i != 0:
            body = UsdPhysics.RigidBodyAPI.Apply(obj)
            body.CreateRigidBodyEnabledAttr(False)
            body.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(obj).CreateMassAttr(.37)
        collider = UsdGeom.Cube.Define(asset, "/Object/Collider")
        collider.CreateSizeAttr(.04)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        asset.GetRootLayer().Save()
        pin["path"] = str(path.with_suffix(".usda").relative_to(root))
        pin["sha256"] = sha256(path.with_suffix(".usda"))
    repin(intake)
    inputs = read_inputs(intake)
    before = physics_modes(stage)
    receipt = spawn_clutter(stage, inputs, root)
    assert physics_modes(stage) == before
    assert spring.GetJointEnabledAttr().Get() is True
    assert len(receipt) == 5
    for row, expected in zip(receipt, [.5, .37, .05, .5, .05]):
        prim = stage.GetPrimAtPath(row["body_path"])
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(expected)
        assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is True
        assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is False
        assert row['collision_geometry_changed'] is False
    assert receipt[0]['added_clutter_rigid_body_api'] is True


def test_probe_json_rejects_duplicate_keys_even_with_matching_digest(intake):
    intake.path.write_text('{"schema":"x","schema":"y"}')
    intake.digest = sha256(intake.path)
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_inputs(intake)


@pytest.mark.parametrize("kind", ["Dome", "Rect", "Distant", "Sphere", "Cylinder", "Disk"])
def test_appearance_matches_explicit_baseline_light_and_mdl_uv_contract(intake, kind):
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from mind2act_eval.scene_probe_native import apply_appearance, physics_modes
    root = intake.path.parent
    material_file = root/"wood.mdl"
    material_file.write_text("mdl 1.0; // CPU shader-binding fixture, not rendered\n")
    intake.catalog["materials"]["wood"]["sha256"] = sha256(material_file)
    intake.layout["Appearance"]["material"]["sha256"] = sha256(material_file)
    intake.layout["ground"]["sha256"] = sha256(material_file)
    intake.layout["Appearance"]["light"]["type"] = kind
    repin(intake)
    inputs = read_inputs(intake)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    for path in ("/World/Worktable/Top", "/World/Floor"):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    table = stage.GetPrimAtPath("/World/Worktable/Top")
    authored_uv = [(0, 0), (.2, 0), (.2, .2), (0, .2)]
    UsdGeom.PrimvarsAPI(table).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                         UsdGeom.Tokens.faceVarying).Set(authored_uv)
    contact = UsdShade.Material.Define(stage, "/World/Contact")
    UsdPhysics.MaterialAPI.Apply(contact.GetPrim()).CreateStaticFrictionAttr(.4)
    UsdShade.MaterialBindingAPI.Apply(table).Bind(contact)
    key = UsdLux.RectLight.Define(stage, "/World/Light")
    key.CreateWidthAttr(2.); key.CreateHeightAttr(2.)
    key.CreateIntensityAttr(1800.); key.CreateExposureAttr(2.)
    dome = UsdLux.DomeLight.Define(stage, "/World/Studio")
    dome.CreateIntensityAttr(700.)
    dome.CreateTextureFileAttr(Sdf.AssetPath("existing-hdr.hdr"))
    protected = physics_modes(stage)
    appearance = apply_appearance(stage, inputs, root)
    assert physics_modes(stage) == protected
    assert np.asarray(UsdGeom.PrimvarsAPI(table).GetPrimvar("st").Get()) == pytest.approx(np.asarray(authored_uv))
    assert np.asarray(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Floor")).GetPrimvar("st").Get()) == pytest.approx(np.asarray(
        [(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert UsdShade.MaterialBindingAPI(table).ComputeBoundMaterial(materialPurpose="physics")[0].GetPath() == contact.GetPath()
    assert dome.GetTextureFileAttr().Get().path == "existing-hdr.hdr"
    active = appearance[-1]
    if kind == "Dome":
        assert active["target"] == "/World/Studio" and active["existing_hdr_retained"]
        assert stage.GetPrimAtPath("/World/Light").IsA(UsdLux.RectLight)
        assert key.GetIntensityAttr().Get() == 1800.
    else:
        assert active["target"] == "/World/Light"
        assert dome.GetIntensityAttr().Get() == 700.
        assert UsdLux.LightAPI(stage.GetPrimAtPath("/World/Light")).GetExposureAttr().Get() == 2.
        if kind == "Distant":
            assert UsdLux.DistantLight(stage.GetPrimAtPath("/World/Light")).GetAngleAttr().Get() == 1.
    for surface in ("Table", "Floor"):
        material = UsdShade.Material(stage.GetPrimAtPath("/World/SceneProbe/"+surface+"Material"))
        assert all(output.GetConnectedSource() for output in (material.GetSurfaceOutput("mdl"),
                    material.GetVolumeOutput("mdl"), material.GetDisplacementOutput("mdl")))


def clean_preview(intake):
    intake.spec = replace(intake.spec, task_id="F1")
    intake.config = dict(schema="mind2act_eval.scene_probe.v1", projection="source-clean-preview-v1",
                         task_id=intake.spec.task_id, difficulty=intake.spec.difficulty,
                         source_spec_sha256=spec_digest(intake.spec))
    intake.digest = write(intake.path, intake.config)
    return intake


def test_source_clean_preview_has_no_random_inputs_and_uses_real_cli(intake):
    intake = clean_preview(intake)
    inputs = read_inputs(intake)
    assert inputs.is_source_clean and inputs.layout is None and inputs.catalog is None and not inputs.clutter
    assert _job(args_for(intake)).consumer_kind is ConsumerKind.SCENE_PROBE


@pytest.mark.parametrize('field,value', [('source_spec_sha256', '0' * 64), ('task_id', 'F2'),
    ('difficulty', 'hard'), ('layout', {}), ('catalog', {})])
def test_source_clean_preview_rejects_source_substitution_or_fake_random_payload(intake, field, value):
    intake = clean_preview(intake)
    intake.config[field] = value
    intake.digest = write(intake.path, intake.config)
    with pytest.raises(ValueError, match='unchanged original clean source'):
        read_inputs(intake)


def test_source_clean_preview_runs_600_without_projection_actions_or_score_and_seals(intake, tmp_path, monkeypatch):
    intake = clean_preview(intake)
    monkeypatch.setattr(FakeNative, 'spawn_clutter', lambda *a: pytest.fail('clean preview added clutter'))
    monkeypatch.setattr(FakeNative, 'apply_appearance', lambda *a: pytest.fail('clean preview changed appearance'))
    monkeypatch.setattr(FakeNative, 'NativeMonitor', lambda *a, **kw: pytest.fail('clean preview fabricated clutter'))
    output, result, sim, scene = run_fake(intake, tmp_path)
    assert result.infra_valid and result.ticks == 600 and result.score is None and not result.model_tested
    assert result.consumer_kind is ConsumerKind.SCENE_PROBE and scene.steps == list(range(1, 601))
    assert not (output/'layout.json').exists() and not (output/'probe-catalog.json').exists()
    receipt = json.loads((output/'scene-probe-receipt.json').read_text())
    assert receipt['layout_sha256'] is None and receipt['logical_slot'] is None
    assert receipt['provenance']['clutter'] == [] and receipt['provenance']['appearance'] == []
    assert receipt['stability']['projection_applied'] is False
    success_markers(output)
    seal_worker_outputs(output, expected_consumer=ConsumerKind.SCENE_PROBE)
    validate_worker_seal(output, durable=True, expected_consumer=ConsumerKind.SCENE_PROBE)
    with pytest.raises(ValueError, match='trusted parent job'):
        validate_worker_seal(output, expected_consumer=ConsumerKind.EXPERT_REPLAY)
    assert not (output/'score.json').exists()


def test_source_clean_preview_preserves_typed_asset_failure_without_scoring(intake, tmp_path, monkeypatch):
    intake = clean_preview(intake)
    class MissingAssets(RuntimeError):
        invalid_reason = 'asset_unresolved'
        missing_paths = ['source-geometry.usda']
    monkeypatch.setattr(FakeSim, 'ready', lambda *a: (_ for _ in ()).throw(MissingAssets('missing source')))
    _, result, _, _ = run_fake(intake, tmp_path)
    assert not result.infra_valid and result.ticks == 0 and result.score is None
    assert result.task_metadata['invalid_reason'] == 'asset_unresolved'
    assert result.task_metadata['missing_paths'] == ('source-geometry.usda',)


def test_source_clean_worker_supports_builder_opened_scene_without_station(intake, tmp_path, monkeypatch):
    import mind2act_eval.worker as worker
    import mind2act_eval.sim as sim_module
    import mind2act_eval.robot as robot_module
    import mind2act_eval.tasks.catalog as catalog
    from mind2act_eval.scene_probe import SOURCE_CLEAN_PREVIEW
    intake.spec = replace(intake.spec, task_id="P1", assets={"scene":"scene.usda", "x5":"x5.usd"})
    intake.config = dict(schema="mind2act_eval.scene_probe.v1", projection=SOURCE_CLEAN_PREVIEW,
        task_id="P1", difficulty="easy", source_spec_sha256=spec_digest(intake.spec))
    intake.digest = write(intake.path, intake.config)
    (tmp_path/'x5.usd').write_text('CPU startup fixture; not loaded by native SDK')
    args = args_for(intake)
    args.output = str(tmp_path/'worker-output'); Path(args.output).mkdir()
    class StartupSim:
        def __init__(self, *a, **kw): pass
        def rgb(self): pytest.fail('startup must not request model/camera data')
        def open_stage(self, *a): pytest.fail('P1 builder owns its scene')
    robot = object(); task = object(); sentinel = object()
    monkeypatch.setattr(sim_module, 'IsaacSimulator', StartupSim)
    monkeypatch.setattr(robot_module, 'X5Robot', lambda **kw: robot)
    monkeypatch.setattr(catalog, 'load_task', lambda task_id: task)
    def dispatch(spec, selected, ctx, inputs, **kw):
        assert spec == intake.spec and selected is task and ctx.robot is robot
        assert inputs.is_source_clean and ctx.assets.resolve('scene') == tmp_path/'scene.usda'
        return sentinel
    (tmp_path/'scene.usda').write_text('source scene remains builder-owned')
    monkeypatch.setattr(worker, 'run_probe_loop', dispatch)
    assert worker.run_scene_probe(args) is sentinel
    assert (Path(args.output)/'probe-config.json').is_file()
    assert not (Path(args.output)/'layout.json').exists()
    assert not (Path(args.output)/'probe-catalog.json').exists()
