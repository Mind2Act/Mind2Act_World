"""CP01 scene: unique build(spec, ctx) shared by XPolicyLab and RoboProbe.

Source: cp01-mh-single-place-keep-final-camera-v2 cases/CP/CP01/sdk_world.py
(build, measure) and sdk_task.py (Timeline, Collector); Easy uses
cp01-easy-source-camera-v1, which differs only by lacking initial_stack_clearance
(handled by setup["difficulty"] inside mechanism.initial_stack_clearance).

Runner contract (docs/runtime.md): each physics tick runs before_step(tick) ->
robot.apply -> sim.step -> after_step(tick) -> evidence(tick). before_step is the
source Timeline.before_tick (belt speed, reference orders at (tick-1)/120 s),
after_step is Timeline.after_tick + Collector.sample (measure, phase, tracker);
evidence() only reads the accumulated tracker. Baseline warm-up (120 held-rest
ticks, then contact ledger cleared) happens once in build.
"""
from __future__ import annotations

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation
from mind2act_eval.tasks.CP01.evidence import Evidence

PHYSICS_HZ = 120  # the source timeline and 120-tick stability windows are authored at 120 Hz
WARMUP_TICKS = 120


class CP01Scene:
    def __init__(self, spec: InstanceSpec, ctx: BuildContext, ns: dict):
        from pxr import Gf, Usd, UsdGeom

        self._Gf = Gf
        self._ctx = ctx
        self._ns = ns
        self._setup = ns["P"]
        native = spec.config["native_inputs"]
        self._spec = spec
        self._instruction = public_instruction(spec)
        self._evidence = Evidence(self._setup, spec.config["max_food_tilt_deg"])
        self._contacts = ctx.sim.contacts
        self._count = ns["food_count"]
        self._paths = ns["paths"]
        self._bodies = ns["bodies"]
        self._stage = ns["stage"]
        self._phase = "observation"
        self._reference_seen = False
        self._error = None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        self._geometry = []
        for path in self._paths:
            bounds = cache.ComputeUntransformedBound(self._stage.GetPrimAtPath(path)).ComputeAlignedRange()
            self._geometry.append(dict(local_center=list(bounds.GetMidpoint()),
                                       half_extents=(np.asarray(bounds.GetSize()) / 2).tolist()))
        camera = self._stage.GetPrimAtPath(native.get("head_camera", "/World/HeadCamera"))
        self._frustum = UsdGeom.Camera(camera).GetCamera(Usd.TimeCode.Default()).frustum
        self._grade = None
        self._warm_up()

    def _warm_up(self):
        """attach(warmup=120): settle the held rest pose (ctx.sim.contacts already subscribed), then forget warm-up contacts."""
        for _ in range(WARMUP_TICKS):
            self._ctx.sim.step(render=False)
        self._contacts.clear()

    # -- source Timeline.before_tick -------------------------------------
    def _set_timeline(self, tick: int):
        setup, ref, Gf = self._setup, self._setup["reference_batch"], self._Gf
        self._ns["speedattr"].Set(Gf.Vec3f(float(setup["supply_spec"]["belt_speed_m_s"]), 0, 0))
        for j, op in enumerate(self._ns["reference_ops"]):
            op.Set(Gf.Vec3d(ref["start_x"] - j * ref["spacing_m"] + tick / PHYSICS_HZ * ref["speed_m_s"],
                            setup["belt_y_m"], 0.859))

    # -- source sdk_world.measure ----------------------------------------
    def _measure(self) -> dict:
        from pxr import Usd, UsdGeom

        touch, count = self._contacts.touch, self._count
        rows = []
        for i, (body, path, geometry) in enumerate(zip(self._bodies, self._paths, self._geometry)):
            p, q = body.get_world_pose()
            rows.append(dict(
                index=i, position=np.asarray(p).tolist(), quaternion=np.asarray(q).tolist(),
                bilateral=bool(self._contacts.bilateral(path)),
                touch_robot=bool(touch(path, robot_only=True)),
                touch_table=bool(self._contacts.supported(path, other="/World/Worktable")),
                touch_trays=[j for j in range(len(self._setup["orders"])) if self._contacts.supported(path, other=self._paths[count + j])],
                tilt_deg=float(self._contacts.held_tilt(path)), **geometry))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        visible = any(self._frustum.Intersects(cache.ComputeWorldBound(p))
                      for p in self._ns["reference_roots"])
        return dict(bodies=rows, reference_visible=visible)

    # -- Scene protocol ----------------------------------------------------
    def before_step(self, tick: int) -> None:
        self._set_timeline(tick - 1)

    def after_step(self, tick: int) -> None:
        """Sample one physics tick (the tracker needs consecutive ticks)."""
        try:
            measured = self._measure()
            ref = self._setup["reference_batch"]
            self._reference_seen |= bool(measured["reference_visible"])
            if (self._reference_seen and not measured["reference_visible"]
                    and tick / PHYSICS_HZ >= ref["duration_s"]):
                self._phase = "execution"
            measured["phase"] = self._phase
            measured["physics_tick"] = int(tick)
            self._grade = self._evidence.sample(measured)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self._error = f"CP01 measured trace invalid: {exc}"
            raise

    def evidence(self, tick: int) -> NativeEvidence:
        grade = self._grade if self._grade is not None else self._evidence.grade()
        from mind2act_eval.cp01_validation_inputs import evidence_fields
        return NativeEvidence("CP01", self._spec.instance_id, tick,
                              dict(grade, difficulty=self._spec.difficulty, **evidence_fields(self._spec)))

    def observe(self, tick: int) -> Observation:
        state = self._ctx.robot.observe()
        return Observation(
            tick=tick, sim_time_s=tick / self._spec.clock.physics_hz,
            left_arm=state["left_arm"], right_arm=state["right_arm"],
            left_gripper=state["left_gripper"], right_gripper=state["right_gripper"],
            rgb=state.get("rgb", {}), instruction=self._instruction,
            protocol_version=self._spec.protocol_version)

    def health(self) -> Health:
        if self._error:
            return Health(False, self._error)
        return Health(True, "")

    def is_success(self) -> bool:
        return bool(self._grade and self._grade["task_success"])

    is_complete = is_success

    def close(self) -> None:
        # Contact subscription belongs to ctx.sim; nothing task-owned outlives the episode.
        self._ns = None


def build(spec: InstanceSpec, ctx: BuildContext) -> CP01Scene:
    if spec.instance_id.startswith('VAL-CP01-') or spec.config.get('layout_split')=='validation':
        from mind2act_eval.cp01_validation_inputs import validate_assets
        validate_assets(spec,ctx.assets)
    from mind2act_eval.tasks.CP01.mechanism.development_scene import initialize

    if spec.clock.physics_hz != PHYSICS_HZ:
        raise ValueError("CP01 mechanism is authored at 120 Hz physics")
    setup = _plain(spec.config["native_inputs"]["setup"])
    if setup["difficulty"] != spec.difficulty:
        raise ValueError("CP01 setup difficulty does not match the instance spec")
    ns = initialize(setup, ctx)
    ns["P"] = setup
    return CP01Scene(spec, ctx, ns)


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value
