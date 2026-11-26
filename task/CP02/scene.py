"""CP02 scene: unique build(spec, ctx) shared by XPolicyLab and RoboProbe.

Sources (cp02-hard-support-v3; Easy/Medium identical flow in cp02-em-qualified-batch):
cases_unified/CP02/world.py (build, measure, cover_force_telemetry) and timeline.py
(NativeTimeline). Hooks follow docs/runtime.md: before_step(tick) -> robot.apply ->
sim.step -> after_step(tick) -> evidence(tick). The source applied the measured
schedule after each tick (NativeTimeline.after_tick), which is after_step here;
evidence() only reads the accumulated gates.
"""
from __future__ import annotations

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation
from mind2act_eval.tasks.CP02.evidence import MeasuredTask, quat_matrix, roll_deg

PHYSICS_HZ = 120
WARMUP_TICKS = 120


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def cover_force_telemetry(view):
    """Actual filtered PhysX tensor force; diagnostic only, never used for support."""
    if view is None:
        return dict(schema="cp02-cover-force-telemetry-v1", valid=False, reason="No sensor")
    try:
        def array(value):
            if value is None:
                raise ValueError("Sensor returned no measurement")
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=float)
            if value.ndim < 2 or value.shape[-1] != 3 or value.size == 0 or not np.isfinite(value).all():
                raise ValueError("Nonfinite or malformed force data")
            return value
        net = array(view.get_net_contact_forces(dt=1/120))
        cake = array(view.get_contact_force_matrix(dt=1/120))
        return dict(schema="cp02-cover-force-telemetry-v1", valid=True, physics_dt_s=1/120, units="N",
                    net_contact_force_N=net.reshape(-1, 3).sum(axis=0).tolist(),
                    cake_contact_force_N=cake.reshape(-1, 3).sum(axis=0).tolist(),
                    net_shape=list(net.shape), filter_shape=list(cake.shape),
                    supported_flag_overridden=False)
    except (ValueError, RuntimeError, AttributeError, TypeError) as error:
        return dict(schema="cp02-cover-force-telemetry-v1", valid=False,
                    error_type=type(error).__name__, message=str(error), supported_flag_overridden=False)


class CP02Scene:
    def __init__(self, spec: InstanceSpec, ctx: BuildContext, ns: dict, manifest: dict, config: dict):
        self._spec, self._ctx, self._ns = spec, ctx, ns
        self._manifest, self._config = manifest, config
        self._contacts = ctx.sim.contacts
        self._instruction = public_instruction(spec)
        self._error = None
        self._phase = "observation"
        self._rotor_omega_z = 0.0
        self._warm_up()
        from pxr import Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        self._local_centers = [list(cache.ComputeUntransformedBound(
            ns["stage"].GetPrimAtPath(p)).ComputeAlignedRange().GetMidpoint()) for p in ns["paths"]]
        self._task = MeasuredTask(manifest["spec"], config, self._measure())
        ns["speed_attr"].Set(self._task.timeline.speed)

    def _warm_up(self):
        """attach(warmup=120): settle the held rest pose, then forget warm-up contacts."""
        for _ in range(WARMUP_TICKS):
            self._ctx.sim.step(render=False)
        self._contacts.clear()

    # -- source world.measure ------------------------------------------------
    def _measure(self) -> dict:
        ns, touch = self._ns, self._contacts.touch

        def body(value):
            p, q = value.get_world_pose()
            return dict(position=np.asarray(p).tolist(), quaternion_wxyz=np.asarray(q).tolist())

        def contacts(path):
            return dict(bilateral=self._contacts.bilateral(path), touch_robot=touch(path, robot_only=True),
                        supported=self._contacts.supported(path, other=ns["cake_path"]))

        rotor = dict(body(ns["rotor"]), angular_velocity_rad_s=ns["rotor"].get_angular_velocity().tolist())
        props, V = ns["props"], ns["V"]
        lid_angle = -roll_deg(props["Lid"].get_world_pose()[1])
        buttons = {name: dict(body(props[name + "Button"]), reference_z_m=float(V["buttons"][name][2] + .015),
                              robot_contact=touch("/World/CP02V1/" + name + "Button", robot_only=True))
                   for name in V["buttons"] if name + "Button" in props}

        def pressed(name):
            row = buttons.get(name)
            return bool(row and row["reference_z_m"] - row["position"][2] > .0025 and row["robot_contact"])

        rows = [dict(body(value), **contacts(path), identity=item["identity"],
                     local_center_m=center, held_tilt_deg=self._contacts.held_tilt(path))
                for value, path, item, center in zip(ns["bodies"], ns["paths"],
                    self._manifest["spec"]["items"], self._local_centers)]
        raw = dict(rotor=rotor, bodies=rows,
                   lid=dict(body(props["Lid"]), robot_contact=touch("/World/CP02V1/Lid", robot_only=True)),
                   buttons=buttons, lid_angle_deg=lid_angle,
                   lid_contact=touch("/World/CP02V1/Lid", robot_only=True),
                   start_pressed=pressed("Start"), stop_pressed=pressed("Stop"))
        if self._manifest["spec"]["difficulty"] == "hard":
            path = "/World/CP02V1/Cover"
            from pxr import Gf, Usd, UsdGeom
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            # Each authored wall is a unit cube, outward local +X. Its
            # inner face is local x=-0.5, including authored scale/rotation.
            cake_center = np.asarray(rotor['position']) + quat_matrix(rotor['quaternion_wxyz'])@np.array([0,0,.0805/2])
            faces = [ns['stage'].GetPrimAtPath(path+f'/Collision/Wall{i}') for i in range(48)]
            if not all(faces):
                raise ValueError('CP02 authored cover cavity geometry missing')
            contains = all(cache.GetLocalToWorldTransform(wall).GetInverse().Transform(
                Gf.Vec3d(*cake_center))[0] <= -.5 for wall in faces)
            raw.update(cover_contains_cake_center=bool(contains),
                       cover=dict(body(props["Cover"]), **contacts(path),
                                  touches_food=any(touch(path, other=food) for food in ns["paths"])),
                       cover_source_z_m=V["cover_source"][2], cover_target_world_m=list(V["cover_target"]),
                       cover_contact_telemetry=cover_force_telemetry(ns["cover_contact_view"]))
        return raw

    # -- Scene protocol ------------------------------------------------------
    def before_step(self, tick: int) -> None:
        return None

    def after_step(self, tick: int) -> None:
        ns = self._ns
        try:
            raw = self._measure()
            self._rotor_omega_z = float(raw["rotor"]["angular_velocity_rad_s"][2])
            event = self._task.update(tick, raw)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self._error = f"CP02 measured trace invalid: {exc}"
            raise
        ns["speed_attr"].Set(event["speed_deg_s"])
        self._phase = event["phase"]
        if event["cue_transition"]:
            for item, path in zip(self._manifest["spec"]["items"], ns["paths"]):
                ns["UsdGeom"].Imageable(ns["stage"].GetPrimAtPath(
                    ns["cake_path"] + "/Demonstration/" + item["identity"])).MakeInvisible()
                ns["UsdGeom"].Imageable(ns["stage"].GetPrimAtPath(path)).MakeVisible()
            for _ in range(4):
                self._ctx.world.render()

    def evidence(self, tick: int) -> NativeEvidence:
        grade = self._task.grade()
        from mind2act_eval.tasks.CP02.validation_inputs import evidence_fields
        return NativeEvidence("CP02", self._spec.instance_id, tick, dict(
            grade, difficulty=self._spec.difficulty, stop_tick=self._task.stop_tick,
            rotor_angular_velocity_z_rad_s=self._rotor_omega_z, phase=self._phase, **evidence_fields(self._spec)))

    def observe(self, tick: int) -> Observation:
        state = self._ctx.robot.observe()
        return Observation(
            tick=tick, sim_time_s=tick / self._spec.clock.physics_hz,
            left_arm=state["left_arm"], right_arm=state["right_arm"],
            left_gripper=state["left_gripper"], right_gripper=state["right_gripper"],
            rgb=state.get("rgb", {}), instruction=self._instruction,
            protocol_version=self._spec.protocol_version)

    def health(self) -> Health:
        return Health(False, self._error) if self._error else Health(True, "")

    def _completion_grade(self):
        task = getattr(self, "_task", None)
        return task.grade() if task is not None else None

    def is_complete(self) -> bool:
        grade = self._completion_grade()
        return bool(grade and grade["physical_task_success"] and grade["task_process_pass"])

    def _credit_ceiling(self):
        task = getattr(self, "_task", None)
        raw = getattr(task, "latest", None)
        spec = getattr(task, "spec", None)
        if not raw or not spec or "assembly_z_m" not in spec:
            return None
        cutoff = spec["assembly_z_m"] - .40
        rows = raw["bodies"]
        dropped = sum(row["position"][2] < cutoff and not row["touch_robot"] for row in rows)
        cover = raw.get("cover")
        cover_dropped = cover and cover["position"][2] < cutoff and not cover["touch_robot"]
        # The hard-mode cover gates full success; only macarons contribute core credit.
        return len(rows) - dropped if dropped or cover_dropped else None

    def irreversible_failure(self) -> str | None:
        if self._credit_ceiling() == 0:
            grade = self._task.grade()
            if (not (grade["physical_task_success"] and grade["task_process_pass"])
                    and grade["credited_macarons"] <= grade["core_wrong_or_extra_items"]):
                return "macarons_fell_off_station"
        return None

    def is_success(self) -> bool:
        grade = self._completion_grade()
        if grade is None:
            return False
        if grade["physical_task_success"] and grade["task_process_pass"]:
            return True
        ceiling = self._credit_ceiling()
        if ceiling is None or ceiling == 0:
            return False
        return grade["credited_macarons"] == ceiling and grade["core_wrong_or_extra_items"] == 0

    def close(self) -> None:
        self._ns = None


def build(spec: InstanceSpec, ctx: BuildContext) -> CP02Scene:
    if spec.instance_id.startswith('VAL-CP02-') or spec.config.get('layout_split')=='validation':
        from mind2act_eval.tasks.CP02.validation_inputs import validate_assets
        validate_assets(spec,ctx.assets)
    from mind2act_eval.tasks.CP02.mechanism.cake_scene import initialize

    if spec.clock.physics_hz != PHYSICS_HZ:
        raise ValueError("CP02 mechanism is authored at 120 Hz physics")
    native = spec.config["native_inputs"]
    manifest, config = _plain(native["manifest"]), _plain(native["config"])
    if (manifest["spec"]["task"] != "CP02" or manifest["spec"]["difficulty"] != spec.difficulty
            or config["difficulty"] != spec.difficulty or manifest["difficulty"] != spec.difficulty):
        raise ValueError("CP02 native payload does not match the instance spec")
    expected = {"easy": 4, "medium": 6, "hard": 6}[spec.difficulty]
    if len(manifest["spec"]["items"]) != expected:
        raise ValueError("CP02 native payload item count differs")
    ns = initialize(manifest, config, ctx)
    return CP02Scene(spec, ctx, ns, manifest, config)
