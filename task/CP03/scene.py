"""CP03 scene: unique build(spec, ctx) shared by XPolicyLab and RoboProbe.

Sources (cp03-em-b-20261028 for Easy/Medium, cp03-hard-b-20261028 for Hard; the scene,
world, quality and selector modules are byte-identical across them): cases_unified/CP03/
world.py (create: rail, selector controls, Start, per-plate stable-placement tracking,
grade), quality_runtime.py (plate tilt windows), evidence.py (Collector.finalize gates and
document-stage tracking) and cases/CP/CP03/runtime/selector_mechanics.py.

Scoring follows FINAL-VERSION.md: gear then Start, measured rack motion,
slot entry while moving, and current supported source placement.
Source plates must be physically picked up and transported by the gripper.
Simultaneous gripper/support contact is not required.
Quality/contact histories remain diagnostic.
Physical source controls and dynamics remain unchanged.

Tick order is preserved: before_step = world `before` (carriage), after_step = control/placement
logic -> selector pawl -> plate quality -> document-stage sampling; evidence() reads results.

Not migrated: expert controller, planner boxes, camera audit and the controller-quality gates
`tracking` / `command_continuity` of the source Collector (they measured the expert's own
command smoothness against the recorded command row, which an environment cannot observe for an
arbitrary consumer). docs/CP03.md lists this decision.
"""
from __future__ import annotations

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation
from mind2act_eval.tasks.CP03.scorer import RULE_VERSION
from mind2act_eval.tasks.CP03.evidence import StablePlacement, dish_target, euler_xyz_deg, quat_matrix
from mind2act_eval.tasks.CP03.mechanism.observable_quality import (
    PlateQuality, plate_tilt_deg, slot_mouth_intersects)

PHYSICS_HZ = 120
INSERTION_MIN_SPEED_M_S = 0.  # Actual rack displacement, with no extra speed threshold.
PARK_JOINT1_DELTA_RAD = -.3  # baseline build(): clear the selector grip from the initial palm pose


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


class CP03Scene:
    def __init__(self, spec: InstanceSpec, ctx: BuildContext, ns: dict, M: dict, C: dict):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self._spec, self._ctx, self._ns, self.M, self.C = spec, ctx, ns, M, C
        self._Gf, self._UsdPhysics = Gf, UsdPhysics
        self._contacts = ctx.sim.contacts
        self._touch = self._contacts.touch
        self._relaxed = C.get("evaluation_protocol") == "physco-task-centric-v2"
        self._instruction = public_instruction(spec)
        self._error = None
        self._tick = 0
        S = self.S = ns["S"]
        self._bodies, self._sources, self._preloaded = ns["bodies"], ns["sources"], ns["preloaded"]
        self._controls, self._paths = ns["controls"], ns["control_paths"]
        self._origins = ns["control_origins"]
        self._base = ns["BASE"]
        self._initial = {p: b.get_world_pose()[0].copy() for p, b in self._bodies.items()}
        self._trackers = {p: StablePlacement(120, .004, require_manipulation=True) for p in self._sources}
        self._assigned, self._failures = {}, set()
        self._terminal_placements = {}
        self._slot_entries = {}
        self._inside_slot = {p:None for p in self._sources}
        self._rack_motion_tick = None
        self._motor = {"started": False, "gear_touched": False, "mode_touched": False,
                       "speed_touched": False, "sent": False,
                       "start_down": False, "start_released": False}
        self._rail = {"x": ns["initial_rail"], "sign": 0, "reversals": 0}
        self._insertion_speeds = {p: [] for p in self._sources}
        self._was_seated = {p: False for p in self._sources}
        self._stage_doc = dict(valid_start=False, gear_qualified=False, gear_consecutive_ticks=0)
        self._start_tick = None
        self._gear_set_tick = None
        self._plate_first = {}
        self._wrong = {p: False for p in self._sources}
        self._duplicates = 0
        self._pair_stats = {}
        self._subscribe_robot_contacts()
        self._selector = None
        if "gear" in self._controls:
            from mind2act_eval.tasks.CP03.mechanism.selector import SelectorMechanics
            self._selector = SelectorMechanics(ns["stage"], self._controls["gear"], self._paths["gear"] + "/ControlJoint")
            ctx.robot.park_arm("Left", (PARK_JOINT1_DELTA_RAD, 0., 0., 0., 0., 0.))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        self._cache = cache
        self._quality = {p: PlateQuality(C, float(self._bodies[p].get_world_pose()[0][2])) for p in self._sources}
        self._local_bounds = {}
        for path in self._sources:
            b = cache.ComputeUntransformedBound(ns["stage"].GetPrimAtPath(path)).ComputeAlignedRange()
            lo, hi = np.asarray(b.GetMin()), np.asarray(b.GetMax())
            self._local_bounds[path] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        self._mouth_paths = {i: [self._base + f"/Carriage/Cradle{i}/Side{k}" for k in (0, 2)] for i in S["empty_slot_indices"]}
        for paths in self._mouth_paths.values():
            for path in paths:
                prim = ns["stage"].GetPrimAtPath(path)
                if not prim or not any(p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
                                       for p in Usd.PrimRange(prim)):
                    raise ValueError("Missing opposing enabled slot collider: " + path)

    # -- native robot contact statistics (source: w.contact_pairs max impulse / min separation)
    def _subscribe_robot_contacts(self):
        import math
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        def receive(headers, data):
            for h in headers:
                pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1)))
                if not any("/Robot" in p for p in pair):
                    continue
                points = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
                impulse = max((math.hypot(*p.impulse) for p in points), default=0.)
                separation = min((float(p.separation) for p in points), default=1.)
                row = self._pair_stats.setdefault(pair, [0., 1.])
                row[0], row[1] = max(row[0], impulse), min(row[1], separation)
        self._robot_subscription = get_physx_simulation_interface().subscribe_contact_report_events(receive)

    def _finger(self, path):
        return any(self._touch(path, other="/World/Robot" + side, link=link)
                   for side in ("Left", "Right") for link in ("link7", "link8"))

    # -- world `before`
    def before_step(self, tick: int) -> None:
        ns = self._ns
        x = ns["rail_x"](tick / PHYSICS_HZ)
        ns["carriage_op"].Set(self._Gf.Vec3d(x, *ns["carriage_base"][1:]))

    # -- world `after` + selector + quality + document stage
    def after_step(self, tick: int) -> None:
        try:
            self._tick = tick
            self._control_and_placement(tick)
            if self._selector is not None:
                self._selector.step()
            self._observe_quality(tick)
            self._sample_document_stage()
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self._error = f"CP03 measured state invalid: {exc}"
            raise

    def _control_and_placement(self, step):
        ns, S, C, relaxed, motor = self._ns, self.S, self.C, self._relaxed, self._motor
        controls, origins, paths, finger, touch = self._controls, self._origins, self._paths, self._finger, self._touch
        UsdPhysics, stage, bodies = self._UsdPhysics, self._ns["stage"], self._bodies
        gear_ok = True
        if "gear" in controls:
            angle = euler_xyz_deg(controls["gear"].get_world_pose()[1])[0]
            motor["gear_touched"] |= finger(paths["gear"])
            gear_ok = bool(abs(angle - [18., 0., -18.][S["mode"] - 1]) <= 2)
            if finger(paths["gear"]) and gear_ok:
                drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(paths["gear"] + "/ControlJoint"), "angular")
                drive.GetTargetPositionAttr().Set([18., 0., -18.][S["mode"] - 1])
                drive.GetStiffnessAttr().Set(3.)
                drive.GetMaxForceAttr().Set(.15)
        if "mode" in controls:
            angle = euler_xyz_deg(controls["mode"].get_world_pose()[1])[2]
            motor["mode_touched"] |= finger(paths["mode"])
            gear_ok &= abs(angle - 30 * (S["mode"] - 1)) <= 10 and motor["mode_touched"]
        if "speed" in controls:
            travel = float(controls["speed"].get_world_pose()[0][1] - origins["speed"][1])
            motor["speed_touched"] |= finger(paths["speed"])
            reached = abs(travel - (S["mode"] - 1) * .035) < .004
            gear_ok &= reached and motor["speed_touched"]
            if reached and finger(paths["speed"]):
                drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(paths["speed"] + "/ControlJoint"), "linear")
                drive.GetTargetPositionAttr().Set((S["mode"] - 1) * .035)
                drive.GetStiffnessAttr().Set(180.)
        selector_released = not any(finger(paths[n]) for n in ("gear", "mode", "speed") if n in controls)
        if "start" in controls:
            depth = float(origins["start"][2] - controls["start"].get_world_pose()[0][2])
            if depth >= .0015:
                fresh_press = not motor["start_down"]
                motor["start_down"] = True
                ready = self._spec.difficulty == "easy" or self._stage_doc["gear_qualified"]
                if fresh_press and not motor["started"] and gear_ok and ready:
                    ns["start_motor"](step / PHYSICS_HZ)
                    motor["started"] = True
                    self._start_tick = step
            elif depth < .0015:
                if motor["started"] and motor["start_down"]:
                    motor["start_released"] = True
                motor["start_down"] = False
        x = ns["rail_x"](step / PHYSICS_HZ)
        velocity = (x - self._rail["x"]) * PHYSICS_HZ
        self._rail["x"] = x
        if abs(velocity) > 0:
            sign = int(np.sign(velocity))
            if self._rail["sign"] and self._rail["sign"] != sign:
                self._rail["reversals"] += 1
            self._rail["sign"] = sign
        if (motor["started"] and self._start_tick is not None and step >= self._start_tick
                and abs(velocity) > INSERTION_MIN_SPEED_M_S and self._rack_motion_tick is None):
            self._rack_motion_tick = step
        failures = self._failures
        if motor["started"] and not gear_ok:
            failures.add("selector_left_requested_gear")
        for p in self._preloaded:
            if np.linalg.norm(bodies[p].get_world_pose()[0] - self._initial[p] - np.array([x - ns["initial_rail"], 0, 0])) >= .025:
                failures.add("preloaded_plate_disturbed")
        self._cache.Clear()
        mouths = {}
        for slot_index, paths in self._mouth_paths.items():
            bounds = []
            for path in paths:
                r = self._cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
                bounds.append((np.asarray(r.GetMin()), np.asarray(r.GetMax())))
            mouths[slot_index] = bounds
        occupied = []
        for p in self._sources:
            pos, q = bodies[p].get_world_pose()
            relative = pos - np.array([x, 0, 0])
            tracker = self._trackers[p]
            bilateral = self._contacts.bilateral(p)
            held = finger(p)
            normal = quat_matrix(q) @ np.array([0., 0., 1.])
            tilt = float(np.degrees(np.arccos(np.clip(abs(normal[0]), 0, 1))))
            elevated = pos[2] > self._initial[p][2] + .04
            if not relaxed and bilateral and elevated and tilt > C.get("transient_peak_limit_deg", C["max_transport_tilt_deg"]):
                failures.add("transport_tilt")
            if not relaxed and not tracker.grasped and not held and np.linalg.norm(pos - self._initial[p]) >= .012:
                failures.add("unpicked_stock_disturbed")
            slot = min(S["empty_slot_indices"], key=lambda i: abs(relative[0] - (i - (S["total_slots"] - 1) / 2) * S["slot_pitch_m"]))
            local = (slot - (S["total_slots"] - 1) / 2) * S["slot_pitch_m"]
            supported = self._contacts.supported(p, other=self._base + f"/Carriage/Cradle{slot}")
            correct = dish_target(relative[0], pos[2], normal[0], local, S["slot_width_m"], ns["carriage_base"][2] + .123)
            # A real gripper must carry the source, without a prescribed lift,
            # bilateral time fraction or release. A real pinch must transport
            # the plate; a single-finger push is not a gripper transfer.
            tracker.grasped |= bool(bilateral)
            tracker.lifted |= bool(bilateral and held and not self._contacts.supported(p)
                                   and np.linalg.norm(pos-self._initial[p]) > 0)
            center = pos+quat_matrix(q)@self._local_bounds[p].mean(axis=0)
            correct = dish_target(center[0]-x, center[2], normal[0], local,
                                  S["slot_width_m"], ns["carriage_base"][2]+.123)
            tracker.update(relative, bilateral=bilateral, elevated=elevated, released=not held, supported=supported, correct=correct)
            # Admission concerns this placement into the slot, independent of
            # whether support starts before or after the gripper releases it.
            # Leaving and re-entering creates a new placement attempt; there is
            # no once-per-episode "first support" veto on legitimate retries.
            points = self._local_bounds[p] @ quat_matrix(q).T + pos
            plate_bounds = (points.min(axis=0), points.max(axis=0))
            in_slot = bool(supported or slot_mouth_intersects(plate_bounds, mouths[slot]))
            if in_slot and self._inside_slot[p] != slot:
                enabled = self._loading_enabled(step)
                self._slot_entries[p] = dict(tick=step, slot=slot, rack_speed_m_s=abs(velocity),
                    loading_enabled=enabled, held=bool(held), supported=bool(supported),
                    gripper_transport=bool(tracker.grasped and tracker.lifted),
                    dynamic=bool(enabled and tracker.grasped and tracker.lifted
                                 and abs(velocity)>INSERTION_MIN_SPEED_M_S))
            self._inside_slot[p] = slot if in_slot else None
            # Retain the old overlap measurement only as a diagnostic.
            if held and supported and correct:
                self._insertion_speeds[p].append(abs(velocity))
            self._terminal_placements[p] = dict(slot=slot, supported=bool(supported),
                released=not held, correct=bool(correct), stable=tracker.complete,
                relative_position_m=relative.tolist(), world_z_m=float(pos[2]), normal_x=float(normal[0]),
                center_world_m=center.tolist(), slot_center_m=local, slot_width_m=S['slot_width_m'], seated_height_m=ns['carriage_base'][2]+.123,
                stable_positions_m=[point.tolist() for point in tracker.positions],
                observed_through_tick=step)
            # Preserve the source wrong-item penalty before Start. Geometry-only
            # terminal tracking must not restore gear credit for an invalid pre-Start placement.
            self._wrong[p] = bool(supported and not held and not (correct and motor["started"]))
            if (tracker.complete and self._slot_entries.get(p,{}).get("dynamic",False)
                    and p not in self._plate_first):
                self._plate_first[p] = step
            if tracker.complete:
                self._assigned[p] = slot
                occupied.append(slot)
            else:
                self._assigned.pop(p, None)
        self._duplicates = len(occupied) - len(set(occupied))
        if self._duplicates:
            failures.add("duplicate_slot_occupancy")
        if "send" in controls and not relaxed:
            depth = float(origins["send"][2] - controls["send"].get_world_pose()[0][2])
            if depth > .0015 and finger(paths["send"]):
                if len(self._assigned) == len(self._sources):
                    motor["sent"] = True
                else:
                    failures.add("send_before_all_plates_stable")

    def _observe_quality(self, tick):
        self._cache.Clear()

        def bounds(path):
            r = self._cache.ComputeWorldBound(self._ns["stage"].GetPrimAtPath(path)).ComputeAlignedRange()
            return np.asarray(r.GetMin()), np.asarray(r.GetMax())
        mouths = {i: [bounds(p) for p in paths] for i, paths in self._mouth_paths.items()}
        for path, q in self._quality.items():
            pos, quat = self._bodies[path].get_world_pose()
            R = quat_matrix(quat)
            points = self._local_bounds[path] @ R.T + pos
            pb = (points.min(axis=0), points.max(axis=0))
            fingers = [any(self._touch(path, other="/World/Robot" + s, link=f) for s in ("Left", "Right")) for f in ("link7", "link8")]
            supported = any(self._touch(path, other=self._base + f"/Carriage/Cradle{i}") for i in mouths)
            intersect = any(slot_mouth_intersects(pb, side) for side in mouths.values())
            q.observe(tick, z=float(pos[2]), tilt_deg=plate_tilt_deg(quat), fingers=fingers,
                      slot_mouth_intersection=intersect, slot_support=supported)

    # -- evidence.Collector.sample: valid Start and 120 consecutive released correct-gear ticks
    def _sample_document_stage(self):
        controls, S = self._controls, self.S
        touches = {n: any(self._touch(self._paths[n], other="/World/Robot" + side, link=link)
                          for side in ("Left", "Right") for link in ("link7", "link8")) for n in controls}
        angle = float(euler_xyz_deg(controls["gear"].get_world_pose()[1])[0]) if "gear" in controls else None
        target = [18., 0., -18.][S["mode"] - 1]
        released = not any(touches.get(k, False) for k in ("gear", "mode", "speed"))
        gear_ok = angle is None or abs(angle-target) <= 2
        if "mode" in controls:
            gear_ok &= abs(euler_xyz_deg(controls["mode"].get_world_pose()[1])[2]-30*(S["mode"]-1))<=10 and self._motor["mode_touched"]
        if "speed" in controls:
            gear_ok &= abs(float(controls["speed"].get_world_pose()[0][1]-self._origins["speed"][1])-(S["mode"]-1)*.035)<.004 and self._motor["speed_touched"]
        doc = self._stage_doc
        doc["valid_start"] = bool(self._motor["started"])
        if not self._motor["started"]:
            doc["gear_consecutive_ticks"] = int(gear_ok)
            doc["gear_qualified"] = bool(gear_ok)
            if not gear_ok:
                self._gear_set_tick = None
            elif self._gear_set_tick is None and any(n in controls for n in ("gear","mode","speed")):
                self._gear_set_tick = self._tick

    def _loading_enabled(self, tick):
        """Valid ordered controls and actual rack motion must precede slot entry."""
        gear = self._spec.difficulty=="easy" or (
            self._stage_doc["gear_qualified"] and self._gear_set_tick is not None
            and self._start_tick is not None and self._gear_set_tick < self._start_tick)
        return bool(gear and self._motor["started"] and self._stage_doc["valid_start"]
                    and self._start_tick is not None and self._start_tick <= tick
                    and self._rack_motion_tick is not None and self._rack_motion_tick <= tick)

    # -- world `grade` + Collector.finalize gates
    def _original_grade(self):
        S, C, relaxed = self.S, self.C, self._relaxed
        dyn = {p: bool(v and min(v) > .0005) for p, v in self._insertion_speeds.items()}
        if relaxed:
            dyn = {p: bool(v and abs(v[0] - C["speed_m_s"]) <= .05 * C["speed_m_s"]) for p, v in self._insertion_speeds.items()}
        elif "loading_bay_half_width_m" in S:
            dyn = {p: bool(dyn[p] and np.allclose(v, C["speed_m_s"], atol=1e-5)) for p, v in self._insertion_speeds.items()}
        moving = all(dyn.values())
        reversal = relaxed or C["difficulty"] != "hard" or self._rail["reversals"] > 0
        assigned, sources = self._assigned, self._sources
        return dict(
            physical_task_success=bool(self._motor["started"] and not self._failures
                                       and len(assigned) == len(sources) and len(set(assigned.values())) == len(sources)
                                       and (relaxed or "send" not in self._controls or self._motor["sent"]) and moving and reversal),
            placements=dict(assigned), failures=sorted(self._failures), started=self._motor["started"],
            dynamic_insertions=dyn, moving_insertions=bool(moving), rail_reversals=self._rail["reversals"],
            required_count=len(sources), score_version="common-cp03-measured-v1")

    def _unexpected_contacts(self):
        out = []
        for pair, (impulse, separation) in self._pair_stats.items():
            if impulse <= .0001 and separation >= -.001:
                continue
            mount = any("/base_link/" in p for p in pair) and any("/Pedestal" in p for p in pair)
            fingers = any("/link7/" in p or "/link8/" in p for p in pair)
            targets = self._sources + list(self._paths.values())
            permitted = fingers and any(any(p == t or p.startswith(t + "/") for p in pair) for t in targets)
            if not mount and not permitted:
                out.append(dict(pair=list(pair), max_impulse_Ns=impulse, min_separation_m=separation))
        return out

    def _facts(self):
        from collections import Counter

        original = self._original_grade()
        quality = {p: q.result() for p, q in self._quality.items()}
        unexpected = self._unexpected_contacts()
        gates = dict(terminal_source_placements=len(self._assigned)==len(self._sources),
                     unique_empty_slots=len(set(self._assigned.values()))==len(self._sources))
        slots = Counter(original["placements"].values())
        credited = [s for s, slot in original["placements"].items()
                    if s in self._sources and slots[slot] == 1
                    and self._slot_entries.get(s,{}).get("dynamic",False)
                    and self._slot_entries[s]["slot"]==slot]
        gates["dynamic_source_insertions"] = len(credited)==len(self._sources)
        return original, quality, unexpected, gates, credited

    # -- Scene protocol
    def evidence(self, tick: int) -> NativeEvidence:
        original, quality, unexpected, gates, credited = self._facts()
        doc = self._stage_doc
        level = self._spec.difficulty
        first_plate = min(self._plate_first.values(), default=None)
        started = bool(doc["valid_start"] and self._start_tick is not None
                       and self._rack_motion_tick is not None
                       and (first_plate is None or self._start_tick <= first_plate))
        gear = bool(level!="easy" and doc["gear_qualified"] and self._gear_set_tick is not None
                    and (first_plate is None or self._gear_set_tick <= first_plate)
                    and (self._start_tick is None or self._gear_set_tick < self._start_tick))
        required = len(self._sources) + 1 + (0 if level == "easy" else 1)
        satisfied = len(credited) + int(started) + int(gear)
        wrong = sum(self._wrong.values()) + self._duplicates
        success = bool(all(gates.values()) and started and (level == "easy" or gear))
        from mind2act_eval.tasks.CP03.validation_inputs import evidence_fields
        return NativeEvidence("CP03", self._spec.instance_id, tick, dict(
            difficulty=level, scoring_version=RULE_VERSION, task_success=success,
            terminal_placements=self._terminal_placements, slot_entries=self._slot_entries,
            rack_motion_tick=self._rack_motion_tick, start_tick=self._start_tick, gear_set_tick=self._gear_set_tick,
            required_steps=required, steps_satisfied=satisfied,
            wrong_or_extra_items=wrong, credited_plates=len(credited), credited_sources=credited,
            valid_start=started, gear_qualified=gear if level!='easy' else True,
            gates=gates, diagnostic_process_grade=original, plate_quality=quality,
            unexpected_robot_contacts=unexpected, **evidence_fields(self._spec)))

    def observe(self, tick: int) -> Observation:
        state = self._ctx.robot.observe()
        return Observation(
            tick=tick, sim_time_s=tick / self._spec.clock.physics_hz,
            left_arm=state["left_arm"], right_arm=state["right_arm"],
            left_gripper=state["left_gripper"], right_gripper=state["right_gripper"],
            rgb=state.get("rgb", {}), instruction=self._instruction,
            protocol_version=self._spec.protocol_version)

    def result_metadata(self):
        return dict(scoring_version=RULE_VERSION, terminal_window_ticks=0,
                    insertion_min_speed_m_s=INSERTION_MIN_SPEED_M_S)

    def health(self) -> Health:
        return Health(False, self._error) if self._error else Health(True, "")

    def _completion_values(self):
        tick = getattr(self, "_tick", None)
        if not tick or not getattr(self, "_terminal_placements", None) or getattr(self, "_spec", None) is None:
            return None
        return self.evidence(tick).values

    def is_complete(self) -> bool:
        values = self._completion_values()
        return bool(values and values["task_success"])

    def _credit_ceiling(self):
        placements = getattr(self, "_terminal_placements", None)
        sources = getattr(self, "_sources", None)
        carriage_base = (getattr(self, "_ns", None) or {}).get("carriage_base")
        touch = getattr(self, "_touch", None)
        if not sources or not placements or len(placements) != len(sources) or carriage_base is None or touch is None:
            return None
        cutoff = carriage_base[2] - .40
        dropped = sum(row["world_z_m"] < cutoff and not touch(path, robot_only=True)
                      for path, row in placements.items())
        return len(sources) - dropped if dropped else None

    def irreversible_failure(self) -> str | None:
        if self._credit_ceiling() == 0:
            values = self._completion_values()
            if (values and not values["task_success"]
                    and values["credited_plates"] <= values["wrong_or_extra_items"]):
                return "plates_fell_off_station"
        return None

    def is_success(self) -> bool:
        values = self._completion_values()
        if values is None:
            return False
        if values["task_success"]:
            return True
        ceiling = self._credit_ceiling()
        return (ceiling is not None and ceiling > 0 and values["credited_plates"] == ceiling
                and values["wrong_or_extra_items"] == 0)

    def close(self) -> None:
        self._robot_subscription = None
        self._ns = None


def build(spec: InstanceSpec, ctx: BuildContext) -> CP03Scene:
    if spec.instance_id.startswith('VAL-CP03-') or spec.config.get('layout_split')=='validation':
        from mind2act_eval.tasks.CP03.validation_inputs import validate
        validate(spec)
    from mind2act_eval.tasks.CP03.mechanism.rack_scene import create

    if spec.clock.physics_hz != PHYSICS_HZ:
        raise ValueError("CP03 mechanism is authored at 120 Hz physics")
    native = spec.config["native_inputs"]
    M, C, setup = _plain(native["manifest"]), _plain(native["config"]), _plain(native["setup"])
    if C["difficulty"] != spec.difficulty or M.get("difficulty", spec.difficulty) != spec.difficulty:
        raise ValueError("CP03 native payload does not match the instance spec")
    if M.get("adapter_version") not in (None, "cp03-source-adapter-v4-training-physical-profile"):
        raise ValueError("CP03 obsolete physical profile")
    profile = M["levels"][spec.difficulty]
    if profile["mode"] != {"easy": 1, "medium": 2, "hard": 3}[spec.difficulty]:
        raise ValueError("CP03 selector gear disagrees with difficulty")
    if abs(float(profile["speed"]) - float(C["speed_m_s"])) > 1e-9:
        raise ValueError("CP03 selector speed disagrees with source profile")
    ns = create(M, C, setup, ctx)
    return CP03Scene(spec, ctx, ns, M, C)
