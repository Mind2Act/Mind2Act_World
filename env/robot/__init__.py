"""Single shared X5 owner. Task code may supply mounts, never drive parameters.

Reduced from the pinned official X5 profile initializer; source ledger in
THIRD_PARTY.md. Native contact/material opinions remain asset authored.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from mind2act_eval.contracts import Action, freeze

PROFILE = freeze(json.loads(Path(__file__).with_name("profile.json").read_text()))
ROOTS = ("/World/RobotLeft/Asset", "/World/RobotRight/Asset")


def _get_or_add_xform_op(xformable, name, add):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == name:
            if op.IsInverseOp():
                raise ValueError("wrist camera transform op cannot be inverse: " + name)
            return op
    return add()


def _intrinsic_xyz_quaternion(euler_deg):
    """Return a USD-friendly wxyz quaternion for intrinsic XYZ Euler angles."""
    angles = tuple(math.radians(float(value)) / 2 for value in euler_deg)
    if len(angles) != 3 or not all(math.isfinite(value) for value in angles):
        raise ValueError("intrinsic XYZ orientation needs three finite angles")
    (cx, cy, cz), (sx, sy, sz) = (tuple(math.cos(value) for value in angles),
                                  tuple(math.sin(value) for value in angles))
    # Intrinsic X, then Y, then Z composes qx * qy * qz.
    return (cx*cy*cz - sx*sy*sz,
            sx*cy*cz + cx*sy*sz,
            cx*sy*cz - sx*cy*sz,
            cx*cy*sz + sx*sy*cz)


def normalized_target(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("X5 gripper target must be in [0,1]")
    lower, upper = PROFILE["command"]["target_scale_m"]
    base = lower + (upper - lower) * value
    return (base, base * PROFILE["command"]["mimic_ratio"] + PROFILE["command"]["mimic_offset_m"])


def limited_target(requested_m, actual_base_m):
    if not all(math.isfinite(float(v)) for v in (requested_m, actual_base_m)):
        raise ValueError("nonfinite X5 finger target/state")
    lower, upper = PROFILE["command"]["target_scale_m"]
    cap = (upper - lower) * PROFILE["command"]["target_step_fraction"]
    base = actual_base_m + max(-cap, min(cap, requested_m - actual_base_m))
    return (base, base * PROFILE["command"]["mimic_ratio"] + PROFILE["command"]["mimic_offset_m"])


def normalized_observation(actual_base_m):
    if not math.isfinite(float(actual_base_m)):
        raise ValueError("nonfinite X5 gripper observation")
    lower, upper = PROFILE["command"]["target_scale_m"]
    return max(0., min(1., (actual_base_m - lower) / (upper - lower)))


def base_joint_pose(position_m, euler_deg):
    """Station mount as FixedBase localPos0 / localRot0 (wxyz), extrinsic xyz Euler degrees."""
    position = tuple(float(v) for v in position_m)
    angles = tuple(math.radians(float(v)) / 2 for v in euler_deg)
    if len(position) != 3 or len(angles) != 3 or not all(math.isfinite(v) for v in position + angles):
        raise ValueError("base mount needs finite position (3) and euler_deg (3)")
    (cx, cy, cz), (sx, sy, sz) = (tuple(math.cos(a) for a in angles), tuple(math.sin(a) for a in angles))
    # q = qz * qy * qx (extrinsic x, then y, then z)
    w = cz*cy*cx + sz*sy*sx
    x = cz*cy*sx - sz*sy*cx
    y = cz*sy*cx + sz*cy*sx
    z = sz*cy*cx - cz*sy*sx
    return position, (w, x, y, z)


def verify_asset(path: Path, expected: str):
    """Offline verification of an unmodified distributed robot file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    accepted = {expected}
    if expected == PROFILE["asset_sha256"]:
        accepted.update(pin["sha256"] for pin in PROFILE.get("asset_derivatives", ())
                        if pin["source_sha256"] == expected)
    if actual not in accepted:
        raise ValueError("X5 pinned asset digest mismatch")
    return actual


def configure_stage(stage):
    """Author official core drives once, retaining pristine imported contacts."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        raise ValueError("X5 requires stage units in metres")
    for path in ROOTS:
        root = stage.GetPrimAtPath(path)
        base = stage.GetPrimAtPath(path + "/base_link")
        if not root or not base:
            raise ValueError("missing shared X5 root/base_link")
        joints = {}
        for prim in Usd.PrimRange(root):
            if prim.IsInstance():
                raise ValueError("X5 instance prim must be expanded before initialization")
            if any("mimic" in token.lower() or "tendon" in token.lower() for token in prim.GetAppliedSchemas()):
                raise ValueError("native X5 mimic/tendon constraints are forbidden")
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                joints[prim.GetName()] = prim
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb.CreateDisableGravityAttr(PROFILE["rigid_body"]["disable_gravity"])
                rb.CreateMaxDepenetrationVelocityAttr(PROFILE["rigid_body"]["max_depenetration_velocity_m_s"])
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        expected = PROFILE["arm_joints"] + PROFILE["finger_joints"]
        if set(joints) != set(expected):
            raise ValueError("unexpected X5 movable joints")
        for name in expected:
            angular = name in PROFILE["arm_joints"]
            prim = joints[name]
            if not prim.IsA(UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint):
                raise ValueError("incorrect X5 joint type: " + name)
            props = PROFILE["arm" if angular else "finger"]
            scale = math.pi / 180 if angular else 1.
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular" if angular else "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(props["stiffness"] * scale)
            drive.CreateDampingAttr(props["damping"] * scale)
            drive.CreateMaxForceAttr(props["effort_limit_Nm" if angular else "effort_limit_N"])
            api = PhysxSchema.PhysxJointAPI.Apply(prim)
            api.CreateMaxJointVelocityAttr(math.degrees(props["velocity_limit_rad_s"]) if angular else props["asset_authored_velocity_limit_m_s"])
            if angular:
                api.CreateArmatureAttr(props["armature"])
            else:
                joint = UsdPhysics.PrismaticJoint(prim)
                joint.CreateLowerLimitAttr(props["physical_limits_m"][0])
                joint.CreateUpperLimitAttr(props["physical_limits_m"][1])
        pose = UsdGeom.XformCache().GetLocalToWorldTransform(base)
        rotation = pose.ExtractRotationQuat()
        fixed = UsdPhysics.FixedJoint.Define(stage, path + "/FixedBase")
        fixed.CreateBody0Rel().SetTargets([])
        fixed.CreateBody1Rel().SetTargets([path + "/base_link"])
        fixed.CreateJointEnabledAttr(True)
        fixed.CreateExcludeFromArticulationAttr(False)
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*pose.ExtractTranslation()))
        fixed.CreateLocalRot0Attr(Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary())))
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0.))
        fixed.CreateLocalRot1Attr(Gf.Quatf(1.))
        for other in stage.Traverse():
            if other == fixed.GetPrim() or not other.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(other)
            if not joint.GetBody0Rel().GetTargets() and [str(p) for p in joint.GetBody1Rel().GetTargets()] == [path + "/base_link"]:
                joint.CreateJointEnabledAttr(False)
        UsdPhysics.ArticulationRootAPI.Apply(root)
        api = PhysxSchema.PhysxArticulationAPI.Apply(root)
        api.CreateSolverPositionIterationCountAttr(PROFILE["articulation"]["position_iterations"])
        api.CreateSolverVelocityIterationCountAttr(PROFILE["articulation"]["velocity_iterations"])
        api.CreateEnabledSelfCollisionsAttr(PROFILE["articulation"]["self_collisions"])


class X5Robot:
    def __init__(self, rgb_reader=None, *, usd_path=None, rest=None, physics_hz=120):
        if type(physics_hz) is not int or physics_hz <= 0:
            raise ValueError("physics_hz must be a positive integer")
        self._rgb_reader = rgb_reader or (lambda: {})
        self._usd_path = Path(usd_path) if usd_path is not None else None
        self._rest = rest
        self._physics_hz = physics_hz
        self._arm_reference_slew_rad_s = None
        self._arm_reference = {}
        self._configured = False
        self._arms = {}
        self._mounted = False
        self._asset_sha256 = None
        self._native_receipt = None
        self._appearance_receipt = None
        self._source_pad_binding_receipt = None
        self._active_task_spec = None
        self._task_output_dir = None
        self._task_collision_guard = None
        self._task_collision_receipt = None
        self._source_collision_binding_receipt = None
        self._initialization_started = False

    @property
    def active_task_spec(self):
        """Read-only immutable spec bound by the shared episode runner."""
        return self._active_task_spec

    def bind_task_instance(self, spec, output_dir):
        """Shared runner binds the actual immutable episode spec before task build."""
        if self._active_task_spec is not None:
            if self._active_task_spec is not spec:
                raise ValueError("shared X5 active task binding cannot be replaced")
            return
        if self._configured or self._initialization_started:
            raise RuntimeError("bind the actual task before preparing the X5 stage")
        self._active_task_spec = spec
        self._task_output_dir = Path(output_dir).resolve()

    def apply_task_collision_overlay(self, stage, descriptor):
        """Only the active selected CP09/PR2 source exceptions, before native init."""
        if (self._active_task_spec is None or self._task_output_dir is None
                or not self._configured or not self._mounted or self._initialization_started
                or self._arms or self._task_collision_guard is not None):
            raise RuntimeError("task collision overlay requires a bound prepared uninitialized X5, once")
        if self._active_task_spec.task_id == 'PR2':
            from .pr2_source_contacts import apply
            guard, receipt = apply(stage, self._active_task_spec, descriptor, self._task_output_dir)
        elif self._active_task_spec.task_id == 'CP09':
            from .task_collision_overlay import apply
            guard, receipt = apply(stage, self._active_task_spec, descriptor, self._task_output_dir,
                                   retained_source_pads=self._source_pad_binding_receipt is not None)
        else:
            raise ValueError('task collision overlay requires the active selected CP09 or PR2 acquisition spec')
        self._task_collision_guard, self._task_collision_receipt = guard, receipt
        return deepcopy(receipt)

    def verify_task_collision_overlay(self, stage):
        if self._task_collision_guard is not None:
            self._task_collision_guard.verify(stage)

    def mount(self, usd_path: Path, mounts):
        """Task chooses two station poses (position_m, orientation_wxyz) only."""
        import numpy as np
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import XFormPrim
        if not Path(usd_path).is_file():
            raise FileNotFoundError(usd_path)
        asset_sha256 = None  # Runtime checks the actual robot configuration, not export bytes.
        if self._mounted or len(mounts) != 2:
            raise ValueError("mount exactly two shared X5 arms once")
        for path, mount in zip(ROOTS, mounts):
            add_reference_to_stage(str(usd_path), path)
            XFormPrim(path).set_world_poses(positions=np.asarray([mount["position_m"]]),
                                          orientations=np.asarray([mount["orientation_wxyz"]]))
        self._mounted = True
        self._asset_sha256 = asset_sha256

    def restore_asset(self, stage):
        """Replace prepared-stage robot subtrees with the pinned asset, keep mounts."""
        if self._task_collision_guard is not None:
            raise RuntimeError("official asset cannot be restored over an authorized active collision overlay")
        from pxr import Usd, UsdGeom
        if self._usd_path is None:
            raise ValueError("prepared stage requires explicit x5 asset ID")
        if not Path(self._usd_path).is_file():
            raise FileNotFoundError(self._usd_path)
        asset_sha256 = None
        from .appearance import capture_bindings, restore_bindings, repair_camera_material_binding
        from .pr4_source_collisions import capture, restore
        source_collisions = capture(stage, self._active_task_spec)
        from .cp09_source_pads import capture as capture_source_pads, restore as restore_source_pads
        source_pads = capture_source_pads(stage, self._active_task_spec)
        source_bindings = capture_bindings(stage, ROOTS)
        from mind2act_eval.sim.stage_authoring import materialize_local_layer_stack
        materialize_local_layer_stack(stage)
        for path in ROOTS:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("prepared station lacks shared X5 root")
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            parent = prim.GetParent()
            parent_matrix = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_matrix = matrix * parent_matrix.GetInverse()
            stage.RemovePrim(path)
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(str(self._usd_path))
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(local_matrix)
        if source_collisions is not None:
            self._source_collision_binding_receipt = restore(stage, source_collisions)
        if source_pads is not None:
            self._source_pad_binding_receipt = restore_source_pads(stage, source_pads)
        self._appearance_receipt = restore_bindings(stage, source_bindings, source_asset=self._usd_path)
        self._appearance_receipt['camera_binding_repairs'] = repair_camera_material_binding(stage, ROOTS)
        self._mounted = True
        self._asset_sha256 = asset_sha256
        return {"asset_sha256": asset_sha256, "mounts_preserved": True,
                "appearance": deepcopy(self._appearance_receipt)}

    def configure_stage(self, stage):
        configure_stage(stage)
        self._configured = True
        return {"profile_id": PROFILE["profile_id"]}

    def prepare_stage(self, stage):
        asset = self.restore_asset(stage)
        return {"asset": asset, "configuration": self.configure_stage(stage)}

    def mount_bases(self, stage, mounts, *, legacy_anchors=False):
        """Apply the station's base poses to the shared FixedBase joints (after prepare_stage, before reset).

        mounts: [{"name": "Left"|"Right", "position": [x,y,z], "euler_deg": [rx,ry,rz]}, ...].
        legacy_anchors: also switch off the older station anchors "/World/FixedBase{Left,Right}" when present.
        Only the base-to-world placement is written; no drive, mass or contact value changes.
        """
        if not self._configured:
            raise RuntimeError("prepare/configure X5 before mounting bases")
        if self._arms:
            raise RuntimeError("mount bases before creating articulations / world.reset")
        from pxr import Gf, UsdPhysics
        names = [m["name"] for m in mounts]
        if sorted(names) != ["Left", "Right"]:
            raise ValueError("mount exactly the Left and Right station robots")
        disabled = []
        for side in (names if legacy_anchors else ()):
            prim = stage.GetPrimAtPath("/World/FixedBase" + side)
            if prim:
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                disabled.append(str(prim.GetPath()))
        for mount in mounts:
            position, (w, x, y, z) = base_joint_pose(mount["position"], mount["euler_deg"])
            joint = stage.GetPrimAtPath("/World/Robot" + mount["name"] + "/Asset/FixedBase")
            if not joint or not joint.IsA(UsdPhysics.FixedJoint):
                raise ValueError("shared X5 FixedBase joint missing; call prepare_stage first")
            fixed = UsdPhysics.FixedJoint(joint)
            fixed.CreateLocalPos0Attr(Gf.Vec3f(*position))
            fixed.CreateLocalRot0Attr(Gf.Quatf(w, x, y, z))
            fixed.CreateLocalPos1Attr(Gf.Vec3f(0.))
            fixed.CreateLocalRot1Attr(Gf.Quatf(1.))
            fixed.CreateExcludeFromArticulationAttr(False)
            pos0, rot0 = fixed.GetLocalPos0Attr().Get(), fixed.GetLocalRot0Attr().Get()
            quat = (rot0.GetReal(), *rot0.GetImaginary())
            if (max(abs(a - b) for a, b in zip(pos0, position)) > 1e-6
                    or max(abs(a - b) for a, b in zip(quat, (w, x, y, z))) > 1e-6
                    or not fixed.GetJointEnabledAttr().Get()):
                raise ValueError("FixedBase read-back differs from the requested mount: " + mount["name"])
        self._mount_applied = True
        return {"mounted": sorted(names), "anchors_disabled": disabled}

    def attach_cameras(self, stage):
        from pxr import Gf, UsdGeom, UsdPhysics
        for root in ROOTS:
            for name in ("hand_to_camera_mount", "camera_joint"):
                prim = stage.GetPrimAtPath(root + "/joints/" + name)
                if not prim or not prim.IsA(UsdPhysics.FixedJoint):
                    raise ValueError("official X5 camera fixed joint missing")
                UsdPhysics.FixedJoint(prim).CreateJointEnabledAttr().Set(True)
            camera = UsdGeom.Camera.Define(stage, root + "/camera/PhysCoWristCamera")
            profile = PROFILE["wrist_camera"]
            xformable = UsdGeom.Xformable(camera.GetPrim())
            translate = _get_or_add_xform_op(
                xformable, "xformOp:translate", camera.AddTranslateOp)
            orient = _get_or_add_xform_op(
                xformable, "xformOp:orient", camera.AddOrientOp)
            translate.Set(Gf.Vec3d(*profile["position_m"]))
            w, x, y, z = _intrinsic_xyz_quaternion(profile["rotation_xyz_deg"])
            orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            camera.CreateFocalLengthAttr().Set(profile["focal_length"])
            camera.CreateHorizontalApertureAttr().Set(profile["horizontal_aperture"])
            camera.CreateVerticalApertureAttr().Set(profile["vertical_aperture"])
            camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*profile["clipping_range"]))

    def articulations(self, world):
        from isaacsim.core.api.robots import Robot
        if not self._configured:
            raise RuntimeError("prepare/configure X5 before creating articulations")
        if not self._arms:
            for side, path in zip(("left", "right"), ROOTS):
                self._arms[side] = world.scene.add(Robot(prim_path=path, name="x5_" + side))
        return {side.capitalize(): robot for side, robot in self._arms.items()}

    def rest_pose(self):
        import numpy as np
        if self._rest is None:
            raise ValueError("robot_rest must explicitly record the baseline's eight-value rest pose")
        values = np.asarray(self._rest, dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("robot_rest must contain eight finite SI joint values")
        return values.copy()

    def reset_to_rest(self):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        rest = self.rest_pose()
        for robot in self._arms.values():
            robot.set_joint_positions(rest)
            robot.set_joint_velocities(np.zeros(8))
            robot.apply_action(ArticulationAction(joint_positions=rest, joint_velocities=np.zeros(8)))
        self._arm_reference = {side: rest[:6].copy() for side in self._arms}

    def configure_arm_reference_slew(self, max_rad_s):
        """Opt into a software arm-reference slew limit, applied once per physics tick."""
        value = float(max_rad_s)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("arm reference slew must be finite and positive rad/s")
        if self._arm_reference:
            raise RuntimeError("configure arm reference slew before episode initialization")
        self._arm_reference_slew_rad_s = value
        return self.software_control_receipt()

    def software_control_receipt(self):
        return {
            "gripper_target_step_fraction": PROFILE["command"]["target_step_fraction"],
            "arm_reference_slew_rad_s": self._arm_reference_slew_rad_s,
            "physics_hz": self._physics_hz,
            "per_physics_tick_delta_limit_rad": (
                self._arm_reference_slew_rad_s / self._physics_hz
                if self._arm_reference_slew_rad_s is not None else None
            ),
            "native_parameter": False,
        }

    def park_arm(self, side, joint_deltas):
        """Offset the current initialized pose once before tick 1 (repeated calls accumulate); baseline CP03 clears the selector grip."""
        import numpy as np
        robot = self._arms.get(side.lower())
        deltas = np.asarray(joint_deltas, dtype=float)
        if robot is None or deltas.shape != (6,) or not np.isfinite(deltas).all():
            raise ValueError("park_arm needs a created arm and six finite joint deltas")
        from isaacsim.core.utils.types import ArticulationAction
        target = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        target[:6] += deltas
        robot.set_joint_positions(target)
        robot.set_joint_velocities(np.zeros(8))
        robot.apply_action(ArticulationAction(joint_positions=target, joint_velocities=np.zeros(8)))
        self._arm_reference[side.lower()] = target[:6].copy()
        return {"side": side, "target": target.tolist()}

    def verify(self, robots, stage):
        if set(robots) != {"Left", "Right"} or any(robots[side.capitalize()] is not robot for side, robot in self._arms.items()):
            raise ValueError("native X5 handle identity mismatch")
        self.verify_native()
        return {"profile_id": PROFILE["profile_id"], "native_core_verified": True}

    def initialize(self, world):
        self._initialization_started = True
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()
        if self._arms:
            # This path may be entered after a task prepared the robot itself.
            # Reuse attach_cameras' idempotence without rebuilding its stage.
            self.attach_cameras(stage)
            self.verify_native()
            self.verify_task_collision_overlay(stage)
            return
        if not self._mounted:
            self.prepare_stage(stage)
        if not self._configured:
            self.configure_stage(stage)
        # prepare_stage/restore_asset can replace the Asset subtree, so cameras
        # must be attached only after that work and before articulation/reset.
        self.attach_cameras(stage)
        self.articulations(world)
        world.reset()
        self.reset_to_rest()
        self.verify_native()
        self.verify_task_collision_overlay(stage)

    def verify_native(self):
        import numpy as np
        if set(self._arms) != {"left", "right"} or len({id(robot) for robot in self._arms.values()}) != 2:
            raise RuntimeError("both distinct X5 arms must be initialized before native verification")
        robot_receipts = {}
        for side, robot in self._arms.items():
            if tuple(robot.dof_names) != PROFILE["arm_joints"] + PROFILE["finger_joints"]:
                raise ValueError("native X5 joint order mismatch")
            properties = robot.dof_properties
            raw_properties = {
                name: np.asarray(properties[name]).reshape(-1).tolist()
                for name in ("stiffness", "damping", "maxEffort", "maxVelocity", "lower", "upper")
            }
            for name, expected in (("stiffness", [PROFILE["arm"]["stiffness"]] * 6 + [PROFILE["finger"]["stiffness"]] * 2),
                                   ("damping", [PROFILE["arm"]["damping"]] * 6 + [PROFILE["finger"]["damping"]] * 2),
                                   ("maxEffort", [PROFILE["arm"]["effort_limit_Nm"]] * 6 + [PROFILE["finger"]["effort_limit_N"]] * 2),
                                   ("maxVelocity", [PROFILE["arm"]["velocity_limit_rad_s"]] * 6 + [PROFILE["finger"]["asset_authored_velocity_limit_m_s"]] * 2)):
                if not np.allclose(raw_properties[name], expected, rtol=1e-6):
                    raise ValueError("native X5 property mismatch: " + name)
            lower, upper = PROFILE["finger"]["physical_limits_m"]
            if not np.allclose(raw_properties["lower"][6:], lower, atol=1e-7) or not np.allclose(raw_properties["upper"][6:], upper, atol=1e-7):
                raise ValueError("native X5 finger limits mismatch")
            view = robot._articulation_view
            raw_armatures = np.asarray(view.get_armatures()).reshape(-1).tolist()
            if not np.allclose(raw_armatures[:6], PROFILE["arm"]["armature"], atol=1e-7):
                raise ValueError("native X5 armature mismatch")
            articulation = {
                key: {
                    "value": np.asarray(getattr(view, method)()).reshape(-1).tolist(),
                    "source": "ArticulationView." + method,
                }
                for method, key in (("get_solver_position_iteration_counts", "position_iterations"),
                                    ("get_solver_velocity_iteration_counts", "velocity_iterations"),
                                    ("get_enabled_self_collisions", "self_collisions"))
            }
            for key, observed in articulation.items():
                if observed["value"] != [PROFILE["articulation"][key]]:
                    raise ValueError("native X5 articulation mismatch: " + key)
            joints = {}
            unsupported = ["native drive type", "native mimic/coupling graph", "per-tick joint positions"]
            for index, name in enumerate(robot.dof_names):
                angular = index < len(PROFILE["arm_joints"])
                if angular:
                    expectation = {
                        "stiffness": PROFILE["arm"]["stiffness"],
                        "damping": PROFILE["arm"]["damping"],
                        "maxEffort": PROFILE["arm"]["effort_limit_Nm"],
                        "maxVelocity": PROFILE["arm"]["velocity_limit_rad_s"],
                    }
                else:
                    expectation = {
                        "stiffness": PROFILE["finger"]["stiffness"],
                        "damping": PROFILE["finger"]["damping"],
                        "maxEffort": PROFILE["finger"]["effort_limit_N"],
                        "maxVelocity": PROFILE["finger"]["asset_authored_velocity_limit_m_s"],
                        "lower": PROFILE["finger"]["physical_limits_m"][0],
                        "upper": PROFILE["finger"]["physical_limits_m"][1],
                    }
                units = {
                    "stiffness": "N m/rad" if angular else "N/m",
                    "damping": "N m s/rad" if angular else "N s/m",
                    "maxEffort": "N m" if angular else "N",
                    "maxVelocity": "rad/s" if angular else "m/s",
                    "lower": "rad" if angular else "m",
                    "upper": "rad" if angular else "m",
                }
                joints[name] = {}
                for key in raw_properties:
                    expected = expectation.get(key)
                    joints[name][key] = {
                        "value": raw_properties[key][index],
                        "unit": units[key],
                        "source": "Robot.dof_properties",
                        "profile_value": expected,
                        "matches_profile": None if expected is None else math.isclose(
                            raw_properties[key][index], expected, rel_tol=1e-6, abs_tol=1e-7),
                    }
                if index < len(raw_armatures):
                    joints[name]["armature"] = {
                        "value": raw_armatures[index], "unit": "native_api",
                        "profile_value": PROFILE["arm"]["armature"] if angular else None,
                        "matches_profile": (math.isclose(raw_armatures[index], PROFILE["arm"]["armature"],
                                                          rel_tol=1e-6, abs_tol=1e-7) if angular else None),
                        "source": "ArticulationView.get_armatures",
                    }
                elif not angular:
                    joints[name]["armature"] = {"status": "unsupported"}
            robot_receipts[side] = {
                "joint_names": list(robot.dof_names),
                "joints": joints,
                "articulation": articulation,
                "status": "verified",
                "unsupported": unsupported,
            }
        self._native_receipt = {
            "schema": "mind2act_eval.native_robot_receipt.v1",
            "status": "verified",
            "identity": {
                "profile_id": PROFILE["profile_id"],
                "official_code_commit": PROFILE["official_code_commit"],
                "official_assets_commit": PROFILE["official_assets_commit"],
                "asset_sha256": self._asset_sha256,
                "official_asset_sha256": PROFILE["asset_sha256"],
                "urdf_sha256": PROFILE["urdf_sha256"],
            },
            "robots": robot_receipts,
            "shared_software_control": self.software_control_receipt(),
            "motion_readback": {"initial": self._joint_position_snapshot()},
            "unsupported": [
                "native drive type readback",
                "native mimic/coupling graph readback",
                "per-tick raw joint7/joint8 positions and symmetry error",
                "fixed-base and rigid-body parameter readback",
            ],
        }

    def _joint_position_snapshot(self):
        """Diagnostic getter only: six arm radians and two finger metres per side."""
        import numpy as np
        try:
            positions = {side: np.asarray(robot.get_joint_positions(), dtype=float)
                         for side, robot in self._arms.items()}
            if set(positions) != {"left", "right"} or any(
                    row.shape != (8,) or not np.isfinite(row).all() for row in positions.values()):
                raise ValueError("expected two finite eight-joint native position readbacks")
            return {"status": "observed", "source": "Robot.get_joint_positions",
                    "positions": {side: row.tolist() for side, row in positions.items()},
                    "units_by_index": ["rad"] * 6 + ["m"] * 2}
        except Exception as exc:
            return {"status": "unavailable", "source": "Robot.get_joint_positions",
                    "error": f"{type(exc).__name__}: {exc}"}

    def native_receipt(self):
        """Return the last actual native verification snapshot; never substitute profile values."""
        if self._native_receipt is None:
            return {
                "schema": "mind2act_eval.native_robot_receipt.v1",
                "status": "unavailable",
                "shared_software_control": self.software_control_receipt(),
                "appearance": deepcopy(self._appearance_receipt),
                "source_collision_binding": deepcopy(self._source_collision_binding_receipt),
                "source_pad_binding": deepcopy(self._source_pad_binding_receipt),
                "unsupported": ["native initialization/readback did not complete"],
            }
        receipt = deepcopy(self._native_receipt)
        receipt.setdefault("motion_readback", {})["final"] = self._joint_position_snapshot()
        receipt["appearance"] = deepcopy(self._appearance_receipt)
        if self._source_collision_binding_receipt is not None:
            receipt["source_collision_binding"] = deepcopy(self._source_collision_binding_receipt)
        if self._source_pad_binding_receipt is not None:
            receipt["source_pad_binding"] = deepcopy(self._source_pad_binding_receipt)
        if self._task_collision_receipt is not None:
            receipt["source_collision_overlay"] = deepcopy(self._task_collision_receipt)
        return receipt

    def apply(self, action: Action):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        if len(self._arms) != 2:
            raise RuntimeError("shared X5 is not initialized")
        for side in ("left", "right"):
            robot = self._arms[side]
            requested_arm = np.asarray(getattr(action, side + "_arm"), dtype=float)
            previous = self._arm_reference.get(side)
            if previous is None:
                previous = np.asarray(robot.get_joint_positions(), dtype=float)[:6]
            if self._arm_reference_slew_rad_s is None:
                arm_target = requested_arm
            else:
                max_delta = self._arm_reference_slew_rad_s / self._physics_hz
                arm_target = previous + np.clip(requested_arm - previous, -max_delta, max_delta)
            self._arm_reference[side] = np.asarray(arm_target, dtype=float).copy()
            requested = normalized_target(getattr(action, side + "_gripper"))[0]
            pair = limited_target(requested, float(robot.get_joint_positions()[6]))
            targets = (*arm_target, *pair)
            robot.apply_action(ArticulationAction(joint_positions=np.asarray(targets), joint_velocities=np.zeros(8)))

    def observe(self):
        if len(self._arms) != 2:
            raise RuntimeError("shared X5 is not initialized")
        result = {}
        for side, robot in self._arms.items():
            positions = robot.get_joint_positions()
            result[side + "_arm"] = tuple(float(v) for v in positions[:6])
            result[side + "_gripper"] = normalized_observation(positions[6])
        result["rgb"] = self._rgb_reader()
        return result
