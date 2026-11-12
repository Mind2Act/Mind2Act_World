import pytest
from types import SimpleNamespace
import sys
from mind2act_eval.robot import X5Robot, normalized_target, limited_target


def test_uninitialized_robot_cannot_claim_native_verification():
    robot = X5Robot()
    with pytest.raises(RuntimeError, match="initialized"):
        robot.verify_native()


def test_initialize_attaches_wrist_cameras_before_native_reset(monkeypatch):
    from types import ModuleType
    events = []
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    stage_module = ModuleType("isaacsim.core.utils.stage")
    stage_module.get_current_stage = lambda: "active-stage"
    monkeypatch.setitem(sys.modules, stage_module.__name__, stage_module)

    robot = X5Robot()
    stage = {"camera_exists": True}
    def prepare_stage(active_stage):
        assert active_stage == "active-stage"
        events.append(("prepare_stage", active_stage))
        # restore_asset replaces the whole Asset subtree and drops earlier prims.
        stage["camera_exists"] = False
        robot._mounted = True
        robot._configured = True
    robot.prepare_stage = prepare_stage
    def attach_cameras(active_stage):
        assert active_stage == "active-stage"
        events.append(("attach_cameras", active_stage))
        stage["camera_exists"] = True
    robot.attach_cameras = attach_cameras
    robot.articulations = lambda world: events.append(("articulations", world))
    robot.reset_to_rest = lambda: events.append(("rest_pose",))
    robot.verify_native = lambda: events.append(("verify_native",))
    class World:
        def reset(self):
            assert stage["camera_exists"]
            events.append(("world_reset",))

    world = World()
    robot.initialize(world)

    assert events == [
        ("prepare_stage", "active-stage"),
        ("attach_cameras", "active-stage"),
        ("articulations", world),
        ("world_reset",),
        ("rest_pose",),
        ("verify_native",),
    ]


def test_initialize_with_existing_arms_rechecks_cameras_without_rebuilding_stage(monkeypatch):
    from types import ModuleType

    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    stage_module = ModuleType("isaacsim.core.utils.stage")
    stage_module.get_current_stage = lambda: "active-stage"
    monkeypatch.setitem(sys.modules, stage_module.__name__, stage_module)

    robot = X5Robot()
    robot._arms = {"left": object(), "right": object()}
    events = []
    robot.attach_cameras = lambda stage: events.append(("attach_cameras", stage))
    robot.prepare_stage = lambda stage: pytest.fail("existing arms must not rebuild the stage")
    robot.articulations = lambda world: pytest.fail("existing arms must not be recreated")
    robot.verify_native = lambda: events.append(("verify_native",))

    robot.initialize(object())

    assert events == [("attach_cameras", "active-stage"), ("verify_native",)]


def test_native_verification_receipt_contains_actual_per_joint_parameters():
    from mind2act_eval.robot import PROFILE

    class View:
        def get_armatures(self):
            return [PROFILE["arm"]["armature"]] * 6 + [0., 0.]
        def get_solver_position_iteration_counts(self):
            return [PROFILE["articulation"]["position_iterations"]]
        def get_solver_velocity_iteration_counts(self):
            return [PROFILE["articulation"]["velocity_iterations"]]
        def get_enabled_self_collisions(self):
            return [PROFILE["articulation"]["self_collisions"]]

    class NativeRobot:
        dof_names = PROFILE["arm_joints"] + PROFILE["finger_joints"]
        _articulation_view = View()
        dof_properties = {
            "stiffness": [PROFILE["arm"]["stiffness"]] * 6 + [PROFILE["finger"]["stiffness"]] * 2,
            "damping": [PROFILE["arm"]["damping"]] * 6 + [PROFILE["finger"]["damping"]] * 2,
            "maxEffort": [PROFILE["arm"]["effort_limit_Nm"]] * 6 + [PROFILE["finger"]["effort_limit_N"]] * 2,
            "maxVelocity": [PROFILE["arm"]["velocity_limit_rad_s"]] * 6 + [PROFILE["finger"]["asset_authored_velocity_limit_m_s"]] * 2,
            "lower": [-3.14] * 6 + [PROFILE["finger"]["physical_limits_m"][0]] * 2,
            "upper": [3.14] * 6 + [PROFILE["finger"]["physical_limits_m"][1]] * 2,
        }

    robot = X5Robot()
    positions = {"left": [0.] * 8, "right": [0.] * 8}
    robot._arms = {"left": NativeRobot(), "right": NativeRobot()}
    for side, arm in robot._arms.items():
        arm.get_joint_positions = lambda side=side: positions[side][:]
    robot.verify_native()
    positions["left"][1] = .3
    receipt = robot.native_receipt()
    assert receipt["motion_readback"]["initial"]["positions"]["left"] == [0.] * 8
    assert receipt["motion_readback"]["final"]["positions"]["left"] == [0., .3, 0., 0., 0., 0., 0., 0.]
    assert receipt["motion_readback"]["final"]["source"] == "Robot.get_joint_positions"

    assert receipt["status"] == "verified"
    for side in ("left", "right"):
        joints = receipt["robots"][side]["joints"]
        assert joints["joint1"]["maxEffort"]["value"] == 100.0
        assert joints["joint1"]["maxEffort"]["unit"] == "N m"
        assert joints["joint1"]["maxEffort"]["matches_profile"] is True
        assert joints["joint7"]["maxEffort"]["value"] == 100.0
        assert joints["joint7"]["maxEffort"]["unit"] == "N"
        assert joints["joint7"]["maxEffort"]["matches_profile"] is True
        assert joints["joint8"]["upper"]["value"] == .044
        assert joints["joint7"]["armature"]["value"] == 0.
    assert "per-tick raw joint7/joint8 positions and symmetry error" in receipt["unsupported"]


def test_official_gripper_endpoints_and_actual_state_step_limit():
    assert normalized_target(0.) == pytest.approx((-.01, -.01))
    assert normalized_target(1.) == pytest.approx((.044, .044))
    assert limited_target(-.01, .044) == pytest.approx((.0332, .0332))


def test_arm_reference_slew_is_opt_in_per_physics_tick_and_recorded(monkeypatch):
    import numpy as np
    from types import ModuleType
    from mind2act_eval.contracts import Action

    types_module = ModuleType("isaacsim.core.utils.types")
    types_module.ArticulationAction = lambda **kwargs: kwargs
    monkeypatch_modules = {
        "isaacsim": ModuleType("isaacsim"),
        "isaacsim.core": ModuleType("isaacsim.core"),
        "isaacsim.core.utils": ModuleType("isaacsim.core.utils"),
        "isaacsim.core.utils.types": types_module,
    }
    for name, module in monkeypatch_modules.items():
        if name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    robot = X5Robot(physics_hz=960)
    robot._arms = {"left": SimpleNamespace(get_joint_positions=lambda: np.zeros(8), apply_action=lambda action: None),
                   "right": SimpleNamespace(get_joint_positions=lambda: np.zeros(8), apply_action=lambda action: None)}
    robot.configure_arm_reference_slew(5.0)
    requested = Action((1.,) * 6, (1.,) * 6, 0., 0.)
    robot.apply(requested)
    assert robot._arm_reference["left"] == pytest.approx(np.full(6, 5. / 960.))
    assert robot._arm_reference["right"] == pytest.approx(np.full(6, 5. / 960.))
    assert robot.software_control_receipt() == {
        "gripper_target_step_fraction": .2,
        "arm_reference_slew_rad_s": 5.0,
        "physics_hz": 960,
        "per_physics_tick_delta_limit_rad": pytest.approx(5. / 960.),
        "native_parameter": False,
    }

    unrestricted = X5Robot(physics_hz=960)
    unrestricted._arms = robot._arms
    unrestricted.apply(requested)
    assert unrestricted._arm_reference["left"] == pytest.approx(np.ones(6))


def test_base_joint_pose_matches_extrinsic_xyz_euler():
    from mind2act_eval.robot import base_joint_pose
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    position, (w, x, y, z) = base_joint_pose([-.3, -.45, .765], [10, -20, 90])
    assert position == (-.3, -.45, .765)
    expected = rotation.from_euler("xyz", [10, -20, 90], degrees=True).as_quat()
    assert (x, y, z, w) == pytest.approx(tuple(expected), abs=1e-12)
    with pytest.raises(ValueError):
        base_joint_pose([0, 0], [0, 0, 0])


def test_mount_and_park_require_prepared_robot():
    robot = X5Robot()
    with pytest.raises(RuntimeError):
        robot.mount_bases(None, [{"name": "Left"}, {"name": "Right"}])
    with pytest.raises(ValueError):
        robot.park_arm("Left", [0] * 6)


def test_attach_cameras_creates_camera_prims_at_the_shared_wrist_paths(monkeypatch):
    from mind2act_eval.robot import ROOTS

    created = {}
    enabled = []

    class Attr:
        def __init__(self, on_set=None):
            self.value = None
            self.on_set = on_set
        def Set(self, value):
            self.value = value
            if self.on_set:
                self.on_set(value)

    class Camera:
        def __init__(self, path):
            self.path = path
            self.ops = {}
            self.properties = {}
        def GetPrim(self):
            return self
        def GetOrderedXformOps(self):
            return [SimpleNamespace(GetOpName=lambda name=name: name, IsInverseOp=lambda: False,
                                     Set=attribute.Set)
                    for name, attribute in self.ops.items()]
        def AddTranslateOp(self):
            if "xformOp:translate" in self.ops:
                raise RuntimeError("duplicate xformOp:translate")
            self.ops["xformOp:translate"] = Attr()
            return self.ops["xformOp:translate"]
        def AddOrientOp(self):
            if "xformOp:orient" in self.ops:
                raise RuntimeError("duplicate xformOp:orient")
            self.ops["xformOp:orient"] = Attr()
            return self.ops["xformOp:orient"]
        def __getattr__(self, name):
            if name.startswith("Create") and name.endswith("Attr"):
                key = name[6:-4]
                def create(value=None):
                    attribute = self.properties.setdefault(key, Attr())
                    if value is not None:
                        attribute.Set(value)
                    return attribute
                return create
            raise AttributeError(name)

    class Prim:
        def __init__(self, path):
            self.path = path
        def IsA(self, schema):
            return True

    class FixedJoint:
        def __init__(self, prim):
            self.prim = prim
        def CreateJointEnabledAttr(self, value=None):
            return Attr(lambda actual: enabled.append((self.prim.path, actual)))

    class Stage:
        def GetPrimAtPath(self, path):
            return Prim(path)

    fake_pxr = SimpleNamespace(
        UsdPhysics=SimpleNamespace(FixedJoint=FixedJoint),
        UsdGeom=SimpleNamespace(
            Camera=SimpleNamespace(Define=lambda stage, path: created.setdefault(path, Camera(path))),
            Xformable=lambda prim: prim),
        Gf=SimpleNamespace(Vec3d=lambda *value: value, Vec3f=lambda *value: value,
                           Vec2f=lambda *value: value,
                           Quatf=lambda real, imaginary: (real, *imaginary)),
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    robot = X5Robot()
    robot.attach_cameras(Stage())
    robot.attach_cameras(Stage())

    expected_paths = [root + "/camera/PhysCoWristCamera" for root in ROOTS]
    assert list(created) == expected_paths
    for camera in created.values():
        assert camera.ops["xformOp:translate"].value == (0., 0., .001)
        assert camera.ops["xformOp:orient"].value == pytest.approx(
            (.5416752204197018, .4545194776720437, -.4545194776720437, -.5416752204197018))
        assert len(camera.ops) == 2
        assert camera.properties["FocalLength"].value == 13.
        assert camera.properties["HorizontalAperture"].value == 20.955
        assert camera.properties["VerticalAperture"].value == 15.71625
        assert camera.properties["ClippingRange"].value == (.0001, 10.)
    assert len(enabled) == 8


def test_attach_cameras_twice_on_real_usd_stage_is_idempotent():
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from mind2act_eval.robot import ROOTS, PROFILE

    stage = Usd.Stage.CreateInMemory()
    for root in ROOTS:
        stage.DefinePrim(root, "Xform")
        stage.DefinePrim(root + "/camera", "Xform")
        for name in ("hand_to_camera_mount", "camera_joint"):
            UsdPhysics.FixedJoint.Define(stage, root + "/joints/" + name)

    robot = X5Robot()
    robot.attach_cameras(stage)
    robot.attach_cameras(stage)

    for root in ROOTS:
        camera = UsdGeom.Camera(stage.GetPrimAtPath(root + "/camera/PhysCoWristCamera"))
        xformable = UsdGeom.Xformable(camera.GetPrim())
        source_quaternion_xyzw = Rotation.from_euler(
            "XYZ", PROFILE["wrist_camera"]["rotation_xyz_deg"], degrees=True).as_quat()
        actual_usd_quaternion = xformable.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).ExtractRotationQuat()
        actual_quaternion_xyzw = (*actual_usd_quaternion.GetImaginary(), actual_usd_quaternion.GetReal())
        actual_rotation = Rotation.from_quat(actual_quaternion_xyzw).as_matrix()
        expected_rotation = Rotation.from_quat(source_quaternion_xyzw).as_matrix()
        assert actual_rotation == pytest.approx(expected_rotation, abs=1e-7)
        ops = xformable.GetOrderedXformOps()
        assert [op.GetOpName() for op in ops] == ["xformOp:translate", "xformOp:orient"]
        assert tuple(camera.GetPrim().GetAttribute("xformOp:translate").Get()) == (0., 0., .001)
        assert camera.GetFocalLengthAttr().Get() == 13.
