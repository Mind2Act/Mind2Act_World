"""Official Isaac 5.1 constructor branches on real CPU USD; no Kit acceptance."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


@pytest.fixture
def sdk(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    paths = {side: '/World/Robot'+side+'/Asset/link6' for side in ('Left', 'Right')}
    non_root, calls, writes = set(paths.values()), [], []
    for path in paths.values():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float).Set(.037)

    class RigidView:
        def __init__(self, prim_paths_expr, name, *, reset_xform_properties=True,
                     prepare_contact_sensors=True, track_contact_forces=False,
                     disable_stablization=True, contact_filter_prim_paths_expr=()):
            self.path = prim_paths_expr
            self.initialized = False
            calls.append(self)
            prim = stage.GetPrimAtPath(self.path)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim); writes.append('api')
            if prepare_contact_sensors or track_contact_forces or contact_filter_prim_paths_expr:
                prim.GetAttribute('physxRigidBody:sleepThreshold').Set(0.); writes.append('sleep')
            if reset_xform_properties or disable_stablization:
                prim.CreateAttribute('test:unwantedSDKWrite', Sdf.ValueTypeNames.Bool).Set(True)
                writes.append('reset/stabilization')
        def initialize(self): self.initialized = True
        def get_world_poses(self):
            assert self.initialized
            return np.array([[.3,.4,1.]]), np.array([[1.,0.,0.,0.]])
        def get_linear_velocities(self): return np.array([[.1,.2,.3]])
        def get_angular_velocities(self): return np.array([[.4,.5,.6]])

    for name in ('isaacsim', 'isaacsim.core', 'isaacsim.core.prims', 'isaacsim.core.utils', 'isaacsim.core.utils.prims'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['isaacsim.core.prims'].RigidPrim = RigidView
    sys.modules['isaacsim.core.utils.prims'].is_prim_non_root_articulation_link = lambda *, prim_path: prim_path in non_root
    return SimpleNamespace(stage=stage, paths=paths, non_root=non_root, calls=calls, writes=writes, view=RigidView)


def test_official_default_sensor_branch_reproduces_robot_sleep_write(sdk):
    before = sdk.stage.GetRootLayer().ExportToString()
    sdk.view(sdk.paths['Left'], 'legacy', reset_xform_properties=False)
    assert 'sleep' in sdk.writes
    assert sdk.stage.GetRootLayer().ExportToString() != before


def test_readers_preserve_usd_and_actual_native_pose_velocity(sdk):
    from mind2act_eval.robot.read_only import robot_body_readers
    before = sdk.stage.GetRootLayer().ExportToString()
    registered = []
    world = SimpleNamespace(scene=SimpleNamespace(add=lambda view: registered.append(view) or view))
    readers = robot_body_readers(sdk.stage, sdk.paths, world=world)
    for view in registered: view.initialize()  # world.reset initializes registered native views
    np.testing.assert_array_equal(readers['Left'].get_world_pose()[0], [.3,.4,1.])
    np.testing.assert_array_equal(readers['Left'].get_world_pose()[1], [1.,0.,0.,0.])
    np.testing.assert_array_equal(readers['Left'].get_linear_velocity(), [.1,.2,.3])
    np.testing.assert_array_equal(readers['Left'].get_angular_velocity(), [.4,.5,.6])
    assert not hasattr(readers['Left'], 'set_world_pose')
    assert len(registered) == 2 and sdk.writes == []
    assert sdk.stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('failure', ['missing_prim', 'missing_api'])
def test_all_links_are_validated_before_any_sdk_constructor(sdk, failure):
    from mind2act_eval.robot.read_only import robot_body_readers
    path = sdk.paths['Right']
    if failure == 'missing_prim': sdk.stage.RemovePrim(path)
    elif failure == 'missing_api': sdk.stage.GetPrimAtPath(path).RemoveAPI(UsdPhysics.RigidBodyAPI)
    before = sdk.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='existing rigid body'):
        robot_body_readers(sdk.stage, sdk.paths)
    assert sdk.calls == sdk.writes == []
    assert sdk.stage.GetRootLayer().ExportToString() == before


def test_root_body_uses_live_native_tensors_without_sdk_reset_or_usd_fallback(sdk, monkeypatch):
    from mind2act_eval.robot.read_only import robot_body_readers
    sdk.non_root.clear()
    path = '/World/RobotLeft/Asset/base_link'
    prim = UsdGeom.Xform.Define(sdk.stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    pose = np.array([[.7,.8,.9,0.,0.,np.sqrt(.5),np.sqrt(.5)]])
    velocities = np.array([[1.,2.,3.,4.,5.,6.]])
    tensor = SimpleNamespace(count=1, get_transforms=lambda: pose, get_velocities=lambda: velocities)
    module = ModuleType('isaacsim.core.simulation_manager')
    requested = []
    def create(paths):
        requested.append(paths)
        return tensor
    module.SimulationManager = SimpleNamespace(get_physics_sim_view=lambda:
        SimpleNamespace(create_rigid_body_view=create))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    before = sdk.stage.GetRootLayer().ExportToString()
    reader = robot_body_readers(sdk.stage, {'base': path})['base']
    reader.initialize()
    np.testing.assert_allclose(reader.get_world_pose()[1], [np.sqrt(.5),0.,0.,np.sqrt(.5)])
    np.testing.assert_array_equal(reader.get_world_pose()[0], [.7,.8,.9])
    np.testing.assert_array_equal(reader.get_linear_velocity(), [1.,2.,3.])
    np.testing.assert_array_equal(reader.get_angular_velocity(), [4.,5.,6.])
    pose[0,0] = 2.
    assert reader.get_world_pose()[0][0] == 2.  # actual moving state, never authored USD
    assert sdk.calls == sdk.writes == []
    assert requested == [[path]]
    assert sdk.stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('count', [None, 0, 2])
def test_root_requires_available_unique_native_body(sdk, monkeypatch, count):
    from mind2act_eval.robot.read_only import robot_body_readers
    sdk.non_root.clear()
    module = ModuleType('isaacsim.core.simulation_manager')
    module.SimulationManager = SimpleNamespace(get_physics_sim_view=lambda: None if count is None else
        SimpleNamespace(create_rigid_body_view=lambda paths: SimpleNamespace(count=count)))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    reader = robot_body_readers(sdk.stage, {'base': sdk.paths['Left']})['base']
    with pytest.raises(RuntimeError): reader.get_world_pose()
    assert sdk.calls == sdk.writes == []
