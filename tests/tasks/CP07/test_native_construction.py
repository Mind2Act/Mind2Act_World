"""Actual constructor with CPU dependency substitutes: task/robot write boundary."""
import sys
from types import ModuleType, SimpleNamespace as NS

from mind2act_eval.tasks.CP07.scene import NativeSensors


def test_native_constructor_authors_task_contacts_but_only_reads_robot_bodies(monkeypatch):
    writes, steps = [], []

    class Prim:
        def __init__(self, path):self.path = path
        def IsA(self, kind):return "/wheel" in self.path
        def GetName(self):return self.path.rsplit("/", 1)[-1]
        def HasAPI(self, kind):return self.path.startswith('/World/Robot')

    class Body:
        def __init__(self, prim_path, name, *, prepare_contact_sensors=True, **kwargs):
            self.path = prim_path
            if prepare_contact_sensors:
                writes.append(('sdk_sleep', prim_path))
        def get_world_pose(self):return [0., 0., 1.], [1., 0., 0., 0.]

    class RigidView(Body):
        def __init__(self, prim_paths_expr, name, **kwargs):
            super().__init__(prim_paths_expr, name, **kwargs)
        def get_world_poses(self):return [[0., 0., 1.]], [[1., 0., 0., 0.]]

    class Drive:
        def __init__(self, prim, kind):self.prim = prim
        def GetPrim(self):return self.prim
        def GetTargetVelocityAttr(self):return NS(Set=lambda value: None)

    class ContactAPI:
        @staticmethod
        def Apply(prim):
            writes.append(("Apply", prim.path))
            return NS(CreateThresholdAttr=lambda value: writes.append(("threshold", prim.path)))

    bounds = NS(GetMin=lambda: (-.01, -.01, -.01), GetMax=lambda: (.01, .01, .01))
    bbox = NS(ComputeUntransformedBound=lambda prim: NS(ComputeAlignedRange=lambda: bounds),
              ComputeWorldBound=lambda prim:NS(ComputeAlignedRange=lambda:bounds))
    pxr = ModuleType("pxr")
    pxr.Usd = NS(TimeCode=NS(Default=lambda: 0),
                 PrimRange=lambda prim: [Prim("/World/Car/wheel"+str(i)) for i in range(4)])
    pxr.UsdGeom = NS(BBoxCache=lambda *args: bbox, Tokens=NS(default_="default"))
    pxr.UsdPhysics = NS(DriveAPI=Drive, RevoluteJoint=object(), RigidBodyAPI=object(), CollisionAPI=object())
    pxr.PhysxSchema = NS(PhysxContactReportAPI=ContactAPI)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    isaac, core, prims = (ModuleType(name) for name in ("isaacsim", "isaacsim.core", "isaacsim.core.prims"))
    prims.SingleRigidPrim = Body
    prims.RigidPrim = RigidView
    isaac.core, core.prims = core, prims
    for module in (isaac, core, prims):monkeypatch.setitem(sys.modules, module.__name__, module)
    utils = ModuleType('isaacsim.core.utils.prims')
    utils.is_prim_non_root_articulation_link = lambda *, prim_path: prim_path.startswith('/World/Robot')
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    stage = NS(GetPrimAtPath=Prim)
    world = NS(scene=NS(add=lambda body: body), reset=lambda: None,
               step=lambda **kwargs: steps.append(kwargs))
    robot = NS(prepare_stage=lambda stage: writes.append(("shared_robot_setup", "robot-owner")),
               attach_cameras=lambda stage: None, articulations=lambda world: {"Left": None, "Right": None},
               reset_to_rest=lambda: None, verify=lambda *args: None)
    ctx = NS(sim=NS(stage=stage, contacts=NS(clear=lambda: None)), world=world, robot=robot)
    binding = dict(pieces=[dict(id="public", path="/World/Pieces/public")],
                   solutions=[dict(slots=[dict(support_paths=["/World/Terrain/Foundation"])])])
    sensors = NativeSensors(ctx, binding)
    assert len(sensors.hands) == 10
    assert len(steps) == 180
    assert not any(path.startswith('/World/Robot') for operation,path in writes if operation=='sdk_sleep')
    assert writes[0] == ("shared_robot_setup", "robot-owner")
    authored = [path for operation, path in writes if operation in ("Apply", "threshold")]
    assert not any(path.startswith("/World/Robot") for path in authored)
    assert {path for operation, path in writes if operation == "Apply"} == {
        "/World/Pieces/public", "/World/Car/Chassis", "/World/StartButton"}
