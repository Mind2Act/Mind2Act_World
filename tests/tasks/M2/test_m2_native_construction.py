"""CPU substitutes validate the actual native constructor and task drive owner."""
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from mind2act_eval.tasks.M2.scene import NativeSensors


@pytest.mark.parametrize("count", [4, 8, 16])
def test_native_setup_and_hinge_writes_remain_task_owned(monkeypatch, count):
    monkeypatch.setattr("mind2act_eval.tasks.M2.native_stage.check_stage", lambda stage, n: {"cards": n})
    writes, steps = [], []
    class Prim:
        def __init__(self, path):self.path = path
    class Body:
        def __init__(self, prim_path, name):self.path = prim_path
    class Drive:
        def __init__(self, prim, kind):self.prim = prim
        def GetTargetPositionAttr(self):
            return NS(Set=lambda value: writes.append(("target_deg", self.prim.path, value)))
    class ContactAPI:
        @staticmethod
        def Apply(prim):
            writes.append(("contact", prim.path, None))
            return NS(CreateThresholdAttr=lambda value: None)
    pxr = ModuleType("pxr")
    identity = NS(GetReal=lambda:1., GetImaginary=lambda:(0.,0.,0.))
    joint = NS(GetLocalRot0Attr=lambda:NS(Get=lambda:identity), GetLocalRot1Attr=lambda:NS(Get=lambda:identity))
    pxr.UsdPhysics = NS(DriveAPI=Drive, RevoluteJoint=lambda prim:joint)
    pxr.UsdGeom = NS(Imageable=lambda prim: NS(MakeVisible=lambda: writes.append(("visible", prim.path, None))))
    pxr.PhysxSchema = NS(PhysxContactReportAPI=ContactAPI)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    isaac, core, prims = (ModuleType(name) for name in ("isaacsim", "isaacsim.core", "isaacsim.core.prims"))
    prims.SingleRigidPrim = Body
    isaac.core, core.prims = core, prims
    for module in (isaac, core, prims):monkeypatch.setitem(sys.modules, module.__name__, module)
    stage = NS(GetPrimAtPath=Prim)
    world = NS(scene=NS(add=lambda body: body), reset=lambda: None,
               step=lambda **kwargs: steps.append(kwargs))
    robot = NS(prepare_stage=lambda stage: writes.append(("shared_setup", "robot-owner", None)),
               attach_cameras=lambda stage: None, articulations=lambda world: {},
               reset_to_rest=lambda: None, verify=lambda *args: None)
    ctx = NS(sim=NS(stage=stage, contacts=NS(clear=lambda: None)), world=world, robot=robot)
    sensors = NativeSensors(ctx, dict(card_identities=[i//2 for i in range(count)], button_rest_m=[1.]*count))
    assert len(steps) == 240
    assert writes[0][0] == "shared_setup"
    assert all(path.startswith("/World/Memory/") for kind, path, _ in writes if kind == "contact")
    targets = [2., 2.] + [178.]*(count-2)
    sensors.set_card_targets(targets, {0, 1})
    assert [value for kind, _, value in writes if kind == "target_deg"] == targets
    assert all(path.startswith("/World/Memory/") for kind, path, _ in writes if kind != "shared_setup")
