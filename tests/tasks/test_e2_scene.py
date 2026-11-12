"""USD authoring and measured scene hooks on CPU; no Kit/GPU claim."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json
from mind2act_eval.sim.contacts import ContactLedger
from mind2act_eval.tasks.E2.evidence import finger_contact
from mind2act_eval.tasks.E2.instances import SOURCE_ID, RUBRIC, instruction, validate
from mind2act_eval.tasks.E2.mechanism.case import make_case
from mind2act_eval.tasks.E2.mechanism.clock import display_state
from mind2act_eval.tasks.E2.scene import ArithmeticScene
from mind2act_eval.tasks.E2.scorer import score


def spec(level="easy", seed=2000):
    return InstanceSpec("E2", "synthetic", level, SOURCE_ID, RUBRIC, "test", "v1", seed=seed,
                        clock=ClockSpec(), config={"source_case": make_case(level, seed), "instruction": instruction()},
                        assets={"station": "station.usda", "font": "font.ttf", "x5": "X5.usd"})


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_seeded_training_case_and_canonical_input_binding(level):
    s = spec(level)
    assert validate(s) == make_case(level, 2000)
    suffix = {"easy": "20f63c78109e", "medium": "0ca129abd0d2", "hard": "7bea6776cc75"}[level]
    assert validate(s)["case_id"] == f"e2-arithmetic-{level}-2000-{suffix}"
    for key, value in [("seed", 2001), ("difficulty", "hard" if level != "hard" else "easy"), ("source_id", "other"), ("config_version", "old")]:
        data = to_json(s)
        data[key] = value
        with pytest.raises(ValueError):
            validate(InstanceSpec.from_dict(data))
    data = to_json(s)
    data["config"]["instruction"] += " No attached tool is used."
    with pytest.raises(ValueError, match="canonical"):
        validate(InstanceSpec.from_dict(data))


@pytest.mark.parametrize("tick,phase,angle,expression", [(0,"learn",-105,False), (239,"learn",-105,False), (240,"shuffle",-105,False), (255,"shuffle",0,False), (270,"shuffle",0,True), (299,"shuffle",-98,True), (300,"reasoning",-105,True), (329,"reasoning",-105,True), (330,"input",-105,True)])
def test_training_timeline_has_no_replay_labels(tick, phase, angle, expression):
    actual = display_state(tick, make_case())
    assert actual[0] == phase and actual[2] == expression
    assert actual[1] == pytest.approx(angle)


@pytest.mark.parametrize("other,active,impulse,separation,expected", [
    ("/World/RobotLeft/Asset/link7/collider",True,2e-8,1.,True),
    ("/World/RobotRight/Asset/link8/collider",True,0.,.000029,True),
    ("/World/RobotLeft/Asset/link6/collider",True,1.,-1.,False),
    ("/World/HeldObject/Collision",True,1.,-1.,False),
    ("/World/RobotLeft/Asset/link7/collider",False,1.,-1.,False),
    ("/World/RobotLeft/Asset/link7/collider",True,1e-8,.00003,False),
])
def test_actual_contact_identity_and_original_thresholds(other, active, impulse, separation, expected):
    ledger = ContactLedger()
    cap = "/World/EKeyboard/Key0/Cap"
    ledger.record(cap+"/Collision", other, active=active, impulse_ns=impulse, separation_m=separation)
    assert finger_contact(ledger, cap) is expected


@pytest.fixture(params=["easy", "medium", "hard"])
def authored(tmp_path, monkeypatch, request):
    # usd-core lacks PhysX APIs. Mock only their metadata calls; all USD
    # geometry/joints/materials are real. GPU physics needs native proof.
    import pxr
    class PhysicsMetadata:
        @classmethod
        def Apply(cls, prim):
            return NS(**{name: (lambda value: None) for name in (
                "CreateDisableGravityAttr", "CreateThresholdAttr", "CreateContactOffsetAttr")})
    monkeypatch.setattr(pxr, "PhysxSchema", NS(**{name: PhysicsMetadata for name in (
        "PhysxRigidBodyAPI", "PhysxContactReportAPI", "PhysxCollisionAPI")}), raising=False)
    from mind2act_eval.tasks.E2.mechanism.props import build_props
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, "/World/RobotLeft")
    robot.AddTranslateOp().Set((.3,.2,.1))
    before = stage.GetPrimAtPath("/World/RobotLeft").GetAttribute("xformOp:translate").Get()
    font = Path(__file__).parents[1] / "fixtures" / "fonts" / "DejaVuSans-Bold.ttf"
    case = make_case(request.param, 2000)
    props = build_props(stage, tmp_path, case, font)
    assert stage.GetPrimAtPath("/World/RobotLeft").GetAttribute("xformOp:translate").Get() == before
    return stage, props, case


def test_training_geometry_key_physics_and_private_textures(authored, tmp_path):
    stage, props, case = authored
    for i, xy in enumerate(case["slot_xy_m"]):
        cap = stage.GetPrimAtPath(f"/World/EKeyboard/Key{i}/Cap")
        assert tuple(cap.GetAttribute("xformOp:translate").Get()) == pytest.approx([*xy,.802])
        assert UsdPhysics.MassAPI(cap).GetMassAttr().Get() == pytest.approx(.12)
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"/World/EKeyboard/Key{i}/SpringJoint"))
        assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == pytest.approx((-.008,0))
        drive = UsdPhysics.DriveAPI(joint.GetPrim(), "linear")
        assert (drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get()) == pytest.approx((650.,5.,30.))
        assert (tmp_path / f"digit_{i+1}.png").is_file()
    assert len(props["buttons"]) == 9
    assert not any("Pusher" in str(p.GetPath()) for p in stage.Traverse())
    assert json.loads((tmp_path / "case.private.json").read_text()) == case
    assert UsdGeom.Imageable(props["sequence"].GetPrim()).ComputeVisibility() == "invisible"


def test_real_scene_hooks_produce_independently_scored_evidence(authored, tmp_path):
    stage, props, case = authored
    positions = [[*xy,.802] for xy in case["slot_xy_m"]]
    caps = [NS(get_world_pose=lambda i=i: (positions[i], (1,0,0,0))) for i in range(9)]
    ledger = ContactLedger()
    robot = {"left_arm": (0.,)*6, "right_arm": (0.,)*6, "left_gripper": 0., "right_gripper": 0.}
    ctx = NS(sim=NS(contacts=ledger), robot=NS(observe=lambda: robot), output_dir=tmp_path)
    scene = ArithmeticScene(spec(case["difficulty"]), ctx, case, props, caps)
    tick = 0
    def step(n):
        nonlocal tick
        for _ in range(n):
            tick += 1
            scene.before_step(tick)
            scene.after_step(tick)
    step(330)
    assert scene.observe(tick).instruction == instruction()
    assert "password" not in scene.observe(tick).__dict__
    for digit in case["password"]:
        i = digit - 1
        positions[i][2] = .798
        ledger.record(f"/World/EKeyboard/Key{i}/Cap/Collision", "/World/RobotLeft/Asset/link7/collider", active=True, impulse_ns=1e-6, separation_m=-.001)
        step(3)
        assert not scene.is_complete()
        positions[i][2] = .802
        ledger.clear()
        step(1)
        assert scene.is_complete() is (digit == case["password"][-1])
        step(3)
    assert scene.is_complete()
    assert score(scene.evidence(tick)).value == 100
    assert scene.health().ok
    assert scene.evidence(tick).values["measurements"]["rows"] == tick
    assert len(scene.evidence(tick).values["measurements"]) == 5
    scene.close()
