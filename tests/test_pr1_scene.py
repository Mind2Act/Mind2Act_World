import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind2act_eval.contracts import BuildContext, ConsumerKind, InstanceSpec, NativeEvidence
from mind2act_eval.tasks import TaskRegistry

pxr = pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from mind2act_eval.tasks import PR1  # noqa: E402
from mind2act_eval.tasks.PR1 import mechanism, scene as pr1_scene  # noqa: E402
from mind2act_eval.tasks.PR1.instances import spec_from_row  # noqa: E402
from mind2act_eval.tasks.PR1.scorer import score  # noqa: E402

FIX = Path(__file__).parent / 'fixtures'


class FontAssets:
    ids = {'font_sans': 'DejaVuSans.ttf', 'font_sans_bold': 'DejaVuSans-Bold.ttf'}

    def __init__(self, missing=()):
        self.missing = set(missing)

    def resolve(self, asset_id):
        if asset_id in self.missing:
            raise FileNotFoundError(asset_id)
        return FIX / 'fonts' / self.ids[asset_id]


def spec(**over):
    row = json.loads((FIX / 'pr1_row_synthetic.json').read_text())
    s = spec_from_row(row, asset_version='assets-test')
    return InstanceSpec.from_dict({**json.loads(json.dumps(PR1_dict(s))), **over}) if over else s


def PR1_dict(s):
    from mind2act_eval.contracts import to_json
    return to_json(s)


class FakeContacts:
    touching = False

    def touch(self, root, other=None, robot_only=False, link=None):
        assert robot_only and root == mechanism.CARRIAGE
        return self.touching


class FakeRobot:
    def observe(self):
        return dict(left_arm=[0] * 6, right_arm=[0] * 6, left_gripper=1., right_gripper=1., rgb={})


@pytest.fixture
def built(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    world = SimpleNamespace(stage=stage, contacts=FakeContacts())
    ctx = BuildContext(sim=SimpleNamespace(stage=stage, world=world, contacts=FakeContacts()), robot=FakeRobot(), assets=FontAssets(), output_dir=tmp_path)
    return stage, ctx, pr1_scene.build(spec(), ctx)


def evidence(values, hz=120, target=10., consecutive=True):
    return NativeEvidence('PR1', 'i', len(values), dict(tick_hz=hz, target_value=target, first_tick=1,
                                                        values=tuple(values), consecutive=consecutive))


# ---- scene ------------------------------------------------------------------------------------
def test_initial_carriage_pose_follows_the_instance(built):
    stage, ctx, sc = built
    read = sc._read()
        # synthetic instance: 13.3 ohm -> x = -0.2 + 0.4*13.3/40; y = 0.084 - 0.07 + fixture offset -0.045; z = 0.768 + 0.097
    assert read == pytest.approx((-0.067, -0.031, 0.865), abs=1e-6)
    assert sc.health().ok is True and sc.is_complete() is False


def test_ruler_texture_exists_and_scene_points_at_it(built):
    stage, ctx, sc = built
    prim = stage.GetPrimAtPath(mechanism.ROOT + '/Scale')
    assert prim and prim.GetAttribute('primvars:st').HasValue()
    asset = stage.GetPrimAtPath('/World/Materials/PR1v3_Ruler/Texture').GetAttribute('inputs:file').Get()
    texture = Path(asset.path)
    assert texture.is_file() and texture.stat().st_size > 5000 and texture.parent.is_relative_to(ctx.output_dir)


def test_slider_physics_is_a_damped_dynamic_body(built):
    stage, ctx, sc = built
    car = stage.GetPrimAtPath(mechanism.CARRIAGE)
    assert UsdPhysics.RigidBodyAPI(car).GetRigidBodyEnabledAttr().Get() is True
    assert UsdPhysics.RigidBodyAPI(car).GetKinematicEnabledAttr().Get() is False
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(mechanism.JOINT))
    assert joint.GetJointEnabledAttr().Get() is True and joint.GetAxisAttr().Get() == 'X'
    drive = UsdPhysics.DriveAPI.Get(joint.GetPrim(), 'linear')
    assert (drive.GetDampingAttr().Get(), drive.GetStiffnessAttr().Get(), drive.GetMaxForceAttr().Get()) == (1.0, 0.0, 5.0)
    assert sc.colliders and all(c.startswith(mechanism.CARRIAGE + '/') for c in sc.colliders)
    assert not stage.GetPrimAtPath(mechanism.ROOT + '/ResistanceWindingMesh').HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath(mechanism.ROOT + '/Scale').HasAPI(UsdPhysics.CollisionAPI)
    friction = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(mechanism.CONTACT_MATERIAL))
    assert (friction.GetStaticFrictionAttr().Get(), friction.GetDynamicFrictionAttr().Get()) == pytest.approx((0.9, 0.7))


def test_no_robot_prims_or_robot_settings_in_the_task_module(built):
    stage, ctx, sc = built
    assert not [p for p in stage.Traverse() if '/Robot' in str(p.GetPath())]
    source = Path(mechanism.__file__).read_text() + Path(pr1_scene.__file__).read_text()
    for banned in ('ArticulationAction', 'set_joint', 'x5_runtime', 'mind2act_env', 'import physco', 'from physco'):
        assert banned not in source


def test_held_target_reading_scores_immediately_without_release(built):
    stage, ctx, sc = built
    contacts = ctx.sim.contacts
    car = UsdGeom.Xformable(stage.GetPrimAtPath(mechanism.CARRIAGE)).GetOrderedXformOps()[0]
    contacts.touching = True
    for tick in range(1, 40):                       # pushed in from 13.3 ohm
        car.Set(Gf.Vec3d(-0.067 - 0.064 * tick / 39, 0.084, 0.865))
        sc.before_step(tick)
        sc.after_step(tick)
    car.Set(Gf.Vec3d(-0.131, 0.084, 0.865))          # 6.9 ohm
    for tick in range(40, 140):                     # held in the gripper: never counts
        sc.after_step(tick)
    assert sc.is_complete() is True and score(sc.evidence(139)).value == 100.
    contacts.touching = False
    for tick in range(140, 200):
        sc.after_step(tick)
    assert sc.is_complete() is True                 # no released-hold gate
    sc.after_step(200)
    assert sc.is_complete() is True
    ev = sc.evidence(200)
    assert len(ev.values['values']) == 61 and ev.values['consecutive'] is True and not any(ev.values['touching'])
    assert score(ev).value == 100.


def test_skipped_tick_is_loud_and_marks_the_record(built):
    stage, ctx, sc = built
    sc.after_step(1)
    with pytest.raises(ValueError):
        sc.after_step(3)
    assert sc.evidence(1).values['consecutive'] is False


def test_evidence_for_an_unmeasured_tick_is_not_consecutive(built):
    stage, ctx, sc = built
    sc.after_step(1)
    assert sc.evidence(2).values['consecutive'] is False


def test_observation_carries_instruction_and_robot_state(built):
    stage, ctx, sc = built
    obs = sc.observe(0)
    assert obs.instruction == ('Adjust the resistance between the left terminal and wiper to 6.9 ohm, '
                               'then release the slider and keep that reading for 0.5 seconds.')


def test_start_must_match_instance(built, tmp_path):
    stage, ctx, sc = built
    UsdGeom.Xformable(stage.GetPrimAtPath(mechanism.CARRIAGE)).GetOrderedXformOps()[0].Set(Gf.Vec3d(0., 0.084, 0.865))
    fresh = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(fresh, '/World')
    bad = spec(config=dict(initial_ohm=20., target_ohm=10., fixture_y_offset_m=-.05))
    ctx2 = BuildContext(sim=SimpleNamespace(stage=fresh, contacts=FakeContacts()), robot=FakeRobot(), assets=FontAssets(), output_dir=tmp_path / 'b')
    assert pr1_scene.build(bad, ctx2)._read()[0] == pytest.approx(0.0, abs=1e-6)   # 20 ohm sits at x=0


def test_registry_refuses_difficulty_substitution(tmp_path):
    registry = TaskRegistry([PR1.DEFINITION])
    easy = spec()
    hard = InstanceSpec.from_dict({**PR1_dict(easy), 'difficulty': 'hard'})
    with pytest.raises(ValueError, match='binding'):
        registry.build(hard, easy, None)


@pytest.mark.parametrize('config', [dict(initial_ohm=1., target_ohm=10., fixture_y_offset_m=-.05),
                                    dict(initial_ohm=15., target_ohm=39., fixture_y_offset_m=-.05),
                                    dict(initial_ohm=15., target_ohm=10.)])
def test_invalid_instances_are_rejected_before_building(config):
    bad = InstanceSpec.from_dict({**PR1_dict(spec()), 'config': config})
    with pytest.raises(ValueError):
        pr1_scene.validate(bad)


def test_row_conversion_drops_expert_only_fields():
    row = json.loads((FIX / 'pr1_row_synthetic.json').read_text())
    s = spec_from_row(row, asset_version='a')
    assert set(s.config) == {'initial_ohm', 'target_ohm', 'fixture_y_offset_m'}
    assert (s.task_id, s.instance_id, s.difficulty) == ('PR1', 'PR1_synthetic_easy_0001', 'easy')



def test_missing_font_asset_is_an_asset_error_not_a_default_font(tmp_path):
    from mind2act_eval.tasks._precision.textures import AssetError
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    ctx = BuildContext(sim=SimpleNamespace(stage=stage, contacts=FakeContacts()), robot=FakeRobot(),
                       assets=FontAssets(missing=['font_sans_bold']), output_dir=tmp_path)
    with pytest.raises(AssetError, match='font_sans_bold'):
        pr1_scene.build(spec(), ctx)
    assert not (tmp_path / 'PR1' / 'ruler_ohm.png').exists()
    undeclared = InstanceSpec.from_dict({**PR1_dict(spec()), 'assets': {}})
    with pytest.raises(AssetError, match='does not declare'):
        pr1_scene.build(undeclared, ctx)


def test_same_instance_renders_the_same_scale_pixels(tmp_path):
    from PIL import Image, ImageChops
    images = []
    for name in ('a', 'b'):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/World')
        ctx = BuildContext(sim=SimpleNamespace(stage=stage, contacts=FakeContacts()), robot=FakeRobot(), assets=FontAssets(), output_dir=tmp_path / name)
        pr1_scene.build(spec(), ctx)
        images.append(Image.open(tmp_path / name / 'PR1' / 'ruler_ohm.png').convert('RGB'))
    assert ImageChops.difference(*images).getbbox() is None


def test_observation_carries_the_declared_protocol_version(built, tmp_path):
    stage, ctx, sc = built
    assert sc.observe(0).protocol_version == 'v1'
    other = InstanceSpec.from_dict({**PR1_dict(spec()), 'protocol_version': 'v2'})
    stage2 = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage2, '/World')
    ctx2 = BuildContext(sim=SimpleNamespace(stage=stage2, contacts=FakeContacts()), robot=FakeRobot(), assets=FontAssets(), output_dir=tmp_path / 'p')
    assert pr1_scene.build(other, ctx2).observe(0).protocol_version == 'v2'


def test_evidence_is_a_bounded_trailing_window_and_full_series_is_dumped(built):
    stage, ctx, sc = built
    for tick in range(1, 501):
        sc.after_step(tick)
        assert len(sc.evidence(tick).values['values']) <= 61
    sc.close()
    dumped = json.loads((ctx.output_dir / 'PR1' / 'measurements.json').read_text())
    assert len(dumped['values']) == 500 and dumped['first_tick'] == 1


def test_runs_through_t0_runner_after_transient_goal_is_met(tmp_path):
    """Fake simulator pushes the carriage to the target, releases at tick 31; real runner, scene and scorer."""
    from mind2act_eval.runner import EpisodeLimits, run_episode
    from mind2act_eval.contracts import Action, Termination
    from mind2act_eval.tasks import TaskDefinition

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')

    class Sim:
        world = None

        def __init__(self):
            self.stage, self.ticks, self.contacts = stage, 0, FakeContacts()

        def step(self, *, render):
            self.ticks += 1
            op = UsdGeom.Xformable(stage.GetPrimAtPath(mechanism.CARRIAGE)).GetOrderedXformOps()[0]
            op.Set(Gf.Vec3d(max(-0.131, -0.067 - 0.064 * self.ticks / 30), 0.084, 0.865))   # 13.3 -> 6.9 ohm in 30 ticks
            self.contacts.touching = self.ticks <= 30

        def close(self):
            pass

    class Adapter:
        def reset(self, seed): pass
        def act(self, observation):
            return Action([0] * 6, [0] * 6, 1., 1.)
        def close(self): pass

    ctx = BuildContext(sim=Sim(), robot=SimpleNamespace(observe=FakeRobot().observe, apply=lambda a: None),
                       assets=FontAssets(), output_dir=tmp_path)
    task = TaskDefinition('PR1', pr1_scene.build, score)
    result = run_episode(spec(), task, ctx, Adapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(400, 60))
    assert result.infra_valid, result.error
    assert result.score == 100. and result.termination == Termination.PHYSICS_BUDGET
    assert result.ticks == 400  # no success early stop
    assert (tmp_path / 'PR1' / 'measurements.json').is_file()
