import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind2act_eval.contracts import BuildContext, InstanceSpec, to_json

pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from mind2act_eval.tasks.PR2 import mechanism, scene as pr2_scene  # noqa: E402
from mind2act_eval.tasks.PR2.instances import spec_from_row  # noqa: E402
from mind2act_eval.tasks.PR2.scorer import score  # noqa: E402
from mind2act_eval.tasks._precision.textures import AssetError  # noqa: E402

FIX = Path(__file__).parent / 'fixtures'


class FontAssets:
    ids = {'font_sans': 'DejaVuSans.ttf', 'font_sans_bold': 'DejaVuSans-Bold.ttf'}

    def __init__(self, missing=()):
        self.missing = set(missing)

    def resolve(self, asset_id):
        if asset_id in self.missing:
            raise FileNotFoundError(asset_id)
        return FIX / 'fonts' / self.ids[asset_id]


class FakeContacts:
    touching = False

    def touch(self, root, other=None, robot_only=False, link=None):
        assert robot_only and root == mechanism.KNOB
        return self.touching


class FakeRobot:
    def prepare_stage(self, stage):
        self.stage_prepared = True

    def apply_task_collision_overlay(self, stage, descriptor):
        assert getattr(self, 'stage_prepared', False), 'source contacts need the shared prepared stage before native init'
        self.contact_descriptor = descriptor

    def verify_task_collision_overlay(self, stage):
        pass

    def observe(self):
        return dict(left_arm=[0] * 6, right_arm=[0] * 6, left_gripper=1., right_gripper=1., rgb={})


def spec(**over):
    s = spec_from_row(json.loads((FIX / 'pr2_row_synthetic.json').read_text()), asset_version='assets-test')
    return InstanceSpec.from_dict({**to_json(s), **over}) if over else s


def context(tmp_path, assets=None):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    return stage, BuildContext(sim=SimpleNamespace(stage=stage, contacts=FakeContacts()), robot=FakeRobot(),
                               assets=assets or FontAssets(), output_dir=tmp_path)


@pytest.fixture
def built(tmp_path):
    stage, ctx = context(tmp_path)
    return stage, ctx, pr2_scene.build(spec(), ctx)


def test_initial_knob_follows_the_instance(built):
    stage, ctx, sc = built
    knob = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(mechanism.KNOB))
    # synthetic instance: dial centre y 0.106 + shift (-0.04 + offset 0.005), z = 0.768 + 0.1362 + 0.007
    assert tuple(knob.ExtractTranslation()) == pytest.approx((0., 0.071, 0.9112), abs=1e-6)
    assert sc._read() == pytest.approx(150. - 103.7 * 300. / 260., abs=1e-4)
    assert sc._model.volts(sc._read()) == pytest.approx(103.7, abs=1e-4)
    assert sc.health().ok is True and sc.is_complete() is False


def test_every_generated_texture_exists_and_is_referenced(built):
    stage, ctx, sc = built
    for name in mechanism.TEXTURES:
        assert (ctx.output_dir / 'PR2' / name).stat().st_size > 500
    files = [p.GetAttribute('inputs:file').Get().path for p in stage.Traverse() if p.GetAttribute('inputs:file')]
    assert len(files) == 4 and all(Path(f).is_file() for f in files)
    assert any(f.endswith('voltage_dial.png') for f in files)


def test_knob_is_a_dynamic_body_on_a_light_revolute_drive(built):
    stage, ctx, sc = built
    body = stage.GetPrimAtPath(mechanism.KNOB)
    assert UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is False
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(mechanism.JOINT))
    assert joint.GetJointEnabledAttr().Get() is True and joint.GetAxisAttr().Get() == 'Z'
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == (-150, 150)
    drive = UsdPhysics.DriveAPI.Get(joint.GetPrim(), 'angular')
    assert (drive.GetDampingAttr().Get(), drive.GetStiffnessAttr().Get(), drive.GetMaxForceAttr().Get()) == pytest.approx((0.02, 0., 5.))
    assert sc.colliders and all(c.startswith(mechanism.KNOB + '/') for c in sc.colliders)
    assert not stage.GetPrimAtPath(mechanism.ROOT + '/VoltageScale').HasAPI(UsdPhysics.CollisionAPI)


def test_no_robot_prims_or_robot_settings(built):
    stage, ctx, sc = built
    assert not [p for p in stage.Traverse() if '/Robot' in str(p.GetPath())]
    assert ctx.robot.contact_descriptor['task_id'] == 'PR2'
    assert len(ctx.robot.contact_descriptor['colliders']) == 4
    source = Path(mechanism.__file__).read_text() + Path(pr2_scene.__file__).read_text()
    for banned in ('ArticulationAction', 'set_joint', 'x5_runtime', 'mind2act_env', 'import physco', 'from physco'):
        assert banned not in source


def test_source_bearing_filter_keeps_robot_and_other_fixture_contacts(built):
    stage, ctx, sc = built
    body = stage.GetPrimAtPath(mechanism.KNOB)
    # The source precision factory filters this intentional shaft/hub overlap,
    # not contacts between the gripper and either part of the apparatus.
    assert body.HasAPI(UsdPhysics.FilteredPairsAPI)
    assert list(UsdPhysics.FilteredPairsAPI(body).GetFilteredPairsRel().GetTargets()) == [
        mechanism.ROOT + '/Shaft']
    shaft = stage.GetPrimAtPath(mechanism.ROOT + '/Shaft')
    assert UsdPhysics.CollisionAPI(shaft).GetCollisionEnabledAttr().Get() is True
    assert not shaft.HasAPI(UsdPhysics.FilteredPairsAPI)
    assert all(UsdPhysics.CollisionAPI(stage.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get()
               for p in sc.colliders)


def test_held_final_voltage_at_target_scores_without_waiting(built):
    stage, ctx, sc = built
    contacts = ctx.sim.contacts
    op = UsdGeom.Xformable(stage.GetPrimAtPath(mechanism.KNOB)).GetOrderedXformOps()[1]
    contacts.touching = True
    for tick in range(1, 30):
        op.Set(30.346153 - 0.5 * tick)
        sc.before_step(tick)
        sc.after_step(tick)
    op.Set(sc._model.yaw_deg(151.3))
    for tick in range(30, 200):                      # held in the gripper: never counts
        sc.after_step(tick)
    assert sc.is_complete() is True
    contacts.touching = False
    for tick in range(200, 320):
        sc.after_step(tick)
    assert sc.is_complete() is True                  # no release/hold gate
    sc.after_step(320)
    assert sc.is_complete() is True
    ev = sc.evidence(320)
    assert len(ev.values['values']) == 121 and score(ev).value == 100.
    assert ev.values['values'][-1] == pytest.approx(151.3, abs=1e-6)


def test_hard_uses_the_directional_source_and_other_sources_are_refused(tmp_path):
    row = json.loads((FIX / 'pr2_row_synthetic.json').read_text())
    row['difficulty'] = 'hard'
    row['source_case']['parameters']['difficulty'] = 'hard'
    hard = spec_from_row(row, asset_version='a')
    assert hard.source_id == 'pr2-hard-directional-grasp-run925de9-v1'
    assert spec().source_id == 'pr2-em-edge-side-grasp-run925de9-v1'
    wrong = InstanceSpec.from_dict({**to_json(hard), 'source_id': 'pr2-em-edge-side-grasp-run925de9-v1'})
    with pytest.raises(ValueError, match='baseline'):
        pr2_scene.validate(wrong)


def test_missing_font_asset_is_an_asset_error(tmp_path):
    stage, ctx = context(tmp_path, FontAssets(missing=['font_sans']))
    with pytest.raises(AssetError, match='font_sans'):
        pr2_scene.build(spec(), ctx)
    assert not (tmp_path / 'PR2' / 'voltage_dial.png').exists()


@pytest.mark.parametrize('config', [dict(initial_volt=-5., target_volt=100., fixture_y_offset_m=0.),
                                    dict(initial_volt=100., target_volt=300., fixture_y_offset_m=0.),
                                    dict(initial_volt=100., target_volt=120.)])
def test_invalid_instances_are_rejected(config):
    with pytest.raises(ValueError):
        pr2_scene.validate(InstanceSpec.from_dict({**to_json(spec()), 'config': config}))


def test_observation_protocol_and_instruction(tmp_path):
    stage, ctx = context(tmp_path)
    sc = pr2_scene.build(spec(protocol_version='v2'), ctx)
    obs = sc.observe(0)
    assert obs.protocol_version == 'v2'
    assert obs.instruction == 'Adjust the output voltage to 151.3 V, then release the knob and keep that reading for 1 second.'
