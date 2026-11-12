import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from mind2act_eval.contracts import BuildContext, ConsumerKind, InstanceSpec, to_json  # noqa: E402
from mind2act_eval.runner import EpisodeLimits, run_episode  # noqa: E402
from mind2act_eval.tasks import TaskDefinition  # noqa: E402
from mind2act_eval.tasks.PR3 import scene as pr3_scene  # noqa: E402
from mind2act_eval.tasks.PR3.instances import spec_from_row  # noqa: E402
from mind2act_eval.tasks.PR3.mechanism import marker, paths  # noqa: E402
from mind2act_eval.tasks.PR3.scorer import score  # noqa: E402
from pr3_support import FakeBody, FakeContacts, FakeRobot, FakeSim, prepared_scene  # noqa: E402

ROW = {'task': 'PR3', 'difficulty': 'medium',
       'source_case': {'case_id': 'PR3_synthetic_medium_0001', 'task': 'PR3',
                       'parameters': {'level': 'medium', 'mirror': False, 'angle_delta_deg': 5.0,
                                      'offset_xy_mm': [3.0, -7.5], 'writing_yaw_deg': 90.0}}}


class Assets:
    def __init__(self, scene, missing=False):
        self.scene, self.missing = scene, missing

    def resolve(self, asset_id):
        if self.missing:
            raise FileNotFoundError(asset_id)
        return self.scene


@pytest.fixture
def parts(tmp_path, monkeypatch):
    scene_file = prepared_scene(tmp_path / 'scene.usda')
    contacts, robot = FakeContacts(), FakeRobot()
    sim = FakeSim(contacts)
    bodies = {}
    monkeypatch.setattr(pr3_scene, '_isaac_bodies', lambda world, ps: bodies.update({p: FakeBody((0., -.171, .794)) for p in ps}) or bodies)
    (tmp_path / 'out').mkdir()
    ctx = BuildContext(sim=sim, robot=robot, assets=Assets(scene_file), output_dir=tmp_path / 'out')
    spec = spec_from_row(ROW, scene_path='PR3/scene.usda', direction_gate=False, asset_version='test')
    return spec, ctx, sim, contacts, robot, bodies


def test_scene_physics_matches_the_baseline_station_checks(parts):
    spec, ctx, sim, contacts, robot, bodies = parts
    sc = pr3_scene.build(spec, ctx)
    stage = sim.stage
    nib = stage.GetPrimAtPath('/World/MarkerNib')
    assert UsdPhysics.RigidBodyAPI(nib).GetRigidBodyEnabledAttr().Get() is True
    assert UsdPhysics.RigidBodyAPI(nib).GetKinematicEnabledAttr().Get() is False
    assert UsdGeom.Sphere(stage.GetPrimAtPath('/World/MarkerNib/Tip')).GetRadiusAttr().Get() == pytest.approx(.0025)
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/MarkerNibSpring'))
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), 'linear')
    assert joint.GetAxisAttr().Get() == 'X'
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == pytest.approx((-.003, .003))
    assert (drive.GetTargetPositionAttr().Get(), drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(),
            drive.GetMaxForceAttr().Get()) == pytest.approx((-.003, 40., .2, .3))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        stage.GetPrimAtPath(marker.BOARD)).ComputeAlignedRange()
    assert box.GetMax()[2] == pytest.approx(.7745, abs=1e-6)
    for root in ['/World/Worktable', '/World/TableFrame', '/World/CameraStand', '/World/PedestalLeft',
                 '/World/PedestalRight', '/World/Floor', '/World/MarkerDynamic/Colliders/Barrel', marker.BOARD]:
        assert any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(stage.GetPrimAtPath(root))), root
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath('/World/MarkerDynamic')).GetKinematicEnabledAttr().Get() is False


def test_only_selected_level_is_visible_and_marker_visuals_move_onto_the_body(parts):
    spec, ctx, sim, *_ = parts
    pr3_scene.build(spec, ctx)
    stage = sim.stage
    vis = {d: UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/Marker_{d}')).ComputeVisibility() for d in ('easy', 'medium', 'hard')}
    assert vis['medium'] != 'invisible' and vis['easy'] == vis['hard'] == 'invisible'
    assert stage.GetPrimAtPath('/World/MarkerDynamic/Visuals/MarkerBarrel')
    assert not stage.GetPrimAtPath('/World/Marker_medium/MarkerBarrel')


def test_reference_texture_is_generated_and_bound(parts):
    spec, ctx, sim, *_ = parts
    pr3_scene.build(spec, ctx)
    texture = ctx.output_dir / 'PR3' / 'reference_faint.png'
    assert texture.stat().st_size > 1000
    shader = sim.stage.GetPrimAtPath('/World/Marker_medium/PaperMat/Tex')
    assert Path(shader.GetAttribute('inputs:file').Get().path) == texture.resolve()


def test_robot_is_shared_and_task_touches_no_robot_prim(parts):
    spec, ctx, sim, contacts, robot, bodies = parts
    pr3_scene.build(spec, ctx)
    assert robot.prepared and robot.initialized == 1
    assert not [p for p in sim.stage.Traverse() if str(p.GetPath()).startswith('/World/Robot') and p.GetAttributes()
                and any('physics' in a.GetName() or 'physx' in a.GetName() for a in p.GetAttributes())]
    source = '\n'.join(Path(m.__file__).read_text() for m in (pr3_scene, marker))
    for banned in ('ArticulationAction', 'from mind2act_env', 'import mind2act_env', 'case.dependencies', 'import ast', 'set_joint', 'DriveAPI'):
        assert banned not in source


def test_missing_scene_asset_fails_without_fallback(parts):
    spec, ctx, *_ = parts
    bad = BuildContext(sim=ctx.sim, robot=ctx.robot, assets=Assets(None, missing=True), output_dir=ctx.output_dir)
    with pytest.raises(FileNotFoundError):
        pr3_scene.build(spec, bad)


def test_spec_validation():
    spec = spec_from_row(ROW, scene_path='x', direction_gate=False, asset_version='a')
    pr3_scene.validate(spec)
    for change in ({'difficulty': 'hard'}, {'source_id': 'other'}, {'assets': {}}, {'config': {'level': 'medium'}}, {'config': {**to_json(spec)['config'], 'direction_gate': 'yes'}}):
        with pytest.raises(ValueError):
            pr3_scene.validate(InstanceSpec.from_dict({**to_json(spec), **change}))
    with pytest.raises(ValueError, match='Hz'):
        pr3_scene.validate(InstanceSpec.from_dict({**to_json(spec), 'clock': {'physics_hz': 240, 'control_hz': 30, 'rgb_hz': 30}}))


def test_trace_through_the_t0_runner_scores_100(parts):
    spec, ctx, sim, contacts, robot, bodies = parts
    target = paths.reference(dict(level='medium', mirror=False, angle_delta_deg=5.0, offset_xy_mm=[3.0, -7.5])) / 1000.
    track = [(x, y + .065) for x, y in target[::24]]            # ~2.9 mm per tick along the 350 mm path
    first = 10

    def on_step(tick):
        i = tick - first
        if 0 <= i < len(track):
            contacts.touching = True
            bodies['/World/MarkerNib'].position = np.array([track[i][0], track[i][1], .7745 + .0025])
        else:
            contacts.touching = False
    sim.on_step = on_step

    class Adapter:
        def reset(self, seed): pass
        def act(self, observation):
            from mind2act_eval.contracts import Action
            return Action([0] * 6, [0] * 6, 1., 1.)
        def close(self): pass

    task = TaskDefinition('PR3', pr3_scene.build, score)
    result = run_episode(spec, task, ctx, Adapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(first + len(track) + 140, 120))
    assert result.infra_valid, result.error
    assert result.score == 100.
    saved = json.loads((ctx.output_dir / 'score.json').read_text())
    assert saved['components']['coverage'] > .98
    result = json.loads((ctx.output_dir / 'result.json').read_text())
    assert result['task_metadata'] == {'direction_gate': False, 'normal_completion_settle_ticks': 120}
    assert saved['components']['task_success'] is True
