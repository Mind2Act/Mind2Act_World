import hashlib
import json
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest

from mind2act_eval.contracts import BuildContext
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.PR5.instances import CONFIG_VERSION, SOURCE_REVISION, digest, spec_from_source, validate
from mind2act_eval.tasks.PR5 import scene
from mind2act_eval.tasks.PR5.mechanism.contacts import ContactReadings


def source_row():
    case = dict(id='medium/train/000', level='medium', split='train', index=0, design=CONFIG_VERSION,
        stamps=[dict(id='A', card_xy=[0., .025], holder_xy=[.02, -.2], box_wh=[.08, .08],
                     target_yaw=-10., yaw_tol=2., arm='Right', press_azimuth_deg=None)])
    return dict(task_id='PR5', difficulty='medium', case_id='PR5-medium-train-000-source-semantic-coverage-v1',
        binding=dict(source_revision=SOURCE_REVISION, config_sha256=digest(case),
                     source_seed=dict(split='train', index=0), declared_version=CONFIG_VERSION, native_task_id='PR5'),
        config=dict(source_case=case, scene_sha256='0'*64, native_inputs=dict(manifest=dict(asset_closure=[
            dict(path='prepared/'+name, sha256='1'*64) for name in
            ('scene-config.json', 'source-authoring.json', 'ink_clean.png', 'ink_smeared.png')]))))


ASSETS = dict(station='prepared/scene.usda', x5='robot/ARX.usd', scene_config='prepared/scene-config.json',
    authoring_receipt='prepared/source-authoring.json', ink_clean='prepared/textures/ink_clean.png',
    ink_smeared='prepared/textures/ink_smeared.png')


def test_source_binding_preserves_instance_seed_and_delivered_instruction():
    spec = spec_from_source(source_row(), assets=ASSETS, asset_version='test')
    assert spec.instance_id == source_row()['case_id'] and spec.seed == 0
    assert validate(spec)['split'] == 'train'
    assert spec.clock.physics_hz == 120 and spec.clock.control_hz == 30
    assert spec.config['instruction'].startswith('Pick up square stamp A and stamp once, fully inside the blue box')
    serialized = json.dumps(dict(spec.config), default=dict)
    assert 'native_inputs' not in serialized and 'annotation' not in serialized
    with pytest.raises(ValueError, match='difficulty'):
        validate(replace(spec, difficulty='hard'))


def test_source_rest_reaches_the_real_shared_robot_configuration_without_kit():
    from mind2act_eval.robot import X5Robot
    import numpy as np
    spec = spec_from_source(source_row(), assets=ASSETS, asset_version='test')
    robot = X5Robot(rest=spec.config.get('robot_rest'))
    expected = np.r_[np.radians([0.,65.,100.,-65.,0.,0.]),.044,.044]
    assert robot.rest_pose() == pytest.approx(expected)
    config = dict(spec.config, robot_rest=[0.]*8)
    with pytest.raises(ValueError, match='reset joint'):
        validate(replace(spec, config=config))


def test_source_seed_hash_and_asset_root_cannot_be_substituted():
    row = source_row()
    row['config']['source_case']['index'] = 1
    with pytest.raises(ValueError, match='binding'):
        spec_from_source(row, assets=ASSETS, asset_version='test')
    with pytest.raises(ValueError, match='asset-library'):
        spec_from_source(source_row(), assets=dict(ASSETS, station='/former/host/scene.usda'), asset_version='test')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    for path in (scene.STAMP, '/World/Task/CardA', '/World/Task/HolderA', '/World/Task/Board'):
        UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(scene.STAMP)).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(scene.STAMP)).CreateMassAttr(.100)
    row = source_row()
    pin = json.loads(resources.files('mind2act_eval.tasks.PR5').joinpath('config/source.json').read_text())
    payloads = dict(station=stage.GetRootLayer().ExportToString(), x5='synthetic robot placeholder',
        scene_config=json.dumps(dict(seed=row['config']['source_case'], face_side_m=.050, plate_top_m=.772, card_top_m=.7768)),
        authoring_receipt=json.dumps(dict(source_revision=SOURCE_REVISION,
            scene_author_sha256=pin['scene_author_sha256'], common_sha256=pin['common_sha256'])),
        ink_clean='synthetic clean texture', ink_smeared='synthetic smeared texture')
    for key, value in payloads.items():
        path = tmp_path / ASSETS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    row['config']['scene_sha256'] = hashlib.sha256(payloads['station'].encode()).hexdigest()
    row['config']['native_inputs']['manifest']['asset_closure'] = [
        dict(path=ASSETS[key], sha256=hashlib.sha256(payload.encode()).hexdigest())
        for key, payload in payloads.items()]
    spec = spec_from_source(row, assets=ASSETS, asset_version='synthetic')
    class Body:
        pose = None
        def initialize(self): pass
        def set_world_pose(self, p, q): self.pose = (p, q)
        def set_linear_velocity(self, v): assert v == [0., 0., 0.]
        def set_angular_velocity(self, v): assert v == [0., 0., 0.]
        def get_world_pose(self): return self.pose
    class Robot:
        prepared = initialized = 0
        def prepare_stage(self, s): self.prepared += 1
        def initialize(self, w): self.initialized += 1
        def observe(self):
            return dict(left_arm=[0.]*6, right_arm=[0.]*6, left_gripper=1., right_gripper=1.)
    class Sim:
        world = object()
        steps = 0
        def step(self, *, render):
            assert render is False
            self.steps += 1
    class Subscription:
        closed = False
        def unsubscribe(self): self.closed = True
    class Ink:
        def __init__(self, *args): pass
        def add(self, marks): pass
    sim = Sim()
    sim.stage = stage
    robot, body, subscription = Robot(), Body(), Subscription()
    from mind2act_eval.tasks.PR5.mechanism import contacts, visuals
    monkeypatch.setattr(scene, '_stamp_body', lambda world: body)
    monkeypatch.setattr(scene, '_tcp_positions', lambda stage: {'Left': [0., 0., 1.], 'Right': [0., 0., 1.]})
    monkeypatch.setattr(contacts, 'subscribe', lambda readings: subscription)
    monkeypatch.setattr(visuals, 'Ink', Ink)
    return spec, BuildContext(sim, robot, LocalAssets(tmp_path, ASSETS), tmp_path/'out'), subscription


def test_scene_keeps_source_geometry_and_only_resets_stamp(prepared):
    spec, ctx, subscription = prepared
    before = ctx.sim.stage.GetRootLayer().ExportToString()
    sc = scene.build(spec, ctx)
    assert ctx.sim.stage.GetRootLayer().ExportToString() == before
    assert ctx.sim.steps == 120 and ctx.robot.prepared == ctx.robot.initialized == 1
    assert sc.body.pose == ([.02, -.2, .772], [1., 0., 0., 0.])
    assert sc.observe(0).instruction == spec.config['instruction']
    sc.before_step(1)
    sc.after_step(1)
    assert sc.evidence(1).tick == 1 and sc.health().ok and not sc.is_complete()
    sc.close()
    assert subscription.closed


def test_changed_source_asset_and_wrong_scene_config_fail_before_simulation(prepared):
    spec, ctx, _ = prepared
    ctx.assets.resolve('scene_config').write_text('{}')
    with pytest.raises(ValueError, match='digest mismatch'):
        scene.build(spec, ctx)
    assert ctx.sim.steps == 0


def test_native_tcp_reads_actual_flange_transform_with_the_source_offset():
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.CreateInMemory()
    for side in ('Left', 'Right'):
        flange = UsdGeom.Xform.Define(stage, '/World/Robot'+side+'/Asset/link6')
        flange.AddTranslateOp().Set(Gf.Vec3d(.2, .3, .8))
        flange.AddRotateZOp().Set(90.)
    result = scene._tcp_positions(stage)
    for value in result.values():
        assert value == pytest.approx([.2, .445, .8])


def test_producer_pins_source_bytes_and_never_overwrites_an_existing_spec(tmp_path, monkeypatch):
    from mind2act_eval.tasks.PR5.instances import main
    from mind2act_eval.contracts import InstanceSpec
    source = tmp_path/'source.json'
    source.write_text(json.dumps(source_row()))
    entries = tmp_path/'asset-entries.json'
    entries.write_text(json.dumps(dict(assets=ASSETS)))
    output = tmp_path/'portable-spec.json'
    args = ['producer', '--source-spec', str(source), '--source-spec-sha256',
        hashlib.sha256(source.read_bytes()).hexdigest(), '--asset-entries', str(entries),
        '--asset-version', 'test', '--output', str(output)]
    monkeypatch.setattr('sys.argv', args)
    main()
    spec = InstanceSpec.from_dict(json.loads(output.read_text()))
    assert spec.instance_id == source_row()['case_id']
    assert validate(spec)['index'] == 0
    with pytest.raises(FileExistsError):
        main()
    source.write_text('{}')
    with pytest.raises(ValueError, match='source spec SHA256'):
        main()


def test_goal_recipe_matches_terminal_return_ruling_and_preserves_e2e_instruction():
    from mind2act_eval.tasks.PR5.instances import goal_recipe, instruction
    from mind2act_eval.tasks.PR5.scorer import RULE_VERSION
    data = json.loads(resources.files('mind2act_eval.adapters.agent_input').joinpath('data/goals.json').read_text())
    for level in ('easy', 'medium', 'hard'):
        recipe = goal_recipe(level)
        assert data['PR5']['goals'][level] == recipe
        assert 'Holding or releasing the stamp is allowed.' in recipe
        assert all(text not in recipe for text in ('withdraw', '80 mm above', '0.5 seconds', 'and release it'))
        assert instruction(level).endswith('in its original orientation and release it.')
    spec = spec_from_source(source_row(), assets=ASSETS, asset_version='test')
    assert spec.config['scoring_version'] == RULE_VERSION
