import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from mind2act_eval.contracts import BuildContext, InstanceSpec, to_json
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.DR4 import scene
from mind2act_eval.tasks.DR4.instances import digest, spec_from_source, validate, main
from mind2act_eval.tasks.catalog import load_task

ORIGINAL_FLANGES = scene._flanges


def row():
    condition = dict(generator_version='pr7-task-seed-v1', task_seed=711000, start_offset_xy_m=[.01,-.01],
        goal_offset_xy_m=[-.01,.01], tray_initial_yaw_deg=2., obstacle_offsets_xy_m=[[.002,.001]])
    case = dict(level='medium', radius_mm=25., ball_count=3, source_seed=711000, task_condition_sha256=digest(condition))
    return dict(task_id='PR7', case_id='PR7-medium-train-s711000', difficulty='medium',
        binding=dict(declared_version='pr7-task-seeded-v1', task_definition_version='printed-guide-30-25-20-v1', config_sha256=digest(case)),
        config=dict(source_case=case, task_condition=condition, scene_sha256='0'*64,
                    native_inputs=dict(manifest=dict(station=dict(world_support_z=.768)))))


ASSETS = dict(station='prepared/scene.usda', x5='robot/ARX.usd')


def bind(source, *, assets, asset_version):
    raw = json.dumps(source).encode()
    return spec_from_source(raw, source_spec_sha256=hashlib.sha256(raw).hexdigest(), assets=assets, asset_version=asset_version)


def test_public_alias_preserves_original_case_id_seed_and_conditions():
    original = row()
    spec = bind(original, assets=ASSETS, asset_version='synthetic')
    validate(spec)
    assert spec.task_id == 'DR4' and spec.instance_id == 'PR7-medium-train-s711000' and spec.seed == 711000
    assert spec.config['source_case']['ball_count'] == 3
    assert to_json(spec.config['task_condition']) == original['config']['task_condition']
    assert spec.config['instruction'].startswith('Carry the tray of balls')
    assert 'Goal: '+spec.config['instruction'] in spec.config['goal_recipe']
    assert 'native_inputs' not in spec.config
    assert load_task('DR4').scorer.__module__ == 'mind2act_eval.tasks.DR4.scorer'
    from mind2act_eval.robot import X5Robot
    assert X5Robot(rest=spec.config.get('robot_rest')).rest_pose() == pytest.approx(
        np.r_[np.radians([0.,65.,100.,-65.,0.,0.]),.044,.044])


def test_case_condition_changes_and_absolute_asset_paths_are_rejected():
    original = row()
    raw = json.dumps(original).encode()
    changed = json.loads(raw)
    changed['config']['task_condition']['start_offset_xy_m'][0] += .001
    with pytest.raises(ValueError, match='SHA256'):
        spec_from_source(json.dumps(changed).encode(), source_spec_sha256=hashlib.sha256(raw).hexdigest(), assets=ASSETS, asset_version='test')
    spec = bind(original, assets=ASSETS, asset_version='test')
    config = to_json(spec.config)
    config['task_condition']['start_offset_xy_m'][0] += .001
    with pytest.raises(ValueError, match='condition'):
        validate(replace(spec, config=config))
    with pytest.raises(ValueError, match='asset-library'):
        bind(row(), assets=dict(ASSETS, station='/former/host/scene.usda'), asset_version='test')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage = Usd.Stage.CreateInMemory()
    def transform(path, position):
        prim = UsdGeom.Xform.Define(stage, path)
        prim.AddTranslateOp().Set(Gf.Vec3d(*position))
        return prim
    for name, pos in [('StartPlatform',[0.,-.1,.75]), ('GoalPlatform',[0.,.15,.75]),
                      ('Column',[0.,.04,.9]), ('ColumnMount',[0.,.04,.75]), ('ColumnVisual',[0.,.04,.9]),
                      ('StartLabel',[0.,-.1,.78]), ('GoalLabel',[0.,.15,.78])]:
        transform('/World/PR7/'+name, pos)
    tray = transform('/World/PR7/Tray', [0.,-.1,.784])
    tray.AddOrientOp().Set(Gf.Quatf(1.,Gf.Vec3f(0.,0.,0.)))
    UsdPhysics.MassAPI.Apply(tray.GetPrim()).CreateMassAttr(.25)
    shell = UsdGeom.Mesh.Define(stage, '/World/PR7/Tray/Shell')
    points = np.zeros((240,3))
    a = np.arange(48)*2*np.pi/48
    points[96:144,:2] = .15*np.column_stack((np.cos(a), np.sin(a)))
    points[144,2], points[192,2] = .03, 0.
    shell.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    for i in range(3):
        transform('/World/PR7/Ball'+str(i), [-.03+.03*i,-.1,.804])
    root = tmp_path/'assets'
    (root/'prepared').mkdir(parents=True)
    (root/'robot').mkdir()
    stage.GetRootLayer().Export(str(root/ASSETS['station']))
    (root/ASSETS['x5']).write_text('synthetic placeholder')
    original = row()
    original['config']['scene_sha256'] = hashlib.sha256((root/ASSETS['station']).read_bytes()).hexdigest()
    spec = bind(original, assets=ASSETS, asset_version='synthetic')
    class Body:
        def __init__(self, path=None, position=None): self.path, self.position = path, position
        def initialize(self): pass
        def get_world_pose(self):
            if self.path is None: return np.asarray(self.position), np.asarray([1.,0.,0.,0.])
            prim = stage.GetPrimAtPath(self.path)
            p = list(prim.GetAttribute('xformOp:translate').Get())
            q = prim.GetAttribute('xformOp:orient')
            value = q.Get() if q else Gf.Quatf(1.,Gf.Vec3f(0.,0.,0.))
            return np.asarray(p), np.asarray([value.GetReal(),*value.GetImaginary()])
        def get_linear_velocity(self): return np.zeros(3)
        def get_angular_velocity(self): return np.zeros(3)
        def get_world_poses(self):
            p,q = self.get_world_pose()
            return np.asarray([p]),np.asarray([q])
    class Robot:
        prepared = initialized = 0
        def prepare_stage(self, stage): self.prepared += 1
        def initialize(self, world): self.initialized += 1
        def observe(self):
            return dict(left_arm=[0.]*6,right_arm=[0.]*6,left_gripper=1.,right_gripper=1.)
    class World:
        current_time = 0.
    class Sim:
        world = World()
        steps = 0
        def step(self, *, render):
            assert render is False
            self.steps += 1
            self.world.current_time += 1/120
    class Subscription:
        closed = False
        def unsubscribe(self): self.closed = True
    sim = Sim()
    sim.stage = stage
    robot, subscription = Robot(), Subscription()
    monkeypatch.setattr(scene, '_rigid_bodies', lambda world,n: dict(tray=Body('/World/PR7/Tray'),
        **{'Ball'+str(i):Body('/World/PR7/Ball'+str(i)) for i in range(n)}))
    monkeypatch.setattr(scene, '_flanges', lambda *args: {s:Body(position=[.3,.3,1.]) for s in ('Left','Right')})
    from mind2act_eval.tasks.DR4.mechanism import contacts
    monkeypatch.setattr(contacts, 'subscribe', lambda readings: subscription)
    return spec, BuildContext(sim,robot,LocalAssets(root,ASSETS),tmp_path/'output'), subscription


def test_scene_applies_only_frozen_source_offsets_and_preserves_physics(prepared):
    from pxr import UsdPhysics
    spec,ctx,subscription = prepared
    scene_object = scene.build(spec,ctx)
    assert ctx.sim.steps == 240 and ctx.robot.prepared == ctx.robot.initialized == 1
    assert UsdPhysics.MassAPI(ctx.sim.stage.GetPrimAtPath('/World/PR7/Tray')).GetMassAttr().Get() == .25
    actual = list(ctx.sim.stage.GetPrimAtPath('/World/PR7/StartPlatform').GetAttribute('xformOp:translate').Get())
    assert actual[:2] == pytest.approx([.01,-.11])
    assert scene_object.evidence(0).task_id == 'DR4'
    ctx.sim.step(render=False)
    scene_object.before_step(1)
    scene_object.after_step(1)
    assert scene_object.evidence(1).tick == 1 and not scene_object.is_complete()
    scene_object.close()
    assert subscription.closed


def test_wrong_station_digest_fails_before_scene_changes_or_physics(prepared):
    spec,ctx,_ = prepared
    ctx.assets.resolve('station').write_text('changed')
    with pytest.raises(ValueError,match='scene changed'):
        scene.build(spec,ctx)
    assert ctx.sim.steps == 0


def test_producer_requires_original_sha_and_never_overwrites(tmp_path, monkeypatch):
    source, entries, output = tmp_path/'source.json', tmp_path/'entries.json', tmp_path/'portable.json'
    source.write_text(json.dumps(row()))
    entries.write_text(json.dumps(dict(assets=ASSETS)))
    monkeypatch.setattr('sys.argv',['producer','--source-spec',str(source),'--source-spec-sha256',hashlib.sha256(source.read_bytes()).hexdigest(),
        '--asset-entries',str(entries),'--asset-version','test','--output',str(output)])
    main()
    validate(InstanceSpec.from_dict(json.loads(output.read_text())))
    with pytest.raises(FileExistsError): main()
    source.write_text('{}')
    with pytest.raises(ValueError,match='SHA256'): main()


def sdk_pose_views(stage, monkeypatch):
    """Model official IsaacSim v5.1.0 defaults on real CPU USD prims.

    SingleRigidPrim.__init__:121–134 does not expose/pass sensor preparation;
    RigidPrim.__init__:145–147 defaults preparation=True, and
    _apply_rigid_body_apis:2279–2292 authors sleepThreshold=0 on that path.
    The regression runs the actual task build, not just a kwargs assertion.
    """
    import sys
    from types import ModuleType
    from pxr import UsdGeom, UsdPhysics, Sdf
    paths = {s:'/World/Robot'+s+'/Asset/link6' for s in ('Left','Right')}
    for side, path in paths.items():
        UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, path.rsplit('/',1)[0]).GetPrim())
        prim = UsdGeom.Xform.Define(stage,path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.13)
        prim.CreateAttribute('physxRigidBody:sleepThreshold',Sdf.ValueTypeNames.Float).Set(.037)
    non_root, calls, writes = set(paths.values()), [], []
    class RigidView:
        def __init__(self, prim_paths_expr, name='rigid_prim_view', *, reset_xform_properties=True,
                     track_contact_forces=False, prepare_contact_sensors=True,
                     disable_stablization=True, contact_filter_prim_paths_expr=()):
            self.path, self.initialized = prim_paths_expr, False
            prim = stage.GetPrimAtPath(self.path)
            calls.append(dict(path=self.path, reset=reset_xform_properties, prepare=prepare_contact_sensors,
                              track=track_contact_forces, disable_stablization=disable_stablization,
                              filters=list(contact_filter_prim_paths_expr)))
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
                writes.append((self.path,'RigidBodyAPI'))
            if prepare_contact_sensors or track_contact_forces or contact_filter_prim_paths_expr:
                prim.CreateAttribute('physxRigidBody:sleepThreshold',Sdf.ValueTypeNames.Float).Set(0.)
                writes.append((self.path,'sleepThreshold'))
        def initialize(self): self.initialized = True
        def get_world_poses(self):
            assert self.initialized
            # Native poses deliberately differ from authored USD transforms.
            return np.array([[.3,.3,1.]]), np.array([[np.sqrt(.5),0.,0.,np.sqrt(.5)]])
    class SingleView:
        def __init__(self, prim_path, name='rigid_prim', reset_xform_properties=True):
            self.view = RigidView(prim_path,name,reset_xform_properties=reset_xform_properties)
        def initialize(self): self.view.initialize()
        def get_world_pose(self):
            p,q = self.view.get_world_poses()
            return p[0],q[0]
    for name in ('isaacsim','isaacsim.core','isaacsim.core.prims','isaacsim.core.utils','isaacsim.core.utils.prims'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['isaacsim.core.prims'].RigidPrim = RigidView
    sys.modules['isaacsim.core.prims'].SingleRigidPrim = SingleView
    sys.modules['isaacsim.core.utils.prims'].is_prim_non_root_articulation_link = lambda *,prim_path: prim_path in non_root
    return paths,non_root,calls,writes,SingleView


def robot_authored_state(stage):
    from pxr import Sdf
    layer = Sdf.Layer.CreateAnonymous()
    Sdf.CreatePrimInLayer(layer,'/World')
    for side in ('Left','Right'):
        path = '/World/Robot'+side
        Sdf.CopySpec(stage.GetRootLayer(),path,layer,path)
    return layer.ExportToString()


def test_sdk_default_single_view_reproduces_sleep_write_even_with_reset_disabled(monkeypatch):
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    paths,_,_,writes,SingleView = sdk_pose_views(stage,monkeypatch)
    before = robot_authored_state(stage)
    SingleView(paths['Left'],reset_xform_properties=False).initialize()
    assert writes == [(paths['Left'],'sleepThreshold')]
    assert robot_authored_state(stage) != before


def test_actual_build_flange_views_do_not_author_robot_physics_or_transforms(prepared,monkeypatch):
    spec,ctx,_ = prepared
    paths,_,calls,writes,_ = sdk_pose_views(ctx.sim.stage,monkeypatch)
    before = robot_authored_state(ctx.sim.stage)
    monkeypatch.setattr(scene,'_flanges',ORIGINAL_FLANGES)
    built = scene.build(spec,ctx)
    try:
        assert robot_authored_state(ctx.sim.stage) == before
        assert writes == [] and {c['path'] for c in calls} == set(paths.values())
        assert all(c == dict(path=c['path'],reset=False,prepare=False,track=False,
                             disable_stablization=False,filters=[]) for c in calls)
        assert built.measure(0)['tcps']['Left'] == pytest.approx([.3,.445,1.])
    finally:
        built.close()


@pytest.mark.parametrize('side',['Left','Right'])
@pytest.mark.parametrize('failure',['missing_api','root_link','missing_prim'])
def test_flange_preconditions_refuse_sdk_authoring_before_any_constructor(monkeypatch,side,failure):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    paths,non_root,calls,writes,_ = sdk_pose_views(stage,monkeypatch)
    if failure == 'missing_api': stage.GetPrimAtPath(paths[side]).RemoveAPI(UsdPhysics.RigidBodyAPI)
    elif failure == 'root_link': non_root.remove(paths[side])
    else: stage.RemovePrim(paths[side])
    before = robot_authored_state(stage)
    with pytest.raises(ValueError,match='existing non-root'):
        ORIGINAL_FLANGES(stage)
    assert calls == writes == [] and robot_authored_state(stage) == before
