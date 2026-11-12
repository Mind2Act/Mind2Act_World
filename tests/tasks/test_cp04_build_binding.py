"""Binding-to-build regression: real thermal model, explicit native test doubles."""
from dataclasses import replace
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json
from mind2act_eval.tasks.CP04.scene import build
from mind2act_eval.tasks.CP04.mechanism.thermal_profile497 import VERSION, config, window_seconds


def instance(tier):
    count = 3 if tier == 'hard' else 2
    return InstanceSpec('CP04', f'CP04_{tier}_synthetic', tier, 'synthetic', 'v1', 'v1', 'v1',
        clock=ClockSpec(960, 30, 30), config={
            'difficulty': tier, 'cp04_clock_profile': 'cp04-source-physics960-v1',
            'ingredient_identities': {str(i): name for i, name in enumerate(('chili', 'garlic', 'ginger', 'onion'))},
            'recipe_indices': list(range(count)), 'recipe': list(('chili', 'garlic', 'ginger')[:count]),
            'thermal_profile': config(tier), 'thermal_profile_version': VERSION})


class NoNativeAccess:
    @property
    def sim(self):
        pytest.fail('Conflicting thermal binding reached native scene access')


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize('conflict', ['config_difficulty', 'profile_difficulty', 'config_version',
                                     'profile_version', 'missing_config_version', 'legacy_version',
                                     'unknown_version'])
def test_build_rejects_thermal_conflict_before_native_access(native_context, tier, conflict):
    spec = instance(tier)
    values = to_json(spec.config)
    other = 'easy' if tier != 'easy' else 'hard'
    if conflict == 'config_difficulty':
        values['difficulty'] = other
    elif conflict == 'profile_difficulty':
        values['thermal_profile']['difficulty'] = other
    elif conflict == 'config_version':
        values['thermal_profile_version'] = 'different-version'
    elif conflict == 'profile_version':
        values['thermal_profile']['version'] = 'different-version'
    elif conflict == 'missing_config_version':
        del values['thermal_profile_version']
    elif conflict == 'legacy_version':
        values['thermal_profile_version'] = values['thermal_profile']['version'] = None
    else:
        values['thermal_profile_version'] = values['thermal_profile']['version'] = 'unknown-version'
    with pytest.raises(ValueError, match='CP04.*(difficulty|thermal|version)'):
        build(replace(spec, config=values), NoNativeAccess())


@pytest.fixture
def native_context(monkeypatch):
    """Run all of _initialize; fake native bodies/renderer, keep ThermalState real."""
    rigid_api, collision_api = object(), object()

    class Prim:
        def __init__(self, path): self.path = path
        def GetPath(self): return self.path
        def GetName(self): return self.path.rsplit('/', 1)[-1]
        def IsValid(self): return True
        def HasAPI(self, api): return api is rigid_api and ('/LooseFood' in self.path or self.path.startswith('/World/Robot'))
        def GetAttribute(self, name): return NS(Get=lambda: [0., 0., 0.])

    food_paths = [f'/World/Cooking/LooseFood{i}/Piece{j}' for i in range(4) for j in range(3)]
    stage = NS(Traverse=lambda: iter([Prim(p) for p in food_paths] + [Prim('/World/Cooking/RecipeHidden')]),
               GetPrimAtPath=lambda p: Prim(str(p)))

    class Body:
        def __init__(self, prim_path=None, prim_paths_expr=None, name=None, **kwargs):
            self.prim_paths = [prim_paths_expr] if isinstance(prim_paths_expr,str) else prim_paths_expr or []
        def get_world_pose(self): return np.array([.4, .2, .8]), np.array([1., 0., 0., 0.])
        def get_world_poses(self):
            if self.prim_paths and self.prim_paths[0].startswith('/World/Robot'):
                p,q = self.get_world_pose()
                return np.asarray([p]),np.asarray([q])
            return np.zeros((len(self.prim_paths), 3)), None
        def get_linear_velocity(self): return np.zeros(3)
        def get_angular_velocity(self): return np.zeros(3)

    class JarSet:
        def add_primary(self, index, jar, lid, guide, body_path, lid_path):
            self.items[index] = dict(jar=jar, lid=lid, released=False, body_path=body_path, lid_path=lid_path)
        def build_batch(self, world): pass
        @staticmethod
        def pose(body): return body.get_world_pose()[0], np.eye(3)

    # Native geometry is a separate CPU-tested read boundary; this fixture
    # isolates the thermal binding and does not simulate support contacts.
    from mind2act_eval.tasks.CP04.mechanism import containment
    monkeypatch.setattr(containment, 'DrawerSlots', lambda *args: NS(contains=lambda *args: True))
    monkeypatch.setattr(containment, 'Containers', lambda stage: NS(
        cache=NS(ComputeUntransformedBound=lambda prim:NS(ComputeAlignedRange=lambda:NS(GetMidpoint=lambda:[0.,0.,0.]))),
        pot_contains=lambda point, **kwargs: False,
        tray_lower=[0.,0.],tray_upper=[1.,1.]))

    display_calls = []
    def attach(stage, difficulty, version):
        display_calls.append((difficulty, version))
        return NS(limit=80. + 2.5 * window_seconds(difficulty, version))

    monkeypatch.setitem(sys.modules, 'pxr', NS(
        UsdGeom=NS(Imageable=lambda prim: NS(MakeVisible=lambda: None, MakeInvisible=lambda: None)),
        UsdPhysics=NS(RigidBodyAPI=rigid_api, CollisionAPI=collision_api, Joint=lambda prim: prim),
        PhysxSchema=NS(PhysxContactReportAPI=NS(Apply=lambda prim: NS(CreateThresholdAttr=lambda value: None))),
        PhysicsSchemaTools=NS(intToSdfPath=lambda path: path)))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.prims', NS(SingleRigidPrim=Body, RigidPrim=Body))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.prims', NS(is_prim_non_root_articulation_link=lambda **kwargs: True))
    monkeypatch.setitem(sys.modules, 'mind2act_eval.tasks.CP04.mechanism.contact_contract', NS(apply=lambda stage: None))
    monkeypatch.setitem(sys.modules, 'mind2act_eval.tasks.CP04.mechanism.multi_jars', NS(JarSet=JarSet))
    monkeypatch.setitem(sys.modules, 'mind2act_eval.tasks.CP04.mechanism.appliance', NS(Appliance=type('Appliance', (), {})))
    monkeypatch.setitem(sys.modules, 'mind2act_eval.tasks.CP04.mechanism.display', NS(attach=attach))
    monkeypatch.setitem(sys.modules, 'omni.physx', NS(get_physx_simulation_interface=lambda:
        NS(subscribe_contact_report_events=lambda callback: object())))
    monkeypatch.setitem(sys.modules, 'omni.physx.bindings._physx', NS(ContactEventType=NS(CONTACT_LOST='lost')))
    monkeypatch.setitem(sys.modules, 'scipy.spatial.transform', NS(Rotation=NS(
        from_quat=lambda q: NS(apply=lambda xyz: np.asarray(xyz)))))
    world = NS(scene=NS(add=lambda body: body), reset=lambda: None, step=lambda **kwargs: None)
    robot = NS(configure_arm_reference_slew=lambda rate: dict(
                   physics_hz=960, arm_reference_slew_rad_s=rate,
                   per_physics_tick_delta_limit_rad=rate / 960, native_parameter=False),
               prepare_stage=lambda stage: None, attach_cameras=lambda stage: None,
               articulations=lambda world: {'Left': object(), 'Right': object()},
               initialize=lambda world: None, reset_to_rest=lambda: None, verify=lambda robots,stage: None)
    return NS(sim=NS(stage=stage), world=world, robot=robot), display_calls


@pytest.mark.parametrize('tier,seconds', [('easy', 4.), ('medium', 3.), ('hard', 2.)])
def test_build_freezes_thermal_model_display_and_observer_to_same_binding(native_context, tier, seconds):
    ctx, display_calls = native_context
    spec = instance(tier)
    scene = build(spec, ctx)
    model = scene.native['appliance'].thermal
    assert model.difficulty == tier
    assert model.thermal_profile_version == VERSION
    assert model.snapshot()['window_seconds'] == seconds
    assert model.limit == 80. + 2.5 * seconds
    assert display_calls == [(tier, VERSION)]
    assert scene.native['appliance'].display.limit == model.limit
    evidence = scene.evidence(0)
    assert evidence.values['difficulty'] == tier
    assert evidence.values['cp04_evidence']['thermal_profile_version'] == VERSION
    scene.close()


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_build_selects_shared_reference_slew_before_initialization(monkeypatch, tier):
    import mind2act_eval.tasks.CP04.scene as module
    events = []
    def configure(rate):
        events.append(('configure', rate))
        return dict(physics_hz=960, arm_reference_slew_rad_s=rate,
                    per_physics_tick_delta_limit_rad=rate / 960, native_parameter=False)
    robot = NS(configure_arm_reference_slew=configure)
    ctx = NS(robot=robot)
    def initialize(spec, context, binding):
        assert context is ctx
        assert events == [('configure', 5.)]
        events.append(('initialize',))
        return {'cpu_test_double': True}
    monkeypatch.setattr(module, '_initialize', initialize)
    result = module.build(instance(tier), ctx)
    assert result.native == {'cpu_test_double': True}
    assert events[-1] == ('initialize',)


def test_build_fails_closed_with_robot_without_shared_slew_api(monkeypatch):
    import mind2act_eval.tasks.CP04.scene as module
    monkeypatch.setattr(module, '_initialize', lambda *args: pytest.fail('must not initialize unsupported robot'))
    with pytest.raises(AttributeError):
        module.build(instance('easy'), NS(robot=object()))
