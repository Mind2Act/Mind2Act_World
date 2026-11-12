"""Exercise the production contact-report authoring path without starting Kit."""
from types import SimpleNamespace as NS
import sys

import pytest
from mind2act_eval.contracts import InstanceSpec, ClockSpec
from mind2act_eval.tasks.CP04.scene import build
from mind2act_eval.tasks.CP04.mechanism.thermal_profile497 import VERSION


def test_contact_authoring_leaves_robot_and_unrelated_prims_untouched(monkeypatch):
    collision, rigid = object(), object()
    writes = []

    class Prim:
        def __init__(self, path, schemas):
            self.path, self.schemas = path, schemas
        def GetPath(self):
            return self.path
        def HasAPI(self, schema):
            return schema in self.schemas

    class Report:
        @staticmethod
        def Apply(prim):
            # Record API application too, not merely the threshold's final value.
            writes.append(('apply', prim.path))
            return NS(CreateThresholdAttr=lambda value: writes.append(('threshold', prim.path, value)))

    paths = [('/World/RobotLeft/Asset/link6', [rigid]),
             ('/World/RobotLeft/Asset/link7/Collision', [collision]),
             ('/World/RobotRight/Asset/link8', [rigid, collision]),
             ('/World/Cooking/Jar0', [rigid]),
             ('/World/Cooking/TrialLid/Collision', [collision]),
             ('/World/Cooking/Recipe', []),
             ('/World/CookingReplica/Body', [rigid]),
             ('/World/Floor', [collision])]
    monkeypatch.setitem(sys.modules, 'pxr', NS(
        UsdPhysics=NS(CollisionAPI=collision, RigidBodyAPI=rigid),
        PhysxSchema=NS(PhysxContactReportAPI=Report)))
    stage = NS(Traverse=lambda: iter(Prim(path, schemas) for path, schemas in paths))
    phases = []
    monkeypatch.setitem(sys.modules, 'mind2act_eval.tasks.CP04.mechanism.contact_contract',
                        NS(apply=lambda stage: phases.append('task_contact_contract')))

    class StopBeforeNativeWorld(Exception):
        pass

    class Context:
        sim = NS(stage=stage)
        robot = NS(configure_arm_reference_slew=lambda rate: dict(
                       physics_hz=960, arm_reference_slew_rad_s=rate,
                       per_physics_tick_delta_limit_rad=rate / 960, native_parameter=False),
                   prepare_stage=lambda stage: phases.append('robot_prepare'),
                   attach_cameras=lambda stage: phases.append('robot_cameras'))

        @property
        def world(self):
            raise StopBeforeNativeWorld

    spec = InstanceSpec('CP04', 'synthetic', 'easy', 'synthetic', 'v1', 'v1', 'v1',
        clock=ClockSpec(960, 30, 30), config={
            'cp04_clock_profile': 'cp04-source-physics960-v1',
            'ingredient_identities': {'0': 'a', '1': 'b'},
            'recipe_indices': [0, 1], 'recipe': ['a', 'b'],
            'difficulty': 'easy', 'thermal_profile_version': VERSION,
            'thermal_profile': {'difficulty': 'easy', 'version': VERSION}})
    # Execute build -> binding -> _initialize -> contact authoring, stopping
    # only at native World access. USD/PhysX dependencies are explicit doubles.
    with pytest.raises(StopBeforeNativeWorld):
        build(spec, Context())
    assert phases == ['robot_prepare', 'robot_cameras', 'task_contact_contract']
    assert writes == [
        ('apply', '/World/Cooking/Jar0'), ('threshold', '/World/Cooking/Jar0', 0.),
        ('apply', '/World/Cooking/TrialLid/Collision'),
        ('threshold', '/World/Cooking/TrialLid/Collision', 0.)]
