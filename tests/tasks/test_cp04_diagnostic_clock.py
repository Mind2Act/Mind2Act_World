"""Task-only clock seam; CPU construction never claims native acceptance."""
from dataclasses import replace
from types import SimpleNamespace as NS
import sys

import pytest

from mind2act_eval.contracts import ClockSpec, to_json
from mind2act_eval.tasks.CP04 import scene as module
from .test_cp04_build_binding import instance, native_context  # noqa: F401
from .test_cp04_shared_slew import Arm, articulation_action
from .test_cp04_evaluation_clock import evaluation_spec


DIAGNOSTIC = 'cp04-easy-physics120-diagnostic-v1'


def diagnostic_spec():
    base = instance('easy')
    return replace(base, clock=ClockSpec(120, 30, 30),
                   config={**to_json(base.config), 'cp04_clock_profile': DIAGNOSTIC})


def test_explicit_easy_diagnostic_initializes_at_120_with_equal_preroll(native_context):
    ctx, _ = native_context
    steps = []
    ctx.world.step = lambda **kw: steps.append(kw)
    ctx.robot.configure_arm_reference_slew = lambda rate: dict(
        physics_hz=120, arm_reference_slew_rad_s=rate,
        per_physics_tick_delta_limit_rad=rate / 120, native_parameter=False)
    scene = module.build(diagnostic_spec(), ctx)
    assert scene.native['jars'].physics_dt == 1 / 120
    assert len(steps) == 12
    receipt = scene.result_metadata()
    assert receipt['physics_hz'] == receipt['logical_hz'] == 120
    assert receipt['initialization_physics_ticks'] == 12
    assert receipt['clock_profile_version'] == DIAGNOSTIC
    assert receipt['diagnostic_only'] is True
    scene.close()


@pytest.mark.parametrize('change', ['medium', 'unknown', 'wrong_rate'])
def test_diagnostic_is_explicit_easy_only_and_rejects_before_native(change):
    spec = diagnostic_spec()
    if change == 'medium':
        spec = replace(spec, difficulty='medium')
    elif change == 'unknown':
        spec = replace(spec, config={**to_json(spec.config), 'cp04_clock_profile': 'unknown'})
    else:
        spec = replace(spec, clock=ClockSpec(960, 30, 30))
    with pytest.raises(ValueError, match='CP04'):
        module.build(spec, None)


@pytest.mark.parametrize('spec', [instance('easy'), diagnostic_spec()])
def test_shared_world_receives_real_dt_without_starting_kit(monkeypatch, spec):
    from mind2act_eval.sim import IsaacSimulator
    calls = []
    stage, output = object(), object()
    monkeypatch.setattr(IsaacSimulator, 'stage', property(lambda self: stage))
    monkeypatch.setattr('mind2act_eval.sim.asset_resolution.WorldAssetGate', lambda world, actual_stage, actual_output: NS(world=world, stage=actual_stage, output=actual_output))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.api', NS(World=lambda **kw: calls.append(kw) or kw))
    sim = object.__new__(IsaacSimulator)
    sim._clock, sim._world, sim._output_dir = spec.clock, None, output
    world = sim.world
    assert sim._asset_resolution_gate.stage is stage
    assert sim._asset_resolution_gate.output is output
    assert world['physics_dt'] == 1 / spec.clock.physics_hz
    assert world['rendering_dt'] == 1 / 30
    assert sim.world is world and len(calls) == 1


def test_120_and_960_keep_same_logical_thermal_sample_and_recipe_time(monkeypatch):
    presentations = []
    monkeypatch.setattr('mind2act_eval.tasks.CP04.mechanism.recipe_screen.show_ready',
                        lambda screen, template: presentations.append((screen, template)))
    histories = []
    for spec in [instance('easy'), diagnostic_spec()]:
        thermal, samples = [], []
        n = dict(counter={'step': 0}, visible='visible', hidden='hidden')
        n['observer'] = NS(thermal_sample=lambda tick, row: thermal.append((tick, row['dt'])))
        n['appliance'] = NS(post_step=lambda dt, update_display: dict(dt=dt))
        n['sample'] = lambda: samples.append(n['counter']['step'])
        scene = module.CookingScene(spec, None, n)
        for tick in range(1, spec.clock.physics_hz * 4 + 1):
            scene.after_step(tick)
        histories.append((thermal, samples, n['observer'].hidden))
    assert histories[0] == histories[1]
    assert histories[1][0] == [(i, 1 / 120) for i in range(1, 481)]
    assert histories[1][1] == list(range(4, 481, 4))
    assert histories[1][2] == 420
    assert presentations == [('visible', 'hidden')] * 2


@pytest.mark.parametrize('spec', [diagnostic_spec(), *[evaluation_spec(tier) for tier in ('easy', 'medium', 'hard')]])
def test_shared_slew_and_adjacent_lerp_use_four_real_physics_updates(monkeypatch, spec):
    import numpy as np
    from mind2act_eval.contracts import Action
    from mind2act_eval.robot import X5Robot
    from mind2act_eval.runner import interpolate_action
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types', NS(ArticulationAction=articulation_action))
    robot = X5Robot(rest=[0.] * 6 + [.01, .01], physics_hz=120)
    robot._arms = {'left': Arm(), 'right': Arm()}
    monkeypatch.setattr(module, '_initialize', lambda *a: {})
    module.build(spec, NS(robot=robot))
    robot.reset_to_rest()
    for arm in robot._arms.values():
        arm.commands.clear()
        arm.measured[:6] = .7
    current = Action((.8,) * 6, (-.8,) * 6, 1., 1.)
    following = Action((1.6,) * 6, (-1.6,) * 6, 1., 1.)
    assert spec.clock.ticks_per_action == 4
    requested = []
    for i in range(4):
        action = interpolate_action(current, following, i / 4)
        requested.append(action.left_arm[0])
        robot.apply(action)
    assert requested == pytest.approx([.8, 1., 1.2, 1.4])
    commands = np.asarray(robot._arms['left'].commands)
    np.testing.assert_allclose(commands[:, :6], np.repeat((np.arange(1, 5) * 5 / 120)[:, None], 6, axis=1))
    np.testing.assert_allclose(commands[:, 6:], .0208)  # unchanged0.0108m lead from measured.01
    assert robot.software_control_receipt()['per_physics_tick_delta_limit_rad'] == 5 / 120


def test_task_explicit_source_clock_and_receipt():
    default = module.CookingScene(instance('hard'), None, {})
    assert default.clock.physics_hz == 960 and default.clock.substeps == 8
    receipt = default.result_metadata()
    assert receipt['initialization_physics_ticks'] == 96
    assert receipt['diagnostic_only'] is False
