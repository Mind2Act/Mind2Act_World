"""CP04 formal evaluation profile through the task build/metadata seam."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from mind2act_eval.contracts import ClockSpec, to_json
from mind2act_eval.tasks.CP04 import scene as module
from .test_cp04_build_binding import instance, native_context  # noqa: F401


EVALUATION = 'cp04-evaluation-physics120-v1'


def evaluation_spec(tier, *, explicit=False):
    base = instance(tier)
    config = to_json(base.config)
    config.pop('cp04_clock_profile', None)
    if explicit:
        config['cp04_clock_profile'] = EVALUATION
    return replace(base, clock=ClockSpec(120, 30, 30), config=config)


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize('explicit', [False, True])
def test_formal_120_builds_all_tiers_and_records_version(native_context, tier, explicit):
    ctx, _ = native_context
    steps = []
    ctx.world.step = lambda **kw: steps.append(kw)
    ctx.robot.configure_arm_reference_slew = lambda rate: dict(
        physics_hz=120, arm_reference_slew_rad_s=rate,
        per_physics_tick_delta_limit_rad=rate / 120, native_parameter=False)
    scene = module.build(evaluation_spec(tier, explicit=explicit), ctx)
    receipt = scene.result_metadata()
    assert receipt['clock_profile_version'] == EVALUATION
    assert receipt['physics_hz'] == receipt['logical_hz'] == 120
    assert receipt['control_hz'] == 30 and receipt['ticks_per_control'] == 4
    assert receipt['diagnostic_only'] is False
    assert receipt['comparison_group'] == 'cp04-physics120-logical120-v1'
    assert receipt['source_physics_hz'] == 960
    assert len(steps) == receipt['initialization_physics_ticks'] == 12
    assert scene.native['jars'].physics_dt == 1 / 120
    scene.close()


def test_960_history_requires_explicit_source_and_has_separate_comparison_group():
    old = instance('hard')
    config = to_json(old.config)
    config.pop('cp04_clock_profile', None)
    with pytest.raises(ValueError, match='120/30/30'):
        module.build(replace(old, clock=ClockSpec(960, 30, 30), config=config), None)
    source = replace(old, clock=ClockSpec(960, 30, 30),
                     config={**config, 'cp04_clock_profile': 'cp04-source-physics960-v1'})
    receipt = module.CookingScene(source, None, {}).result_metadata()
    assert receipt['comparison_group'] == 'cp04-physics960-logical120-v1'
    assert receipt['clock_profile_version'] == 'cp04-source-physics960-v1'
    assert receipt['initialization_physics_ticks'] == 96


@pytest.mark.parametrize('clock', [ClockSpec(960, 30, 30), ClockSpec(120, 120, 30), ClockSpec(120, 30, 15)])
def test_formal_profile_rejects_wrong_clock_before_native(clock):
    with pytest.raises(ValueError, match='120/30/30'):
        module.build(replace(evaluation_spec('hard'), clock=clock), None)
