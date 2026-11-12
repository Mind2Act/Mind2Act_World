"""CP04 uses the current shared owner and runner interpolation, without Kit."""
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from mind2act_eval.contracts import Action
from mind2act_eval.robot import X5Robot
from mind2act_eval.runner import interpolate_action
from mind2act_eval.tasks.CP04 import scene as module
from .test_cp04_build_binding import instance


class Arm:
    def __init__(self):
        self.measured = np.zeros(8)
        self.commands = []

    def set_joint_positions(self, values):
        self.measured = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values):
        pass

    def get_joint_positions(self):
        return self.measured.copy()

    def apply_action(self, action):
        # Deliberately lag measured arm positions behind commanded references.
        self.commands.append(action.joint_positions.copy())


def articulation_action(**kwargs):
    return NS(**kwargs)


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_build_uses_shared_rest_seeded_slew_for_32_physics_substeps(monkeypatch, tier):
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types',
                        NS(ArticulationAction=articulation_action))
    rest = [0.] * 6 + [.01, .01]
    robot = X5Robot(rest=rest, physics_hz=960)
    robot._arms = {'left': Arm(), 'right': Arm()}
    control_at_initialize = []

    def initialize(spec, ctx, binding):
        control_at_initialize.append(ctx.robot.software_control_receipt())
        ctx.robot.reset_to_rest()
        return {'cpu_only': True}

    monkeypatch.setattr(module, '_initialize', initialize)
    spec = instance(tier)
    module.build(spec, NS(robot=robot))
    assert spec.clock.ticks_per_action == 32
    assert control_at_initialize[0]['arm_reference_slew_rad_s'] == 5.
    assert control_at_initialize[0]['physics_hz'] == 960
    assert control_at_initialize[0]['native_parameter'] is False
    for arm in robot._arms.values():
        arm.commands.clear()
        arm.measured[:6] = .7  # The shared reference must stay rest-seeded.

    signs = np.array([1., -1., 1., -1., 1., -1.])
    current = Action(tuple(.8 * signs), tuple(-.8 * signs), 1., 1.)
    following = Action(tuple(1.6 * signs), tuple(-1.6 * signs), 1., 1.)
    for subtick in range(spec.clock.ticks_per_action):
        robot.apply(interpolate_action(current, following, subtick / spec.clock.ticks_per_action))
    for side, direction in [('left', 1.), ('right', -1.)]:
        commands = np.asarray(robot._arms[side].commands)
        expected = np.arange(1, 33)[:, None] * (5. / 960) * signs * direction
        np.testing.assert_allclose(commands[:, :6], expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(commands[0, 6:], [.0208, .0208], rtol=0, atol=1e-12)
    receipt = robot.native_receipt()
    assert receipt['status'] == 'unavailable'  # No native readback is fabricated.
    assert receipt['shared_software_control']['per_physics_tick_delta_limit_rad'] == 5. / 960
    assert receipt['shared_software_control']['gripper_target_step_fraction'] == .2


def test_build_rejects_shared_robot_clock_mismatch_before_native_access(monkeypatch):
    monkeypatch.setattr(module, '_initialize', lambda *args: pytest.fail('native access'))
    with pytest.raises(ValueError, match='shared.*960 Hz'):
        module.build(instance('hard'), NS(robot=X5Robot(physics_hz=120)))
