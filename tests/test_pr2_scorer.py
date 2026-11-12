import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks import PR2, TaskRegistry
from mind2act_eval.tasks.PR2.evidence import VoltageModel, yaw_deg
from mind2act_eval.tasks.PR2.scorer import score


def evidence(values, hz=120, target=140., consecutive=True, task='PR2', touching=None):
    touching = tuple(touching if touching is not None else [False] * len(values))
    return NativeEvidence(task, 'i', len(values), dict(difficulty='easy', tick_hz=hz, target_value=target, first_tick=1,
                                                       values=tuple(values), touching=touching, consecutive=consecutive))


def test_registers_without_pxr():
    assert TaskRegistry([PR2.DEFINITION]).task_ids == ('PR2',)


def test_one_volt_tolerance_released_one_second_hold():
    assert score(evidence([140.9] * 121)).value == 100.     # 120 intervals / 120 Hz = 1 s
    assert score(evidence([140.9] * 120)).value == 100.
    assert score(evidence([141.0] * 121)).value == 100.     # inclusive
    assert score(evidence([141.1] * 121)).value == pytest.approx(29.175)
    assert score(evidence([140.] * 200 + [143.])).value == pytest.approx(27.75)   # must hold up to the last sample
    assert score(evidence([140.] * 121, touching=[True] * 20 + [False] * 101)).value == 100.   # 0.83 s released
    assert score(evidence([140.] * 141, touching=[True] * 20 + [False] * 121)).value == 100.


def test_pr1_tolerance_does_not_leak_into_pr2():
    assert score(evidence([140.5] * 121)).value == 100.   # 0.5 V would fail PR1's 0.2 ohm


def test_invalid_evidence_and_other_task_raise():
    for bad in (evidence([140.] * 61, task='PR1'), evidence([])):
        with pytest.raises(ValueError):
            score(bad)


def test_voltage_mapping_matches_annotation():
    m = VoltageModel(150., 0., -300. / 260.)
    assert m.volts(150.) == 0. and m.volts(-150.) == pytest.approx(260.)
    assert m.volts(m.yaw_deg(100.)) == pytest.approx(100.)
    assert m.yaw_deg(100.) == pytest.approx(34.615384, abs=1e-5)
    assert yaw_deg([[0., 0.], [1., 0.]]) == 90.


def test_final_voltage_can_be_held_and_needs_only_t0_sample():
    assert score(evidence([140.], touching=[True])).value == 100.
    assert score(evidence([140., 142.])).value == pytest.approx(28.5)
