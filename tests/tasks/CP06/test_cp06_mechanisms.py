from types import SimpleNamespace
import numpy as np
import pytest
from pxr import Usd
from mind2act_eval.tasks.CP06.controls import BalanceControls
from mind2act_eval.tasks.CP06.rider_latch import RiderLatch


class Body:
    def __init__(self,p):
        self.p=np.asarray(p,dtype=float);self.velocity=np.zeros(3);self.forces=[]
        self._rigid_prim_view=SimpleNamespace(apply_forces_and_torques_at_pos=lambda **kw:self.forces.append(kw))
    def get_world_pose(self):return self.p,np.array([1.,0.,0.,0.])
    def get_linear_velocity(self):return self.velocity
    def get_angular_velocity(self):return np.zeros(3)


def test_original_zero_uses_measured_empty_stability_and_actual_button_travel():
    stage=Usd.Stage.CreateInMemory();button=Body([0.,0.,1.]);base=Body([0.,0.,0.])
    controls=BalanceControls(stage,button,base,Body([0.,0.,0.]))
    for _ in range(62):controls.step(0.,True)
    button.p[2]-=.004
    result=controls.step(0.,True)
    assert result['reading_valid'] and result['calibrated']
    assert controls.events[-1]['accepted'] is True


def test_zero_press_with_loaded_pan_is_rejected_by_original_physical_control():
    stage=Usd.Stage.CreateInMemory();button=Body([0.,0.,1.]);base=Body([0.,0.,0.])
    controls=BalanceControls(stage,button,base,Body([0.,0.,0.]))
    for _ in range(62):controls.step(0.,True)
    button.p[2]-=.004
    assert not controls.step(0.,False)['reading_valid']
    assert controls.events[-1]['accepted'] is False


def test_original_rider_stiction_applies_equal_opposite_source_forces_only():
    beam=Body([0.,0.,0.]);rider=Body([.082,0.,0.]);latch=RiderLatch(beam,rider)
    latch.step();rider.p[0]+=.001;rider.velocity[0]=.02
    result=latch.step()
    assert result['spring_force_n']==pytest.approx(-1.)
    assert result['latch_force_n']==pytest.approx(-1.4)
    assert np.asarray(rider.forces[-1]['forces'])+np.asarray(beam.forces[-1]['forces'])==pytest.approx(np.zeros((1,3)))
    assert np.asarray(rider.forces[-1]['positions'])==pytest.approx(beam.forces[-1]['positions'])


def test_zero_on_empty_pans_needs_no_extra_half_second_window():
    stage=Usd.Stage.CreateInMemory();button=Body([0.,0.,1.]);base=Body([0.,0.,0.])
    controls=BalanceControls(stage,button,base,Body([0.,0.,0.]))
    button.p[2]-=.004
    assert controls.step(2.,True)['calibrated']
