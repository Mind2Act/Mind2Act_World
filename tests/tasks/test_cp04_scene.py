"""CPU sampling-to-scoring integration; not native GPU acceptance."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from mind2act_eval.contracts import InstanceSpec, ClockSpec
from mind2act_eval.tasks.CP04.scene import CookingScene, build as build_cooking


def test_cooking_solver_clock():
    thermal, sampled = [], []
    observer = NS(thermal_sample=lambda tick,row: thermal.append(tick))
    appliance = NS(post_step=lambda dt,update_display: {'dt':dt})
    native = dict(counter={'step':0}, observer=observer, appliance=appliance,
                  sample=lambda: sampled.append(native['counter']['step']))
    scene = CookingScene(None, None, native)
    for tick in range(1,121):scene.after_step(tick)
    assert thermal == list(range(1,121))
    assert sampled == list(range(4,121,4))


def test_cooking_rejects_wrong_physics_before_kit():
    spec = InstanceSpec('CP04','synthetic','easy','synthetic','v1','v1','v1',clock=ClockSpec(960,30,30))
    with pytest.raises(ValueError,match='120/30/30'):build_cooking(spec,None)


def test_release_uses_contact_only_without_a_tcp_clearance_gate():
    from mind2act_eval.tasks.CP04.scene import contact_free
    active = set()
    touching = active.__contains__
    assert contact_free(touching, 'jar')
    assert contact_free(touching, 'power', 'knob')
    active.add('jar')
    assert not contact_free(touching, 'jar')
    active.add('knob')
    assert not contact_free(touching, 'power', 'knob')
