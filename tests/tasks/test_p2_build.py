"""CPU assembly contract: original settle precedes all scored native evidence."""
from types import ModuleType, SimpleNamespace as NS
import sys

from .test_p2_measurements import source
from mind2act_eval.tasks.P2 import scene
from mind2act_eval.tasks.P2.binding import geometry


def test_source_settle_holds_shared_rest_without_consuming_episode_samples(monkeypatch):
    row=source();g=geometry(row);calls=[]
    bodies=[]
    class Body:
        def __init__(self,**kwargs):
            self.path=kwargs['prim_path'];bodies.append(self)
        def set_linear_velocity(self,value):
            assert list(value)==[0.,0.,0.]
        def set_angular_velocity(self,value):
            assert list(value)==[0.,0.,0.]
    module=ModuleType('isaacsim.core.prims');module.SingleRigidPrim=Body
    monkeypatch.setitem(sys.modules,'isaacsim.core.prims',module)
    monkeypatch.setattr(scene,'validate',lambda spec,ctx:(row,g))
    monkeypatch.setattr(scene,'author_task',lambda *a:calls.append('author'))
    monkeypatch.setattr(scene,'prepare_task_contacts',lambda *a:None)
    class Contacts:
        def subscribe(self):calls.append('subscribe')
        def close(self):calls.append('close')
    monkeypatch.setattr(scene,'Manifolds',Contacts)
    def measured(*args):
        assert calls.count('step')==960
        assert calls.count('apply')==960
        calls.append('evidence_tick0')
        return 'ready'
    monkeypatch.setattr(scene,'MeasurementScene',measured)
    rest=[0.,1.,2.,-1.,0.,0.,.044,.044]
    def apply(action):
        assert action.left_arm==tuple(rest[:6]) and action.right_arm==tuple(rest[:6])
        assert action.left_gripper==1. and action.right_gripper==1.
        calls.append('apply')
    def step(*,render):
        assert render is False
        calls.append('step')
    ctx=NS(sim=NS(stage=NS(GetPrimAtPath=lambda path:True),step=step),assets=NS(),
           robot=NS(prepare_stage=lambda stage:None,initialize=lambda world:calls.append('initialize'),
                    articulations=lambda world:{},apply=apply),
           world=NS(scene=NS(add=lambda body:body)))
    assert scene.build(NS(config={'robot_rest':rest}),ctx)=='ready'
    assert calls[:3]==['author','initialize','subscribe']
    assert calls[-1]=='evidence_tick0' and 'close' not in calls
    assert len(bodies)==4
