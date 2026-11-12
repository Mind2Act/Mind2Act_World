from math import sqrt
from types import SimpleNamespace
import numpy as np
import pytest
from mind2act_eval.tasks.CP08.tcp import measured_tcp
from mind2act_eval.tasks.CP08.sensors import Sensors, Contacts
from mind2act_eval.tasks.CP08.instances import selected
from mind2act_eval.tasks.CP08.replay import Replay


def test_source_tcp_offset_is_rotated_in_actual_link6_frame():
    assert measured_tcp([1.,2.,3.],[1.,0.,0.,0.])==[1.145,2.,3.]
    assert measured_tcp([1.,2.,3.],[sqrt(.5),0.,0.,sqrt(.5)])==pytest.approx([1.,2.145,3.])
    with pytest.raises(ValueError):measured_tcp([0.,0.,0.],[0.,0.,0.,0.])


def test_sensor_to_raw_and_replay_binds_actual_flange_not_command():
    cfg=selected('easy')['source_case']
    class Body:
        def __init__(self,p):self.p=p
        def get_world_pose(self):return np.asarray(self.p),np.asarray([1.,0.,0.,0.])
        def get_linear_velocity(self):return np.zeros(3)
        def get_world_poses(self):
            p,q=self.get_world_pose();return np.asarray([p]),np.asarray([q])
    class Arm:
        def get_joint_positions(self):return np.r_[np.zeros(6),.044,.044]
    screen=SimpleNamespace(Get=lambda:SimpleNamespace(path='plan.png'))
    sensors=Sensors(None,{b['id']:Body(b['initial']) for b in cfg['blocks']},
        {s:Arm() for s in ('Left','Right')},screen,{'plan.png':'plan.png'},Contacts(),object(),
        {'Left':Body([1.,2.,3.]),'Right':Body([2.,3.,4.])})
    raw=sensors.read(1)
    assert raw['tcps']['Right']==[2.145,3.,4.]
    row=Replay(cfg).sample(raw)
    assert row['tcp_by_arm']['Right']==raw['tcps']['Right']
    raw['tcps']['Right'][0]+=.01
    with pytest.raises(ValueError,match='source flange frame'):Replay(cfg).sample(raw)


def test_actual_flange_view_constructor_disables_implicit_robot_physics_writes():
    from mind2act_eval.tasks.CP08.tcp import flange_readers
    calls=[]
    class GuardedSdkView:
        def __init__(self,**kw):
            # Model the SDK defaults that caused the original ownership bug.
            assert kw.get('prepare_contact_sensors',True) is False
            assert kw.get('track_contact_forces',False) is False
            assert kw.get('reset_xform_properties',True) is False
            assert kw.get('disable_stablization',True) is False
            assert kw.get('contact_filter_prim_paths_expr')==[]
            calls.append(kw)
    stage=SimpleNamespace(GetPrimAtPath=lambda p:SimpleNamespace(HasAPI=lambda api:True))
    world=SimpleNamespace(scene=SimpleNamespace(add=lambda view:view))
    views=flange_readers(stage,world,view_type=GuardedSdkView,non_root=lambda **kw:True)
    assert set(views)=={'Left','Right'} and len(calls)==2
    assert {c['prim_paths_expr'] for c in calls}=={'/World/RobotLeft/Asset/link6','/World/RobotRight/Asset/link6'}


@pytest.mark.parametrize('has_api,nonroot',[(False,True),(True,False)])
def test_reader_rejects_missing_rigid_api_or_root_before_sdk_can_author(has_api,nonroot):
    from mind2act_eval.tasks.CP08.tcp import flange_readers
    stage=SimpleNamespace(GetPrimAtPath=lambda p:SimpleNamespace(HasAPI=lambda api:has_api))
    def forbidden(**kw):raise AssertionError('SDK must not create APIs/reset root robot state')
    with pytest.raises(ValueError,match='existing non-root'):
        flange_readers(stage,SimpleNamespace(),view_type=forbidden,non_root=lambda **kw:nonroot)
