"""CP02 final circle/support facts and physical control order."""
from copy import deepcopy

import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP02.evidence import MeasuredTask
from mind2act_eval.tasks.CP02.scorer import score


def spec(level='easy'):
    n = 4 if level == 'easy' else 6
    return dict(difficulty=level, speed_deg_s=10, demonstration_speed_deg_s=0,
                cue_seconds=3, region_inner_outer_radius_m=[.0235,.0275],
                acceptance={'stability_seconds':2},
                items=[dict(identity=str(i), target_cake_local_m=[i*.06,0,.02]) for i in range(n)])


def frame(s, *, supported=True, angle=0, lid=0, start=0, omega=1):
    a=np.deg2rad(angle)/2; l=np.deg2rad(-lid)/2
    R=np.array([[np.cos(2*a),-np.sin(2*a),0],[np.sin(2*a),np.cos(2*a),0],[0,0,1]])
    bodies=[dict(identity=i['identity'], position=(R@np.array(i['target_cake_local_m'])).tolist(),
                 quaternion_wxyz=[0,1,0,0], bilateral=False, touch_robot=True,
                 supported=supported, held_tilt_deg=90) for i in s['items']]
    return dict(rotor=dict(position=[0,0,0],quaternion_wxyz=[np.cos(a),0,0,np.sin(a)],angular_velocity_rad_s=[0,0,omega]),
                lid=dict(quaternion_wxyz=[np.cos(l),np.sin(l),0,0],robot_contact=False),
                bodies=bodies,buttons={'Start':dict(reference_z_m=0,position=[0,0,-start],robot_contact=False)})


def result(task,s,tick):
    return score(NativeEvidence('CP02','cpu',tick,dict(task.grade(),difficulty=s['difficulty'])))


def test_visible_circle_support_needs_no_lift_release_height_tilt_or_wait():
    s=spec(); initial=frame(s,supported=False);task=MeasuredTask(s,{'evaluation_protocol':'physco-task-centric-v2'},initial)
    task.update(360,initial)
    row=frame(s);row['bodies'][0]['position'][0]+=.02;row['bodies'][0]['position'][2]+=.04
    task.update(361,row)
    assert result(task,s,361).value == 100


def test_lid_then_depth_start_accepts_any_contact_and_lid_may_close():
    s=spec('medium');initial=frame(s,supported=False);task=MeasuredTask(s,{'evaluation_protocol':'physco-task-centric-v2'},initial)
    task.update(360,initial)
    task.update(361,frame(s,supported=False,lid=100))
    task.update(362,frame(s,supported=False,lid=100,start=.003))
    task.update(363,frame(s,lid=0,angle=1))
    assert result(task,s,363).value == 100
    task.update(364,frame(s,supported=False))
    assert result(task,s,364).value == 0


def test_stationary_correct_placement_earns_core_credit_without_dynamic_success():
    s=spec('medium');initial=frame(s,supported=False);task=MeasuredTask(s,{'evaluation_protocol':'physco-task-centric-v2'},initial)
    task.update(360,initial);task.update(361,frame(s,supported=False,lid=100))
    task.update(362,frame(s,supported=False,lid=100,start=.003))
    task.update(363,frame(s,omega=0))
    scored = result(task,s,363)
    assert scored.value == 10
    assert not scored.components['task_success']


def hard_frame(s, **kwargs):
    row = frame(s, **kwargs)
    row.update(cover=dict(position=[0,0,0],supported=False,touch_robot=True),
               cover_contains_cake_center=True)
    return row


@pytest.mark.parametrize('level,cap', [('easy', 15), ('medium', 10), ('hard', 7)])
def test_supported_target_circles_earn_partial_credit_before_start_without_ending(level, cap):
    from mind2act_eval.tasks.CP02.scene import CP02Scene
    s = spec(level)
    measured = hard_frame if level == 'hard' else frame
    initial = measured(s, supported=False, omega=0)
    task = MeasuredTask(s, {}, initial)
    task.update(360, initial)
    task.update(361, measured(s, supported=False, lid=100, omega=0))
    task.update(362, measured(s, lid=100, omega=0))
    scored = result(task, s, 362)
    assert scored.value == cap
    assert scored.components['core_k'] == len(s['items'])
    assert not scored.components['task_success']
    scene = object.__new__(CP02Scene)
    scene._task = task
    assert not scene.is_success() and not scene.is_complete()
    row = measured(s, lid=100, omega=0)
    row['bodies'][0]['supported'] = False
    task.update(363, row)
    n = len(s['items'])
    assert result(task, s, 363).value == pytest.approx(cap * (n-1) / n)
    row['bodies'][0]['supported'] = True
    row['bodies'][0]['position'][0] += 1.
    task.update(364, row)
    assert result(task, s, 364).value == pytest.approx(cap * (n-2) / n)
    task.update(365, measured(s, lid=100, omega=0))
    assert result(task, s, 365).value == cap
    assert not scene.is_success()


def test_partial_geometry_does_not_prevent_a_later_legitimate_full_success():
    s = spec('medium')
    initial = frame(s, supported=False, omega=0)
    task = MeasuredTask(s, {}, initial)
    task.update(360, initial)
    task.update(361, frame(s, lid=100, omega=0))
    assert result(task, s, 361).value == 10
    task.update(362, frame(s, supported=False, lid=100, omega=0))
    task.update(363, frame(s, supported=False, lid=100, start=.003, omega=0))
    task.update(364, frame(s, angle=1))
    assert result(task, s, 364).value == 100


def test_hard_stop_after_placement_then_supported_cover_without_wait_or_release():
    s=spec('hard'); initial=hard_frame(s,supported=False);task=MeasuredTask(s,{},initial)
    task.update(360,initial)
    task.update(361,hard_frame(s,supported=False,lid=100))
    task.update(362,hard_frame(s,supported=False,lid=100,start=.003))
    task.update(363,hard_frame(s))
    row=hard_frame(s,omega=0)
    row['buttons']['Stop']=dict(reference_z_m=0,position=[0,0,-.003],robot_contact=False)
    task.update(364,row)
    row['buttons']={};row['cover']['supported']=True
    task.update(365,row)
    assert result(task,s,365).value == 100
    row['cover_contains_cake_center']=False
    task.update(366,row)
    assert result(task,s,366).value < 100
