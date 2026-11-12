"""CPU native-fact fixtures; no expert labels or simulated success claims."""
from copy import deepcopy

import pytest

from mind2act_eval.tasks.F2.mechanism import Task
from mind2act_eval.tasks.F2.provenance import canonical, effective_config, VERSION, SOURCE_VERSION, BUDGET


def source_case(level='easy'):
    n={'easy':3,'medium':4,'hard':5}[level]
    names=[f'block{i+1}' for i in range(n)]
    c=dict(task='F2',version=SOURCE_VERSION,difficulty=level,seed=1,physics_hz=120,control_hz=30,
           cue_ticks=600,feedback_ticks=360,stable_ticks=60,max_ticks=108000,max_linear_speed=.015,
           max_angular_speed=.15,support_tolerance=.004,scene_profile='mind-aligned',
           objects={name:dict(position=[i*.15,-.2,.788],size=[.05,.05,.04]) for i,name in enumerate(names)},
           regions={f'slot{i+1}':dict(center=[i*.15,.1],size=[.1,.1],floor_z=.768) for i in range(n)},
           target=list(reversed(names)),attempt_budget={'easy':6,'medium':12,'hard':20}[level],
           button=dict(travel=.01004,release_depth=.001,press_depth=.0035,min_contact_ticks=3,require_confirm_release=True,
                       position=[.475,-.055,.834672],rest_joint_position_m=.005,initial_joint_positions_m=[.005]))
    c['case_sha256']=canonical(c)
    return c


class Facts:
    def __init__(self,level='easy',offset=0):
        self.source=source_case(level)
        self.c=effective_config(self.source,input_version=VERSION)
        self.machine=Task(self.c,source_start_tick=offset)
        self.tick=offset
        self.layout=None
        self.machine.sample(self.frame())

    def frame(self,pressed=False,held=None,missing_support=False):
        bodies={}
        for name,spec in self.c['objects'].items():
            position=list(spec['position']);contacts=[];supports=[]
            if self.layout and name in self.layout:
                region_id=list(self.c['regions'])[self.layout.index(name)]
                region=self.c['regions'][region_id]
                position=[*region['center'],region['floor_z']+spec['size'][2]/2]
                floor=f'/World/Task/Regions/{region_id}/floor'
                if not missing_support:
                    contacts.append(floor)
                    supports.append(dict(path=floor,position=[*position[:2],region['floor_z']],normal=[0.,0.,1.],separation=0.))
            if name==held:contacts.append('/World/RobotLeft/Asset/link7/Collision')
            half=[v/2 for v in spec['size']]
            bodies[name]=dict(lower=[p-h for p,h in zip(position,half)],upper=[p+h for p,h in zip(position,half)],
                position=position,orientation_wxyz=[1.,0.,0.,0.],
                linear=[0.,0.,0.],angular=[0.,0.,0.],contacts=contacts,support_contacts=supports)
        depth=.004 if pressed else 0.
        spec=self.c['button'];p=[*spec['position'][:2],spec['position'][2]-depth]
        button=dict(depth=depth,robot_contact=pressed,pose_depth_m=depth,position=p,orientation_wxyz=[1.,0.,0.,0.],
                    joint_position_m=[spec['rest_joint_position_m']-depth],joint_velocity_m_s=[0.])
        events=[]
        if pressed:
            events.append(dict(pair=['/World/Task/Confirm/Asset/E_button_01_13/Collision','/World/RobotLeft/Asset/link7/Collision'],
                active=True,contact_count=1,min_separation_m=0.,points=[dict(position=p,normal=[0.,0.,1.],separation=0.)]))
        return dict(tick=self.tick,bodies=bodies,button=button,contact_events=events)

    def step(self,count=1,**kwargs):
        for _ in range(count):
            self.tick+=1
            self.machine.sample(self.frame(**kwargs))

    def place(self,layout):
        self.layout=list(layout)
        self.step(60)

    def confirm(self):
        before=len(self.machine.attempts)
        self.step(3,pressed=True)
        assert len(self.machine.attempts)==before  # press cannot consume a valid submission
        self.step()  # exactly ONE physical release frame


def test_source_global_cutoff_reserves_environment_observation():
    f = Facts()
    source = source_case()
    source['max_ticks'] = 602
    source['case_sha256'] = canonical(source)
    f.c = effective_config(source, input_version=VERSION)
    f.machine = Task(f.c)
    f.machine.sample(f.frame())
    f.step(602)
    assert f.machine.status == 'running'


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_third_confirm_after_single_frame_release_is_recognized(level):
    f=Facts(level);f.step(600)
    wrong=list(f.c['objects'])
    f.place(wrong)
    f.confirm();f.confirm()
    assert len(f.machine.attempts)==2 and f.machine.armed
    f.place(f.c['target'])
    f.confirm()
    assert f.machine.status=='success' and len(f.machine.attempts)==3
    assert not f.machine.pending_release


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_last_budget_submission_is_not_terminal_until_release(level):
    f=Facts(level);f.step(600);f.place(list(f.c['objects']))
    for _ in range(BUDGET[level]-1):f.confirm()
    f.place(f.c['target'])
    f.step(3,pressed=True)
    assert f.machine.status=='running' and len(f.machine.attempts)==BUDGET[level]-1
    f.step()
    assert f.machine.status=='success' and len(f.machine.attempts)==BUDGET[level]


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_budget_exhaustion_and_incomplete_layouts(level):
    f=Facts(level);f.step(600);f.place(list(f.c['objects'])[:-1])
    f.confirm()
    assert not f.machine.attempts
    f.place(list(f.c['objects']))
    for _ in range(BUDGET[level]):f.confirm()
    assert f.machine.status=='failure' and f.machine.reason=='attempt_budget_exhausted'
    assert len(f.machine.attempts)==BUDGET[level]


def test_support_is_required_but_light_robot_touch_is_not_a_submission_gate():
    for kwargs, count in [({'held':'block1'},1),({'missing_support':True},0)]:
        f=Facts();f.step(600);f.layout=list(f.c['objects']);f.step(60,**kwargs)
        f.step(3,pressed=True,**kwargs);f.step(**kwargs)
        assert len(f.machine.attempts) == count


def test_long_hold_cannot_repeat_or_consume_budget_before_release():
    f=Facts();f.step(600);f.place(list(f.c['objects']))
    f.step(120,pressed=True)
    assert not f.machine.attempts and f.machine.pending_release
    f.step()
    assert len(f.machine.attempts)==1 and f.machine.armed
    f.step(120)
    assert len(f.machine.attempts)==1


def test_source_clock_offset_does_not_fabricate_physics_or_stability_samples():
    f=Facts(offset=600)
    assert f.machine.tick==600 and f.machine.physics_samples==0
    f.layout=list(f.c['objects'])
    f.step()
    assert len(f.machine.stable_locations())==3 and f.machine.physics_samples==1
    assert f.machine.tick==601


def test_current_budget_does_not_inherit_the_old_diagnostic_budget():
    source=source_case('hard');before=deepcopy(source)
    for version in (VERSION,SOURCE_VERSION):
        current=effective_config(source,input_version=version)
        assert current['attempt_budget']==7 and current['source_attempt_budget']==20
    assert source==before


def test_notice_touch_does_not_fail_and_subsequent_confirmation_is_unchanged():
    f = Facts()
    f.step(10, held='block1')
    assert f.machine.status == 'running'
    assert not any(event.get('reason') == 'manipulation_during_notice' for event in f.machine.events)
    f.layout = list(f.c['target'])
    f.step()
    # The observed physical move is kept; the mechanism does not restore it.
    region = f.c['regions']['slot1']
    block = f.c['target'][0]
    assert tuple(f.machine.bodies[block].center[:2]) == tuple(region['center'])
    f.step(600 - f.tick)
    f.step(60)
    f.confirm()
    assert f.machine.status == 'success' and len(f.machine.attempts) == 1
