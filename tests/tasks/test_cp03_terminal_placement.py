"""Measured CPU scene path: ordered Start, carried dynamic entry and terminal slots."""
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from mind2act_eval.contracts import InstanceSpec, NativeEvidence
from mind2act_eval.tasks.CP03 import scene as module
from mind2act_eval.tasks.CP03.scorer import RULE_VERSION, score


@pytest.fixture
def rack(monkeypatch, request):
    level, protocol, *pickup_flag = getattr(request, 'param', ('easy', 'valid'))
    count = {'easy':3, 'medium':5, 'hard':7}[level]
    mode = {'easy':1, 'medium':2, 'hard':3}[level]
    stage = Usd.Stage.CreateInMemory()
    base = '/World/DishLoading'
    carriage = UsdGeom.Xform.Define(stage, base+'/Carriage')
    for i in range(count+1):
        for k in (0, 2):
            cube = UsdGeom.Cube.Define(stage, base+f'/Carriage/Cradle{i}/Side{k}')
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    sources = ['/World/LiveDishes/Source'+str(i) for i in range(count)]
    preload = '/World/LiveDishes/Preloaded'+str(count)

    class Body:
        def __init__(self, pos): self.pos=np.array(pos,dtype=float);self.q=np.array([1.,0.,0.,0.])
        def get_world_pose(self): return self.pos.copy(),self.q.copy()

    bodies={p:Body([.5+i*.1,0.,.75]) for i,p in enumerate(sources)}
    bodies[preload]=Body([.09,0.,.823])
    for p in bodies: UsdGeom.Cube.Define(stage,p).CreateSizeAttr(.01)
    support=set();held=set();state=dict(down=False,gear_touch=False,activation=None)
    def touch(path,**kw):
        if path=='start': return state['down']
        if path=='gear': return state['gear_touch']
        return path in (held if '/Robot' in kw.get('other','') else support)
    contacts=SimpleNamespace(touch=touch,supported=touch,bilateral=lambda p:p in held)
    spec=InstanceSpec('CP03','cpu-drop',level,'cpu-source','v1','cpu-assets','v1')
    ctx=SimpleNamespace(sim=SimpleNamespace(contacts=contacts),robot=SimpleNamespace())
    def start(t): state['activation']=t
    ns=dict(S=dict(mode=mode,total_slots=count+1,empty_slot_indices=list(range(count)),slot_pitch_m=.06,slot_width_m=.05),
        bodies=bodies,sources=sources,preloaded=[preload],controls={},control_paths={},control_origins={},
        BASE=base,stage=stage,initial_rail=0.,carriage_base=[0.,0.,.7],carriage_op=carriage.AddTranslateOp(),
        rail_x=lambda t: 0. if state['activation'] is None else .012*(t-state['activation']),start_motor=start)
    config=dict(difficulty=level,speed_m_s=.012,max_transport_tilt_deg=3.,transient_peak_limit_deg=6.,max_contiguous_excursion_s=.05,max_total_excursion_s=.1)
    monkeypatch.setattr(module,'public_instruction',lambda spec:'CPU task')
    monkeypatch.setattr(module.CP03Scene,'_subscribe_robot_contacts',lambda self:None)
    s=module.CP03Scene(spec,ctx,ns,{},config)
    s._controls['start']=SimpleNamespace(get_world_pose=lambda:(np.array([0.,0.,-.002 if state['down'] else 0.]),[1.,0.,0.,0.]))
    s._origins['start']=np.zeros(3);s._paths['start']='start'
    if level!='easy':
        angle=np.radians([18.,0.,-18.][mode-1])
        s._controls['gear']=SimpleNamespace(get_world_pose=lambda:(np.zeros(3),[np.cos(angle/2),np.sin(angle/2),0.,0.]))
        s._paths['gear']='gear'
        attr=SimpleNamespace(Set=lambda value:None)
        drive=SimpleNamespace(GetTargetPositionAttr=lambda:attr,GetStiffnessAttr=lambda:attr,GetMaxForceAttr=lambda:attr)
        s._UsdPhysics=SimpleNamespace(DriveAPI=SimpleNamespace(Get=lambda *args:drive))
    def run(step, seated=(), *, drift=0., height=0., tilt=False, slots=None, held_seated=False, supported=True):
        s._tick=step;support.clear()
        for name in seated:
            p=sources[name]
            if supported:support.add(p)
            if held_seated:held.add(p)
            else:held.discard(p)
            local=.06*((slots or {}).get(name,name)-count/2)
            bodies[p].pos=np.array([local+s._ns['rail_x'](step/120)+drift,0.,.823+height])
            bodies[p].q=np.array([1.,0.,0.,0.]) if tilt else np.array([sqrt(.5),0.,sqrt(.5),0.])
        s._control_and_placement(step);s._sample_document_stage()
    # Exercise the real source control protocol before any plate transfer.
    for step in range(1,123):
        state['gear_touch']=step==1
        state['down']=protocol=='early' or (protocol=='valid' and step==122)
        run(step)
    state['down']=False;state['gear_touch']=False
    run(123);run(124)
    def pickup(names):
        for i in names:
            p=sources[i];held.add(p);bodies[p].pos=s._initial[p]+[0.,0.,.05]
        run(s._tick+1)
        for i in names:held.discard(sources[i])
    if not pickup_flag or pickup_flag[0]:
        pickup(range(count))
    offset=s._tick
    def tick(i,seated=(),**kw): run(offset+i,seated,**kw)
    s._cpu_run=run;s._cpu_state=state
    return s,tick,sources,bodies,preload,support


@pytest.mark.parametrize('rack',[(x,'valid') for x in ('easy','medium','hard')],indirect=True)
@pytest.mark.parametrize('held_entry',[False,True])
def test_carried_plate_in_moving_slot_counts_with_or_without_gripper_overlap(rack,held_entry):
    s,tick,sources,_,_,_=rack
    for i in range(1,11):tick(i)
    tick(11,(0,),held_seated=held_entry)
    for i in range(12,132):tick(i,(0,))
    e=s.evidence(s._tick)
    assert e.values['credited_sources']==(sources[0],)
    first=e.values['slot_entries'][sources[0]]
    assert first['dynamic'] and first['held'] is held_entry
    assert first['rack_speed_m_s']>.0005
    assert score(e).value>0


def test_all_carried_source_plates_can_finish_despite_diagnostic_quality_history(rack):
    s,tick,sources,_,_,_=rack
    s._failures.update(('transport_tilt','preloaded_plate_disturbed'))
    for i in range(1,121):tick(i,range(len(sources)))
    assert score(s.evidence(s._tick)).value==100.


@pytest.mark.parametrize('failure',['height','normal','short','unstable','unsupported'])
def test_terminal_placement_ignores_removed_qualification_gates(rack,failure):
    s,tick,_,_,_,_=rack
    end=119 if failure=='short' else 120
    for i in range(1,end+1):
        tick(i,() if failure=='unsupported' else (0,),height=.013 if failure=='height' else 0.,
             tilt=failure=='normal',drift=.006 if failure=='unstable' and i%2 else 0.)
    assert s.evidence(s._tick).values['credited_plates']==int(failure != 'unsupported')


def test_preloaded_plate_is_never_a_task_plate(rack):
    s,tick,_,_,_,_=rack
    for i in range(1,121):tick(i)
    assert s.evidence(s._tick).values['credited_plates']==0


@pytest.mark.parametrize('rack',[('easy','valid',False)],indirect=True)
def test_pushed_source_without_measured_pickup_cannot_count(rack):
    s,tick,sources,_,_,_=rack
    for i in range(1,121):tick(i,(0,))
    assert s.evidence(s._tick).values['credited_plates']==0


def test_two_task_plates_in_one_slot_are_not_credited(rack):
    s,tick,_,_,_,_=rack
    for i in range(1,121):tick(i,(0,1),slots={1:0})
    assert s.evidence(s._tick).values['credited_plates']==0


@pytest.mark.parametrize('version',[None,'cp03-terminal-placement-20261104-v2','cp03-dynamic-terminal-placement-20261104-v3'])
def test_old_aggregate_cannot_be_relabeled_as_current_rule(version):
    with pytest.raises(ValueError,match='old aggregate cannot be rescored'):
        score(NativeEvidence('CP03','old',120,dict(scoring_version=version,difficulty='easy',
            required_steps=4,steps_satisfied=1,wrong_or_extra_items=0,task_success=False)))


@pytest.mark.parametrize('rack',[('easy','missing'),('medium','missing'),('hard','missing'),('medium','early'),('hard','early')],indirect=True)
def test_missing_start_or_start_before_gear_does_not_enter_plate_scoring(rack):
    s,tick,_,_,_,_=rack
    assert not s._motor['started']
    # External motion alone cannot replace an ordered physical Start.
    s._ns['rail_x']=lambda t:.012*t
    for i in range(1,121):tick(i,(0,))
    e=s.evidence(s._tick)
    assert e.values['credited_plates']==0
    assert not e.values['valid_start']
    assert not e.values['task_success']
    assert score(e).value==0.  # Existing wrong-item penalty must survive without Start.


@pytest.mark.parametrize('speed',[0.,.0005,.0005001])
def test_slot_entry_uses_real_motion_without_extra_speed_threshold(rack,speed):
    s,tick,_,_,_,_=rack
    now=s._ns['rail_x'](s._tick/120);start=s._tick/120
    s._ns['rail_x']=lambda t:now+speed*(t-start)
    for i in range(1,121):tick(i,(0,))
    assert s.evidence(s._tick).values['credited_plates']==int(speed>0)


def test_starting_motion_after_stationary_placement_cannot_upgrade_it(rack):
    s,tick,sources,_,_,_=rack
    now=s._ns['rail_x'](s._tick/120)
    s._ns['rail_x']=lambda t:now
    tick(1,(0,))
    start=s._tick/120;s._ns['rail_x']=lambda t:now+.012*(t-start)
    for i in range(2,122):tick(i,(0,))
    e=s.evidence(s._tick)
    assert e.values['credited_plates']==0
    assert e.values['slot_entries'][sources[0]]['rack_speed_m_s']==0


def test_plate_entering_before_start_cannot_be_qualified_by_later_start(rack):
    s,tick,_,_,_,_=rack
    # A received plate existed before Start, even if the rack later moves.
    s._start_tick=s._tick+1000
    tick(1,(0,));s._start_tick=122
    for i in range(2,122):tick(i,(0,))
    assert s.evidence(s._tick).values['credited_plates']==0


def test_entry_while_moving_can_get_support_later_without_a_first_support_gate(rack):
    s,tick,sources,_,_,_=rack
    tick(1,(0,),supported=False)
    now=s._ns['rail_x'](s._tick/120);s._ns['rail_x']=lambda t:now
    for i in range(2,122):tick(i,(0,))
    e=s.evidence(s._tick)
    assert e.values['credited_plates']==1
    assert not e.values['slot_entries'][sources[0]]['supported']


def test_stationary_placement_can_be_removed_and_reinserted_while_moving(rack):
    s,tick,src,bodies,_,_=rack
    now=s._ns['rail_x'](s._tick/120);s._ns['rail_x']=lambda t:now
    tick(1,(0,))
    bodies[src[0]].pos=np.array([.5,0.,.9]);tick(2)
    start=s._tick/120;s._ns['rail_x']=lambda t:now+.012*(t-start)
    for i in range(3,123):tick(i,(0,))
    assert s.evidence(s._tick).values['credited_plates']==1


@pytest.mark.parametrize('rack',[('easy','missing')],indirect=True)
def test_pre_start_placement_can_be_picked_up_and_reinserted_after_valid_start(rack):
    s,tick,src,bodies,_,_=rack
    for i in range(1,121):tick(i,(0,))
    assert s.evidence(s._tick).values['credited_plates']==0
    # A new physical press/release starts the rack; an actual new entry follows.
    s._cpu_state['down']=True;s._cpu_run(s._tick+1)
    s._cpu_state['down']=False;s._cpu_run(s._tick+1)
    bodies[src[0]].pos=np.array([.5,0.,.9]);s._cpu_run(s._tick+1)
    for i in range(120):s._cpu_run(s._tick+1,(0,))
    e=s.evidence(s._tick)
    assert e.values['credited_plates']==1
    assert e.values['valid_start']
    assert score(e).value==5.


@pytest.mark.parametrize('rack',[('easy','valid',False)],indirect=True)
def test_single_finger_push_is_not_a_gripper_transfer(rack):
    s,tick,sources,bodies,_,_=rack
    s._contacts.bilateral=lambda path:False
    s._finger=lambda path:path==sources[0]
    bodies[sources[0]].pos=s._initial[sources[0]]+[.01,0,0]
    tick(1)
    tick(2,(0,))
    e=s.evidence(s._tick)
    assert not e.values['slot_entries'][sources[0]]['gripper_transport']
    assert e.values['credited_plates']==0


@pytest.mark.parametrize('rack',[('easy','valid',False)],indirect=True)
def test_pushing_while_supported_is_not_a_free_gripper_carry(rack):
    s,tick,sources,_,_,_=rack
    s._contacts.bilateral=lambda path:path==sources[0]
    s._finger=lambda path:path==sources[0]
    s._contacts.supported=lambda *args,**kw:True
    tick(1,(0,))
    assert not s.evidence(s._tick).values['slot_entries'][sources[0]]['gripper_transport']
