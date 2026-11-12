"""11-06 authored bands, physical readers remain covered by task integration tests."""
import importlib
from types import SimpleNamespace

import pytest
from mind2act_eval.contracts import NativeEvidence

LEVELS = [('easy',30.),('medium',22.),('hard',15.)]


@pytest.mark.parametrize('task,tolerances,distances',[
    ('PR1',[.2,.15,.12],[8.,4.,2.]),('PR2',[1.,.5,.5],[40.,20.,10.])])
@pytest.mark.parametrize('i,level,cap',[(i,*p) for i,p in enumerate(LEVELS)])
def test_precision_bands_and_exact_tolerance(task,tolerances,distances,i,level,cap):
    score=importlib.import_module(f'mind2act_eval.tasks.{task}.scorer').score
    def grade(error):
        return score(NativeEvidence(task,'fixture',1,dict(difficulty=level,tick_hz=120,
            target_value=100.,first_tick=1,values=[100.+error],touching=[True],consecutive=True)))
    assert grade(tolerances[i]).value==100
    assert grade(tolerances[i]+1e-6).value==pytest.approx(cap*(1-(tolerances[i]+1e-6)/distances[i]))
    assert grade(distances[i]/2).value==pytest.approx(cap/2)
    assert grade(distances[i]).value==0
    assert grade(distances[i]+1).value==0


@pytest.mark.parametrize('level,cap',LEVELS)
def test_e1_core_batches(level,cap):
    from mind2act_eval.tasks.E1.scorer import score
    n={'easy':2,'medium':4,'hard':6}[level]
    e=NativeEvidence('E1','fixture',1,dict(difficulty=level,rounds=[dict(round=1,**{'pass':True})]))
    assert score(e).value==pytest.approx(cap/n)
    full=NativeEvidence('E1','fixture',n,dict(difficulty=level,rounds=[dict(round=i+1,**{'pass':True}) for i in range(n)]))
    assert score(full).value==100


@pytest.mark.parametrize('level,cap',LEVELS)
@pytest.mark.parametrize('count,fraction',[(3,.5),(2,.25)])
def test_s1_authored_bands(level,cap,count,fraction):
    from mind2act_eval.tasks.S1.scorer import score
    blocks=[dict(object_id=str(i),inside=i<count,support=i<count,released=False,
        original_entity_slot_match=i<count,placed_after_program_finished=False,
        stable_seconds=0.,position_range_m=0.) for i in range(4)]
    e=NativeEvidence('S1','fixture',1,dict(difficulty=level,blocks=blocks,early_placement=False,rotation_program_finished=False))
    assert score(e).value==pytest.approx(cap*fraction)


@pytest.mark.parametrize('level,cap',LEVELS)
@pytest.mark.parametrize('missing,fraction',[(1,.5),(2,.25)])
def test_p1_authored_defect_bands(level,cap,missing,fraction):
    from mind2act_eval.tasks.P1.scorer import score
    n={'easy':4,'medium':5,'hard':7}[level]
    foods=[dict(object_id=str(i),inside=True,support=True,released=False,stable_seconds=0.,
        position_range_m=0.,entered_tick=1,container='cup',kind='food') for i in range(n-missing)]
    e=NativeEvidence('P1','fixture',1,dict(difficulty=level,occupancy=foods,occupancy_complete=True,
        ambiguous_membership=[],published_demand={'cup/food':n},all_orders_published=True))
    assert score(e).value==pytest.approx(cap*fraction)


@pytest.mark.parametrize('level,n,low,half,top',[('easy',3,10,20,30),('medium',4,7.5,15,22),('hard',4,5,10,15)])
def test_p2_exact_authored_values_not_implicit_ratio(level,n,low,half,top):
    from mind2act_eval.tasks.P2.stages import stage_value
    def grade(k,done=False):
        return stage_value(valid_disassembly=False,restored_prefix=k,target_count=n,
                           final_conditions_met=done,difficulty=level)
    assert grade(0)==0
    assert grade(1)==low
    assert grade((n+1)//2)==half
    assert grade(n)==top
    assert grade(n,True)==100


@pytest.mark.parametrize('level,cap',LEVELS)
def test_pr3_single_metric_is_capped_not_old_25(monkeypatch,level,cap):
    m=importlib.import_module('mind2act_eval.tasks.PR3.scorer')
    monkeypatch.setattr(m,'metrics',lambda e:dict(coverage=1.,arclength_weighted_rmse_m=.01,
        real_contact_trace=True,nonzero_trace=True))
    e=NativeEvidence('PR3','fixture',1,dict(parameters=dict(level=level),direction_gate=False,normal_completion=False))
    assert m.score(e).value==cap


@pytest.mark.parametrize('level,cap',LEVELS)
@pytest.mark.parametrize('task,original,fraction',[('PR5',15,.5),('PR5',30,1),('PR6',10,.5),('PR6',20,1)])
def test_stamp_and_chain_physical_bands(monkeypatch,level,cap,task,original,fraction):
    m=importlib.import_module(f'mind2act_eval.tasks.{task}.scorer')
    facts=dict(single=True,inside=True,clean=True,heading=original!=15,success=False,score=original)
    monkeypatch.setattr(m,'reconstruct',lambda e:facts)
    v=dict(source_case=dict(level=level)) if task=='PR5' else dict(config=dict(difficulty=level))
    assert m.score(NativeEvidence(task,'fixture',1,v)).value==pytest.approx(cap*fraction)


@pytest.mark.parametrize('level,cap',LEVELS)
def test_dr1_core_parcels(monkeypatch,level,cap):
    m=importlib.import_module('mind2act_eval.tasks.DR1.scorer');n={'easy':3,'medium':5,'hard':7}[level]
    monkeypatch.setattr(m,'parcel_state',lambda rows,i,tray:dict(placed=i==0))
    e=NativeEvidence('DR1','fixture',1,dict(count=n,ticks=[dict(tick=1,parcels=[{}]*n,belt_velocity=[0.,0.,0.])],
        tray_bounds_m=[],belt_speed_m_s=0.))
    assert m.score(e).value==pytest.approx(cap/n)


@pytest.mark.parametrize('level,cap',LEVELS)
def test_dr2_independent_correct_cues(monkeypatch,level,cap):
    m=importlib.import_module('mind2act_eval.tasks.DR2.scorer')
    monkeypatch.setattr(m,'replay',lambda v:SimpleNamespace(records=[],valid_count=3,sequence=list(range(10)),unlit_presses=[],done=True))
    e=NativeEvidence('DR2','fixture',1,dict(difficulty=level,live_records=[]))
    assert m.score(e).value==pytest.approx(cap*3/10)


@pytest.mark.parametrize('level,cap',LEVELS)
def test_dr4_recovered_balls_band(level,cap):
    from mind2act_eval.tasks.DR4.scorer import grade_facts
    facts=dict(no_obstacle_contacts=True,current_delivered=True,all_balls_currently_in_circles=True,all_balls_retained=False)
    assert grade_facts(facts,difficulty=level)==cap
    assert grade_facts(dict(facts,all_balls_retained=True),difficulty=level)==100


@pytest.mark.parametrize('level,cap',LEVELS)
@pytest.mark.parametrize('missing,fraction',[(1,.5),(2,.25)])
def test_m1_route_defects(monkeypatch,level,cap,missing,fraction):
    m=importlib.import_module('mind2act_eval.tasks.M1.scorer');n={'easy':4,'medium':5,'hard':6}[level]
    expected=list('ABCDEF')[:n]
    monkeypatch.setattr(m,'analyse',lambda v:dict(route=expected[:-missing],uncovery=[],extracted_at=None,placements=[]))
    monkeypatch.setattr(m,'Ledger',lambda *a:SimpleNamespace(station_at=lambda *a:expected[-1]))
    e=NativeEvidence('M1','fixture',1,dict(expected_path=expected,target_cover_id='cover',ticks=[dict(target_center=[0,0,0],station_support={})],submission_stations={},target_dimensions=[1,1,1]))
    assert m.score(e).value==pytest.approx(cap*fraction)


@pytest.mark.parametrize('level,cap',LEVELS)
@pytest.mark.parametrize('released,fraction',[(True,.5),(False,.25)])
def test_f1_goal_bands(monkeypatch,level,cap,released,fraction):
    m=importlib.import_module('mind2act_eval.tasks.F1.scorer')
    grade=dict(protocol_violations=[],prerequisites=True,task_success=False,at_target=True,final_released=released,
        execution_presses=4,shortest_presses=2,path_redundancy=2,effective_probes=4)
    monkeypatch.setattr(m,'Judge',lambda *a:SimpleNamespace(sample=lambda row:grade))
    space=dict(kind='line') if level=='easy' else dict(kind='grid',side=3 if level=='medium' else 4)
    assert m.score(NativeEvidence('F1','fixture',1,dict(case=dict(space=space),protocol={},rows=[{}]))).value==pytest.approx(cap*fraction)


@pytest.mark.parametrize('level,expected',[('easy',15),('medium',11),('hard',7.5)])
def test_m2_core_pairs_partial_for_all_tiers(level,expected):
    from mind2act_eval.tasks.M2.scorer import score
    n={'easy':2,'medium':4,'hard':8}[level]
    pairs=[dict(cards=[2*i,2*i+1],valid_press_release=True,same_picture=True,physical_faceup=True) for i in range(n//2)]
    e=NativeEvidence('M2','fixture',1,dict(schema='m2.physical.v1',difficulty=level,card_count=2*n,
        angle_measurement='hinge',card_angle_errors_deg=[180.]*(2*n),all_buttons_released=True,normal_termination=True,
        max_unmatched_faceup=2,pairs=pairs,binding_mode='fixture'))
    assert score(e).value==expected


@pytest.mark.parametrize('level,cap',LEVELS)
@pytest.mark.parametrize('held',[False,True])
def test_e2_position_matches_not_lcs_and_return_required_for_100(monkeypatch,level,cap,held):
    m=importlib.import_module('mind2act_eval.tasks.E2.scorer')
    n={'easy':2,'medium':3,'hard':4}[level];target=list(range(1,n+1))
    fake=SimpleNamespace(digits=target if held else [1]+[9]*(n-1),released=not held,last_tick=0,serial_violation=False)
    fake.sample=lambda tick,*a,**kw:setattr(fake,'last_tick',tick)
    fake.complete=lambda goal:fake.digits==list(goal) and fake.released
    monkeypatch.setattr(m,'Keys',lambda:fake)
    monkeypatch.setattr(m,'read_measurements',lambda r:[dict(tick=1,travel_m=[0]*9,fingertip_contacts=[False]*9)])
    e=NativeEvidence('E2','fixture',1,dict(difficulty=level,target_digits=target,
        timing=dict(learn_seconds=2.,shuffle_seconds=.5,reasoning_seconds=.25),measurements={}))
    assert m.score(e).value==pytest.approx(cap if held else cap/n)
    fake.digits=target[::-1];fake.released=True
    assert m.score(e).value==pytest.approx(cap*sum(a==b for a,b in zip(fake.digits,target))/n)
    fake.digits=target
    assert m.score(e).value==100


@pytest.mark.parametrize('level,expected',[('easy',7.5),('medium',5.5),('hard',3.75)])
def test_dr3_ordered_gates_partial(monkeypatch,level,expected):
    m=importlib.import_module('mind2act_eval.tasks.DR3.scorer')
    result=dict(evidence_complete=True,unknown_reasons=[],score=5.,success=False,
        event_detector_contract='test',measurement_contract='test',events=[],attempts=[],windows=[],zero_proof=None)
    result.update({key:False for key in ('violation_latched','passed_aperture','passed_aperture_count',
        'delivered','terminal_tray_support','center_inside_tray','cutoff_simulation_time_s')})
    monkeypatch.setattr(m,'read_manifest',lambda r:{})
    monkeypatch.setattr(m,'read_trace',lambda *a:[])
    monkeypatch.setattr(m,'rebuild_events',lambda *a:result)
    e=NativeEvidence('DR3','fixture',1,dict(difficulty=level,source_id=m.SOURCE_ID,condition=dict(instance_id='fixture'),
        manifest=dict(path='/test/manifest.json'),raw_trace=dict(path='/test/raw-trace.jsonl')))
    assert m.score(e).value==expected


@pytest.mark.parametrize('task,target,key,tolerances',[
    ('PR1',10.,'target_ohm',[.2,.15,.12]),('PR2',100.,'target_volt',[1.,.5,.5])])
@pytest.mark.parametrize('i,level',list(enumerate(['easy','medium','hard'])))
def test_live_success_uses_the_same_tier_tolerance_as_scorer(task,target,key,tolerances,i,level):
    m=importlib.import_module(f'mind2act_eval.tasks.{task}.scene')
    cls=m.RheostatScene if task=='PR1' else m.VoltageScene
    scene=cls.__new__(cls);scene.spec=SimpleNamespace(difficulty=level,config={key:target})
    scene._recorder=SimpleNamespace(last=target+tolerances[i]+1e-6)
    assert not scene.is_success()
    scene._recorder.last=target+tolerances[i]
    assert scene.is_success()
