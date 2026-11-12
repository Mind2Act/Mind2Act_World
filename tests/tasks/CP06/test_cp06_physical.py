from copy import deepcopy
import pytest
from mind2act_eval.tasks.CP06.instances import selected
from mind2act_eval.tasks.CP06.physical import evaluate


def fixture(level,load=False,zero=False,green=False,rider=False):
    cfg=selected(level)['source_task'];definitions=cfg['all_objects']
    objects={o['path']:dict(robot_contact=False,external_supported=False,linear_velocity_m_s=[0,0,0],position=list(o['initial_position_m']),finger_contacts=[False,False],
        pan_supported={'Left':False,'Right':False},pan_center_local_m={'Left':[0.,0.,.02],'Right':[0.,0.,.02]},max_pan_local_radius_m=.01,min_pan_local_z_m=.02) for o in definitions}
    initial=dict(pan_bottom_local_m=dict(Left=0.,Right=0.),simulation_time_s=0.,objects=deepcopy(objects))
    fruits=[o['path'] for o in definitions if o['kind']=='fruit']
    weights=[o['path'] for o in definitions if o['kind']=='weight'][:2]
    loaded=fruits+weights if load else []
    records=[]
    for tick in range(1,601):
        state_objects=deepcopy(objects)
        for index,path in enumerate(loaded):
            start=70+index*2
            if tick==start:
                state_objects[path]['position'][2]+=.05;state_objects[path]['finger_contacts']=[True,True]
            if tick>start:
                objects[path]['pan_supported']['Left' if path in fruits else 'Right']=True
                state_objects[path]['pan_supported']=deepcopy(objects[path]['pan_supported'])
        calibrated=level=='easy' or zero and tick>=62
        beam=0. if green or tick<200 else 5.
        q=.09 if rider and tick>202 else .01 if rider and tick==200 else .05 if rider and tick==201 else 0.
        contact=rider and tick in (200,201)
        if contact:beam=.1 if tick==200 else .6
        state=dict(pan_bottom_local_m=dict(Left=0.,Right=0.),simulation_time_s=tick/120,objects=state_objects,beam_deg=beam,
            corrected_angle_deg=beam,reading_valid=calibrated,rider_m=q,rider_velocity_m_s=0.,
            rider_fingers=[contact,contact],unexpected_robot_contacts=[],carried_fruit_neighbor_contacts=[],
            robot_base_motion_m=0.,balance_base_motion_m=0.,steady_rim_contacts=[],carried_fruit_rim_contacts=[],
            control=dict(button_travel_m=.004 if zero and tick==62 else 0.,brake_locked=False),
            new_control_events=[dict(type='calibration_press',accepted=True)] if zero and tick==62 else [])
        records.append(dict(physics_tick=tick,simulation_time_s=tick/120,raw=state))
    return cfg,records,initial


@pytest.mark.parametrize('level,n',[('easy',4),('medium',6),('hard',8)])
def test_empty_green_pointer_credits_only_independent_green(level,n):
    c,r,i=fixture(level,green=True);result=evaluate(c,r,i)
    assert not result['success'] and result['k']==1 and result['n']==n


def test_medium_zero_only_one_of_six():
    c,r,i=fixture('medium',zero=True);result=evaluate(c,r,i)
    assert result['physical_zero'] and result['k']==1 and result['n']==6


def test_easy_loaded_without_balance_is_three_of_four_without_command_hover():
    c,r,i=fixture('easy',load=True);result=evaluate(c,r,i)
    assert result['k']==3 and not result['success']


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_complete_physical_success_without_expert_commands(level):
    c,r,i=fixture(level,load=True,zero=level!='easy',green=True,rider=level=='hard')
    result=evaluate(c,r,i)
    assert result['success'] and result['k']==result['n']


def test_hard_loaded_without_rider_and_with_rider_without_balance():
    c,r,i=fixture('hard',load=True,zero=True);assert evaluate(c,r,i)['k']==6
    c,r,i=fixture('hard',load=True,zero=True,rider=True);assert evaluate(c,r,i)['k']==7


def test_earlier_loading_can_be_corrected_by_empty_zero_then_loading():
    c,r,i=fixture('medium',load=True,zero=True)
    path=c['all_objects'][0]['path'];r[0]['raw']['objects'][path]['pan_supported']['Right']=True
    result=evaluate(c,r,i)
    assert not result['success'] and result['k']==5
    assert result['physical_zero'] and result['first_error'] is None


def test_world_pose_outlier_without_pan_loss_does_not_prove_a_drop():
    c,r,i=fixture('easy',load=True)
    r[400]['raw']['objects'][c['all_objects'][0]['path']]['position'][2]=.60
    result=evaluate(c,r,i)
    assert result['k']==3 and result['first_error'] is None and not result['success']


@pytest.mark.parametrize('level,threshold',[('easy',.030),('medium',.040),('hard',.045)])
def test_terminal_placement_does_not_require_the_old_lift_boundary(level,threshold):
    from mind2act_eval.tasks.CP06.physical import object_placed
    c,r,i=fixture(level,load=True,zero=level!='easy')
    path=next(o['path'] for o in c['all_objects'] if o['kind']=='fruit')
    lift=next(s['raw']['objects'][path] for s in r if any(s['raw']['objects'][path]['finger_contacts']))
    lift['position'][2]=i['objects'][path]['position'][2]+threshold-1e-5
    assert object_placed(path,'Left',c,[s['raw'] for s in r],i)
    lift['position'][2]=i['objects'][path]['position'][2]+threshold+1e-5
    assert object_placed(path,'Left',c,[s['raw'] for s in r],i)


@pytest.mark.parametrize('bad_ticks,success',[(12,True),(13,False)])
def test_medium_past_green_interruptions_do_not_veto_terminal_green(bad_ticks,success):
    c,r,i=fixture('medium',load=True,zero=True,green=True)
    for row in r[-360:-360+bad_ticks]:
        row['raw']['beam_deg']=row['raw']['corrected_angle_deg']=c['green_angle_interval_deg'][1]+1.
    result=evaluate(c,r,i)
    assert result['success'] and result['k']==6


@pytest.mark.parametrize('offset,success',[(-10,True),(-1,False)])
def test_hard_green_is_checked_at_t0(offset,success):
    c,r,i=fixture('hard',load=True,zero=True,green=True,rider=True)
    r[offset]['raw']['corrected_angle_deg']=r[offset]['raw']['beam_deg']=c['green_angle_interval_deg'][1]+1.
    result=evaluate(c,r,i)
    assert result['success'] is success and result['k']==(8 if success else 7)


def test_nonfinite_measured_corrected_angle_is_invalid_not_policy_zero():
    c,r,i=fixture('easy');r[0]['raw']['corrected_angle_deg']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):evaluate(c,r,i)


def test_final_support_does_not_require_pickup_release_or_green_history():
    c,r,i=fixture('hard',load=True,zero=True,green=True,rider=True)
    for row in r:
        for body in row['raw']['objects'].values():
            body['finger_contacts']=[True,True]
        row['raw']['corrected_angle_deg']=10.
    r[-1]['raw']['corrected_angle_deg']=0.
    result=evaluate(c,r,i)
    assert result['success'] and result['k']==8


def test_final_steps_are_independent_after_an_earlier_contact_error():
    c,r,i=fixture('medium',load=True,zero=True,green=False)
    r[0]['raw']['unexpected_robot_contacts']=[{'pair':['/robot','/pan']}]
    result=evaluate(c,r,i)
    assert result['k']==5 and not result['success']


def test_final_center_outside_pan_does_not_count_despite_contact():
    c,r,i=fixture('easy',load=True,green=True)
    fruit=next(o['path'] for o in c['all_objects'] if o['kind']=='fruit')
    r[-1]['raw']['objects'][fruit]['pan_center_local_m']['Left'][0]=c['pan_inner_radius_m']+.001
    result=evaluate(c,r,i)
    assert result['k']==3 and not result['success']


def test_final_green_is_independent_of_loading_units():
    c,r,i=fixture('easy',green=True)
    result=evaluate(c,r,i)
    assert result['k']==1 and result['steps']==[False,False,False,True]


def test_proven_fall_only_disqualifies_that_object_and_does_not_stop_other_units():
    c,r,i=fixture('easy',load=True,green=True)
    fruit=next(o['path'] for o in c['all_objects'] if o['kind']=='fruit')
    body=r[400]['raw']['objects'][fruit]
    body.update(pan_supported=dict(Left=False,Right=False),pan_center_local_m=dict(Left=[.2,0,-.02],Right=[.2,0,-.02]),
                linear_velocity_m_s=[0,0,-1],robot_contact=False,external_supported=False)
    r[400]['raw']['pan_bottom_local_m']=dict(Left=0.,Right=0.)
    result=evaluate(c,r,i)
    assert not result['success'] and result['k']==3
    assert result['dropped_objects']==[fruit]


def test_gripper_transport_below_pan_is_not_a_free_fall():
    c,r,i=fixture('easy',load=True,green=True)
    fruit=next(o['path'] for o in c['all_objects'] if o['kind']=='fruit')
    b=r[400]['raw']['objects'][fruit]
    b.update(pan_supported=dict(Left=False,Right=False),pan_center_local_m=dict(Left=[.2,0,-.02],Right=[.2,0,-.02]),
             linear_velocity_m_s=[0,0,-1],robot_contact=True,external_supported=False)
    result=evaluate(c,r,i)
    assert result['success'] and result['dropped_objects']==[]


def extra_weight_fall_fixture(level='easy'):
    config, rows, initial = fixture(level, load=True, zero=level!='easy',
                                    green=True, rider=level=='hard')
    # Full success has not occurred before the fall; green is first reached at T0.
    for row in rows[:-1]:
        row['raw']['corrected_angle_deg'] = 10.
    path = [o['path'] for o in config['all_objects'] if o['kind']=='weight'][2]
    rows[300]['raw']['objects'][path]['pan_supported']['Right'] = True
    rows[301]['raw']['objects'][path].update(
        pan_supported=dict(Left=False, Right=False),
        pan_center_local_m=dict(Left=[.2,0,-.02], Right=[.2,0,-.02]),
        linear_velocity_m_s=[0,0,-1], robot_contact=False, external_supported=False)
    return config, rows, initial, path


@pytest.mark.parametrize('level', ['easy','medium','hard'])
def test_extra_weight_fall_prevents_full_success_but_keeps_independent_units(level):
    config, rows, initial, path = extra_weight_fall_fixture(level)
    result = evaluate(config, rows, initial)
    assert result['dropped_objects'] == [path]
    assert len(result['placed_weights']) == 2
    assert result['k'] == result['n']
    assert not result['success']
