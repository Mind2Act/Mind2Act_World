from copy import deepcopy
import pytest
from mind2act_eval.tasks.CP08.instances import selected
from mind2act_eval.tasks.CP08.physical import evaluate


def trajectory(names=('B','A','C')):
    cfg=selected('easy')['source_case']
    initial={'blocks':{b['id']:dict(position=b['initial'],orientation_wxyz=[1.,0.,0.,0.]) for b in cfg['blocks']}}
    bodies={b['id']:dict(position=list(b['initial']),quaternion=[1.,0.,0.,0.],tilt_deg=0.,yaw_error_deg=0.,
        table_support=True,supports=[],robot_contact=False,bilateral_by_arm={'Left':False,'Right':False}) for b in cfg['blocks']}
    rows=[]
    def emit():rows.append(dict(tick=len(rows)+1,blocks=deepcopy(bodies),support_measurements_complete=True,contact_impulses={},tcp_by_arm={'Left':[0.,0.,1.], 'Right':[0.,0.,1.]},
        measured_jaws={side:[.01,.01] if any(b['bilateral_by_arm'][side] for b in bodies.values()) else [.044,.044] for side in ('Left','Right')}))
    for name in names:
        b=next(b for b in cfg['blocks'] if b['id']==name);body=bodies[name]
        body.update(position=[*b['initial'][:2],b['initial'][2]+.04],table_support=False,
            supports=[],robot_contact=True,bilateral_by_arm={'Left':True,'Right':False})
        emit()
        body.update(position=list(b['target']),table_support=not b['supports'],supports=b['supports'],
            robot_contact=False,bilateral_by_arm={'Left':False,'Right':False})
        for _ in range(120):emit()
    return cfg,initial,rows


def test_success_requires_all_ordered_steps_and_finish():
    cfg,initial,rows=trajectory()
    result=evaluate(rows,cfg,initial)
    assert result['success'] and result['completed_steps']==4


@pytest.mark.parametrize('names,k',[(('B',),1),(('B','A'),2)])
def test_partial_prefix(names,k):
    cfg,initial,rows=trajectory(names)
    assert evaluate(rows,cfg,initial)['completed_steps']==k


def test_forbidden_contact_records_no_permanent_failure():
    cfg,initial,rows=trajectory()
    rows[121]['contact_impulses']={('/World/RobotLeft/link3','/World/Worktable'):1.}
    assert evaluate(rows,cfg,initial)['success']


def test_displaced_block_can_be_restored():
    cfg,initial,rows=trajectory()
    rows[121]['blocks']['B']['position'][0]+=.02
    assert evaluate(rows,cfg,initial)['success']


@pytest.mark.parametrize('impulse', [0., .001])
def test_contact_without_displacement_does_not_freeze_completed_prefix(impulse):
    cfg,initial,rows=trajectory()
    # The native easy/medium trace first fails when a finger approaches B
    # after placing A. B remains at target with its table support intact.
    rows[121]['blocks']['B']['robot_contact']=True
    rows[121]['contact_impulses']={
        ('/World/CP11/Blocks/B/Normalized/Asset',
         '/World/RobotLeft/Asset/link8/collisions/link8/node_STL_BINARY_'):impulse}
    result=evaluate(rows,cfg,initial)
    assert result['first_error'] is None
    assert result['success'] and result['completed_steps']==4


def test_contact_does_not_cancel_actual_support():
    cfg,initial,rows=trajectory()
    rows[-1]['blocks']['B']['robot_contact']=True
    assert evaluate(rows,cfg,initial)['success']


def test_intermediate_support_loss_is_repairable():
    cfg,initial,rows=trajectory()
    rows[121]['blocks']['B']['table_support']=False
    assert evaluate(rows,cfg,initial)['success']


def test_terminal_displacement_removes_only_affected_units():
    cfg,initial,rows=trajectory()
    rows[-1]['blocks']['B']['position'][0]+=.02
    r=evaluate(rows,cfg,initial)
    assert not r['success'] and r['completed_steps']==2


def hard_trajectory(sequence):
    cfg=selected('hard')['source_case']
    specs={b['id']:b for b in cfg['blocks']+cfg['distractors']}
    center=specs['C']['target']
    initial={'blocks':{n:dict(position=b['initial'],orientation_wxyz=[1.,0.,0.,0.]) for n,b in specs.items()},
        'build_area_bounds_m':dict(lower=[center[0]-.0915,center[1]-.0515],upper=[center[0]+.0915,center[1]+.0515])}
    bodies={n:dict(position=list(b['initial']),quaternion=[1.,0.,0.,0.],tilt_deg=0.,yaw_error_deg=0.,
        table_support=n!='C',supports=['A','B'] if n=='C' else [],
        robot_contact=False,bilateral_by_arm={'Left':False,'Right':False}) for n,b in specs.items()}
    rows=[]
    def emit():rows.append(dict(tick=len(rows)+1,blocks=deepcopy(bodies),support_measurements_complete=True,contact_impulses={},tcp_by_arm={'Left':[0.,0.,1.], 'Right':[0.,0.,1.]},
        measured_jaws={side:[.01,.01] if any(b['bilateral_by_arm'][side] for b in bodies.values()) else [.044,.044] for side in ('Left','Right')}))
    for stage in sequence:
        name='C' if stage=='parkC' else stage;spec=specs[name];body=bodies[name]
        body.update(position=[*body['position'][:2],body['position'][2]+.04],table_support=False,
            supports=[],robot_contact=True,bilateral_by_arm={'Left':True,'Right':False})
        emit()
        parked=stage=='parkC'
        body.update(position=[.17,-.24,cfg['table_z']+spec['size'][2]/2] if parked else list(spec['target']),
            table_support=parked or not spec['supports'],supports=[] if parked else spec['supports'],
            robot_contact=False,bilateral_by_arm={'Left':False,'Right':False})
        for _ in range(600):emit()
    return cfg,initial,rows


def test_hard_park_repair_prefix_has_two_of_seven():
    cfg,initial,rows=hard_trajectory(('parkC','B'))
    result=evaluate(rows,cfg,initial)
    assert not result['success'] and result['completed_steps']==2 and result['total_steps']==7


def test_hard_complete_rework_succeeds():
    cfg,initial,rows=hard_trajectory(('parkC','B','C','D','E','F'))
    result=evaluate(rows,cfg,initial)
    assert result['success'] and result['completed_steps']==7


def test_hard_base_before_unload_loses_finish_but_retains_independent_units():
    cfg,initial,rows=hard_trajectory(('B','parkC','C','D','E','F'))
    r=evaluate(rows,cfg,initial)
    assert not r['success'] and r['completed_steps']==6


@pytest.mark.parametrize('names,k',[(('A','B','C'),0),(('B','C','A'),1)])
def test_ordinary_block_order_is_arbitrary(names, k):
    cfg,initial,rows=trajectory(names)
    assert evaluate(rows,cfg,initial)['success']


def test_upper_blocks_have_no_extra_order_after_base_repair():
    cfg,initial,rows=hard_trajectory(('parkC','B','D','C','E','F'))
    assert evaluate(rows,cfg,initial)['success']


def test_no_jaw_release_requirement():
    cfg,initial,rows=trajectory(('B',))
    for row in rows:row['measured_jaws']['Left']=[.01,.01]
    assert evaluate(rows,cfg,initial)['completed_steps']==1


@pytest.mark.parametrize('gap_ticks', [1, 2, 3])
def test_contact_history_fraction_is_not_a_score_gate(gap_ticks):
    cfg,initial,rows=trajectory()
    for row in rows[::gap_ticks]:row['blocks']['B']['bilateral_by_arm']['Left']=False
    assert evaluate(rows,cfg,initial)['success']


def test_jaws_not_needed_for_terminal_placement():
    cfg,initial,rows=trajectory(('B',))
    rows[0].pop('measured_jaws')
    assert evaluate(rows,cfg,initial)['completed_steps']==1


def dual_trajectory():
    cfg,initial,rows=trajectory(('B',))
    first=rows[0];body=first['blocks']['B'];position=body['position']
    body['bilateral_by_arm']={'Left':True,'Right':True}
    first['tcp_by_arm']={'Left':[position[0]+.03,*position[1:]],'Right':[position[0]+.01,*position[1:]]}
    first['measured_jaws']={'Left':[.01,.01],'Right':[.01,.01]}
    carrying=deepcopy(first)
    carrying['blocks']['B']['bilateral_by_arm']={'Left':False,'Right':True}
    carrying['measured_jaws']['Left']=[.044,.044]
    rows.insert(1,carrying)
    idle=deepcopy(first)
    idle['blocks']['B'].update(position=list(next(b for b in cfg['blocks'] if b['id']=='B')['initial']),
        table_support=True,supports=[],robot_contact=False,bilateral_by_arm={'Left':False,'Right':False})
    idle['measured_jaws']={'Left':[.044,.044],'Right':[.044,.044]}
    rows.insert(0,idle)
    for tick,row in enumerate(rows,1):row['tick']=tick
    return cfg,initial,rows


def test_dual_hand_choice_does_not_change_independent_placement():
    cfg,initial,rows=dual_trajectory()
    assert evaluate(rows,cfg,initial)['completed_steps']==1


def test_tcp_not_needed_for_terminal_placement():
    cfg,initial,rows=trajectory(('B',))
    rows[0].pop('tcp_by_arm')
    assert evaluate(rows,cfg,initial)['completed_steps']==1


@pytest.mark.parametrize('disturbance', ['table_contact', 'other_block_motion'])
def test_visible_cue_contacts_do_not_freeze_a_valid_physical_prefix(disturbance):
    cfg,initial,rows=trajectory()
    for row in rows:row['cue_visible']=row['tick']<=288
    if disturbance == 'table_contact':
        rows[121]['contact_impulses']={('/World/RobotLeft/Asset/link3/collision','/World/Worktable'):1.}
    else:
        rows[0]['blocks']['A']['robot_contact'] = True
        rows[0]['blocks']['A']['position'][0] += .02
    result=evaluate(rows,cfg,initial)
    assert result['first_error'] is None
    assert result['success'] and result['completed_steps']==4


def test_final_independent_units_allow_recovery_contact_and_arbitrary_order():
    cfg, initial, rows = trajectory(('C', 'A', 'B'))
    for body in rows[-1]['blocks'].values():
        body['robot_contact'] = True
        body['tilt_deg'] = 12.
        body['yaw_error_deg'] = 30.
        body['supports'] += ['harmless']
    rows[0]['contact_impulses'] = {('/World/RobotLeft/link3', '/World/Worktable'): 1.}
    result = evaluate(rows, cfg, initial)
    assert result['success'] and result['completed_steps'] == 4


def test_nonreference_support_cannot_complete_hard_structure():
    cfg,initial,rows=hard_trajectory(('parkC','B','C','D','E','F'))
    rows[-1]['blocks']['D']['supports'].append(cfg['distractors'][0]['id'])
    result=evaluate(rows,cfg,initial)
    assert not result['success'] and result['completed_steps']==6


def test_extra_block_inside_actual_marked_area_loses_only_finish():
    cfg,initial,rows=hard_trajectory(('parkC','B','C','D','E','F'))
    initial['build_area_bounds_m']=dict(lower=[-.1,-.2],upper=[.1,-.05])
    rows[-1]['blocks']['G']['position']=[0,-.13,.785]
    r=evaluate(rows,cfg,initial)
    assert not r['success'] and r['completed_steps']==6


@pytest.mark.parametrize('initially_inside', [True, False])
def test_scene_refreshes_distractor_placement_while_reference_blocks_stay_correct(initially_inside):
    from types import SimpleNamespace
    from mind2act_eval.tasks.CP08.scene import StructureScene
    from mind2act_eval.tasks.CP08.physical import StabilityWindow
    config, initial, rows = hard_trajectory(('parkC','B','C','D','E','F'))
    name = config['distractors'][0]['id']
    outside = deepcopy(rows[-1]['blocks'][name]['position'])
    area = initial['build_area_bounds_m']
    inside = [(a+b)/2 for a,b in zip(area['lower'],area['upper'])]
    def position(occupied):
        if occupied:
            rows[-1]['blocks'][name]['position'][:2] = inside
        else:
            rows[-1]['blocks'][name]['position'] = deepcopy(outside)
    position(initially_inside)
    scene = object.__new__(StructureScene)
    scene.replay = SimpleNamespace(rows=rows,config=config)
    scene.initial = initial
    scene.windows = {b['id']:StabilityWindow(b,config) for b in config['blocks']}
    for name_,window in scene.windows.items():window.append(rows[-1]['blocks'][name_])
    scene._candidate = scene._complete = False
    assert scene.is_success() is (not initially_inside)
    rows.append(deepcopy(rows[-1]));rows[-1]['tick'] += 1
    position(not initially_inside)
    for name_,window in scene.windows.items():window.append(rows[-1]['blocks'][name_])
    assert all(window.stable for window in scene.windows.values())
    assert scene.is_success() is initially_inside
    assert scene.is_success() == evaluate(rows,config,initial)['success']
