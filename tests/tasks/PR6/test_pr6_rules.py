"""Physical positive, partial, negative and boundary traces; no expert labels."""
import copy
import json
import math
from pathlib import Path
import pytest
from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.PR6.evidence import Recorder, rule_frame
from mind2act_eval.tasks.PR6.instances import digest, validate_case
from mind2act_eval.tasks.PR6.scorer import score
from mind2act_eval.tasks.PR6.mechanism.contact_policy import contact_actions


def case(difficulty='easy'):
    count = {'easy':5,'medium':7,'hard':9}[difficulty]
    objects = {'domino'+str(i+1):dict(position=[i*.03,0.,.81],size=[.01,.02,.08]) for i in range(count)}
    objects.update({n:dict(position=[.35,0.,.81],size=[.04,.04,.08]) for n in ('bell','clapper')})
    c = dict(task='CP11',public_task_id='PR8',version='pr8-four-case-native-v4',difficulty=difficulty,
        seed=0,physics_hz=120,control_hz=30,push_budget=8,stall_ticks=360,stable_ticks=60,cue_ticks=600,
        max_linear_speed=.015,max_angular_speed=.15,support_tolerance=.004,scene_profile='cp-aligned',
        native_four_case={'schema':'four-case-native-v1'},
        prop_solver=dict(position_iterations=128,velocity_iterations=8),objects=objects,
        bell=dict(mechanism='hollow_shell_passive_clapper_v1',ring_angle_deg=5.),
        route_region=dict(centerline=[[-.05,0.],[.5,0.]],half_width_m=.043))
    c['case_sha256'] = digest(c)
    return c


def contact(a,b,normal=(0.,0.,1.)):
    own = '/World/Task/Objects/'+a+'/Collider' if a not in ('table','robot') else {'table':'/World/Table','robot':'/World/RobotLeft/Asset/finger1'}[a]
    other = '/World/Task/Objects/'+b+'/Collider' if b not in ('table','robot') else {'table':'/World/Table','robot':'/World/RobotLeft/Asset/finger1'}[b]
    pair = sorted([own,other]); sign = 1 if pair[0] == own else -1
    return dict(pair=pair,active=True,points=[dict(position=[0.,0.,.77],normal=[sign*v for v in normal],separation=0.)])


def frame(c,tick,*,fallen=(),contacts=(),responding=(),bell_angle=0.,positions=None):
    bodies = {}
    for name,item in c['objects'].items():
        angle = math.pi/2 if name in fallen else math.radians(bell_angle) if name == 'bell' else 0.
        bodies[name] = dict(position=(positions or {}).get(name,item['position']),
            orientation_wxyz=[math.cos(angle/2),0.,math.sin(angle/2),0.],linear=[0.,0.,0.],
            angular=[0.,.2 if name in responding else 0.,0.])
    return dict(tick=tick,time_s=tick/120,bodies=bodies,
        contacts=[contact(n,'table') for n in bodies]+list(contacts))


def setup(rec):
    for tick in range(660):
        rec.record(frame(rec.config,tick))
    assert rec.rules.setup_tick == 600


def topple(rec,count,*,bell=False,clapper=False,angle=0.):
    c = rec.config; names = rec.rules.chain
    for i in range(count):
        edges = [contact('robot',names[0])] if i == 0 else [contact(names[i-1],names[i])]
        if i == len(names)-1 and bell:
            edges.append(contact(names[i],'bell'))
        if clapper:
            edges.append(contact('bell','clapper'))
        rec.record(frame(c,rec.frames,fallen=names[:i+1],contacts=edges,responding=[names[i]],bell_angle=angle))


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_strict_native_chain_rings_bell(tmp_path,difficulty):
    r = Recorder('fixture',case(difficulty),trace_path=tmp_path/'trace')
    setup(r); topple(r,len(r.rules.chain),bell=True,clapper=True,angle=5.01)
    result = score(r.evidence(r.frames-1))
    assert result.value == 100 and result.components['task_success']
    assert r.rules.pushes == 1
    # Further raw native frames are audited, without raising after natural terminal.
    r.record(frame(r.config,r.frames)); assert score(r.evidence(r.frames-1)).value == 100
    r.close()


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_historical_setup_and_half_prefix(tmp_path,difficulty):
    r = Recorder('fixture',case(difficulty),trace_path=tmp_path/'trace')
    setup(r); assert score(r.evidence(r.frames-1)).value == {'easy':15,'medium':11,'hard':7.5}[difficulty]
    half = (len(r.rules.chain)+1)//2
    topple(r,half-1); assert score(r.evidence(r.frames-1)).value == {'easy':15,'medium':11,'hard':7.5}[difficulty]
    i = half-1
    r.record(frame(r.config,r.frames,fallen=r.rules.chain[:half],responding=[r.rules.chain[i]],
                   contacts=[contact(r.rules.chain[i-1],r.rules.chain[i])]))
    assert score(r.evidence(r.frames-1)).value == {'easy':30,'medium':22,'hard':15}[difficulty]
    assert r.rules.setup_tick == 600
    r.close()


@pytest.mark.parametrize('bell,clapper,angle',[(False,False,6),(True,False,6),(True,True,4.99)])
def test_no_ring_is_valid_partial_not_invalid(tmp_path,bell,clapper,angle):
    r = Recorder('fixture',case(),trace_path=tmp_path/'trace')
    setup(r); topple(r,5,bell=bell,clapper=clapper,angle=angle)
    assert score(r.evidence(r.frames-1)).value == (100 if bell else 30)
    assert r.rules.status == ('success' if bell else 'running')
    r.close()


def test_zero_actions_has_numeric_zero(tmp_path):
    c = case(); c['objects']['domino1']['position'][1] = -.2; c['case_sha256'] = digest({k:v for k,v in c.items() if k!='case_sha256'})
    r = Recorder('fixture',c,trace_path=tmp_path/'trace'); r.record(frame(c,0))
    assert score(r.evidence(0)).value == 0
    r.close()


def test_unactivated_unrelated_falls_do_not_count(tmp_path):
    r = Recorder('fixture',case(),trace_path=tmp_path/'trace'); setup(r)
    r.record(frame(r.config,r.frames,fallen=r.rules.chain))
    assert score(r.evidence(r.frames-1)).value == 15
    r.close()


@pytest.mark.parametrize('name',['bell','clapper'])
def test_illegal_contact_zero_overrides_partial(tmp_path,name):
    r = Recorder('fixture',case(),trace_path=tmp_path/'trace'); setup(r); topple(r,3)
    r.record(frame(r.config,r.frames,contacts=[contact('robot',name)]))
    assert score(r.evidence(r.frames-1)).value == 0
    r.close()


def test_repeated_static_touch_does_not_consume_the_push_budget(tmp_path):
    r = Recorder('fixture',case(),trace_path=tmp_path/'trace'); setup(r)
    for i in range(9):
        r.record(frame(r.config,r.frames,contacts=[contact('robot','domino1')]))
        if i == 7: assert score(r.evidence(r.frames-1)).value == 15
        r.record(frame(r.config,r.frames))
    assert score(r.evidence(r.frames-1)).value == 15
    assert r.rules.pushes == 0 and not r.rules.activated
    topple(r, 5, bell=True)
    assert score(r.evidence(r.frames-1)).value == 100 and r.rules.pushes == 1
    r.close()


def test_ninth_actual_push_scores_zero_without_terminating_physics(tmp_path):
    from mind2act_eval.tasks.PR6.scene import DominoScene
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    for count in range(1, 10):
        if count > 1:
            while r.frames < r.rules.last_progress + r.config['stall_ticks']:
                r.record(frame(r.config, r.frames))
        r.record(frame(r.config, r.frames, responding=['domino1'],
                       contacts=[contact('robot', 'domino1')]))
        if count <= 8:
            assert r.rules.pushes == count
            assert r.rules.status == 'running'
            assert score(r.evidence(r.frames-1)).value == 15
            r.record(frame(r.config, r.frames))
    assert r.rules.pushes == 9 and r.rules.status == 'failure'
    assert r.rules.reason == 'push_budget_exceeded'
    scene = object.__new__(DominoScene)
    scene.recorder = r
    scene.measure = lambda tick: frame(r.config, tick)
    assert not scene.is_success() and not scene.is_complete()
    next_tick = r.frames
    scene.after_step(next_tick)
    assert r.frames == next_tick + 1
    assert score(r.evidence(next_tick)).value == 0
    r.close()


def test_held_static_contact_becomes_one_push_only_when_native_motion_starts(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    touch = [contact('robot', 'domino1')]
    r.record(frame(r.config, r.frames, contacts=touch))
    assert r.rules.pushes == 0 and not r.rules.activated
    r.record(frame(r.config, r.frames, contacts=touch, responding=['domino1']))
    assert r.rules.pushes == 1 and r.rules.activated == {'domino1'}
    r.record(frame(r.config, r.frames, contacts=touch, responding=['domino1']))
    assert r.rules.pushes == 1 and r.rules.status == 'running'
    r.close()


def test_eighth_actual_push_can_still_start_a_complete_contact_chain(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    names = r.rules.chain
    for count in range(1, 9):
        if count > 1:
            while r.frames < r.rules.last_progress + r.config['stall_ticks']:
                r.record(frame(r.config, r.frames))
        r.record(frame(r.config, r.frames, responding=['domino1'],
                       contacts=[contact('robot', 'domino1')]))
        if count < 8:r.record(frame(r.config, r.frames))
    r.record(frame(r.config, r.frames, fallen=names[:1]))
    for i in range(1, len(names)):
        contacts = [contact(names[i-1], names[i])]
        if i == len(names)-1:contacts.append(contact(names[i], 'bell'))
        r.record(frame(r.config, r.frames, fallen=names[:i+1],
                       responding=[names[i]], contacts=contacts))
    assert r.rules.pushes == 8
    assert score(r.evidence(r.frames-1)).value == 100
    r.close()


def test_first_push_and_restarts_after_three_seconds_share_one_eight_push_budget(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    topple(r, 1)
    assert r.rules.pushes == 1 and r.rules.fallen == {'domino1'}
    for count in range(2, 10):
        # Only the first domino may be pushed: stand it back up and restart it.
        while r.frames < r.rules.last_progress + r.config['stall_ticks']:
            r.record(frame(r.config, r.frames))
        r.record(frame(r.config, r.frames, responding=['domino1'],
                       contacts=[contact('robot', 'domino1')]))
        if count < 9:
            assert r.rules.pushes == count and r.rules.status == 'running'
            r.record(frame(r.config, r.frames))
    assert r.rules.pushes == 9 and r.rules.status == 'failure'
    assert sum(e['kind'] == 'physical_chain_push' and e['restart'] for e in r.rules.events) == 8
    assert score(r.evidence(r.frames-1)).value == 0
    r.close()


@pytest.mark.parametrize('idle_ticks,expected', [(359, 0), (360, 0)])
def test_actual_restart_uses_the_published_three_second_boundary(tmp_path, idle_ticks, expected):
    from mind2act_eval.tasks.PR6.scene import DominoScene
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    topple(r, 1)
    while r.frames < r.rules.last_progress + idle_ticks:
        r.record(frame(r.config, r.frames, fallen=['domino1']))
    r.record(frame(r.config, r.frames, fallen=['domino1'], responding=['domino2'],
                   contacts=[contact('robot', 'domino2')]))
    assert score(r.evidence(r.frames-1)).value == expected
    # 2026-11-06: pushing any domino other than the first is rejected outright.
    assert r.rules.reason == 'push_not_first_domino'
    if idle_ticks < 360:
        scene = object.__new__(DominoScene)
        scene.recorder = r
        scene.measure = lambda tick: frame(r.config, tick, fallen=['domino1'])
        assert not scene.is_success() and not scene.is_complete()
        next_tick = r.frames
        scene.after_step(next_tick)
        assert r.frames == next_tick + 1
    r.close()


def test_static_restart_touch_can_wait_for_three_seconds_without_a_penalty(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    topple(r, 1)
    touch = [contact('robot', 'domino2')]
    while r.frames < r.rules.last_progress + 360:
        r.record(frame(r.config, r.frames, fallen=['domino1'], contacts=touch))
        assert r.rules.status == 'running' and r.rules.pushes == 1
    r.record(frame(r.config, r.frames, fallen=['domino1'], contacts=touch,
                   responding=['domino2']))
    assert r.rules.status == 'failure' and r.rules.reason == 'push_not_first_domino'
    r.close()


@pytest.mark.parametrize('idle_ticks,expected', [(359, 0), (360, 15)])
def test_first_domino_reactivation_is_also_a_restart_with_a_three_second_wait(tmp_path, idle_ticks, expected):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    topple(r, 1)
    assert r.rules.fallen == {'domino1'}
    while r.frames < r.rules.last_progress + idle_ticks:
        # The first tile has been stood up again and is now stationary.
        r.record(frame(r.config, r.frames))
    r.record(frame(r.config, r.frames, responding=['domino1'],
                   contacts=[contact('robot', 'domino1')]))
    assert score(r.evidence(r.frames-1)).value == expected
    assert r.rules.pushes == 2
    if idle_ticks == 359:
        assert r.rules.reason == 'restart_before_stall_wait'
    else:
        pushes = [e for e in r.rules.events if e['kind'] == 'physical_chain_push']
        assert len(pushes) == 2 and pushes[-1]['item'] == 'domino1'
        assert pushes[-1]['restart'] is True
    r.close()


def test_first_domino_static_contact_can_wait_until_legal_restart_without_consuming_pushes(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    topple(r, 1)
    r.record(frame(r.config, r.frames))  # repaired upright, no physical push
    touch = [contact('robot', 'domino1')]
    while r.frames < r.rules.last_progress + 360:
        r.record(frame(r.config, r.frames, contacts=touch))
        assert r.rules.status == 'running' and r.rules.pushes == 1
    r.record(frame(r.config, r.frames, contacts=touch, responding=['domino1']))
    assert r.rules.status == 'running' and r.rules.pushes == 2
    assert r.rules.events[-1]['kind'] == 'physical_chain_push'
    assert r.rules.events[-1]['restart'] is True
    r.close()


@pytest.mark.parametrize('fallen_count,expected', [(0, 15), (3, 30), (4, 30)])
def test_bell_contact_waits_for_the_whole_legally_activated_chain_to_fall(tmp_path, fallen_count, expected):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    names = r.rules.chain
    for i, name in enumerate(names):
        contacts = [contact('robot', name)] if i == 0 else [contact(names[i-1], name)]
        if i == len(names)-1:contacts.append(contact(name, 'bell'))
        r.record(frame(r.config, r.frames, responding=[name], contacts=contacts,
                       fallen=names[:min(i+1, fallen_count)]))
    assert r.rules.bell_contact and r.rules.activated == set(names)
    assert score(r.evidence(r.frames-1)).value == expected
    r.record(frame(r.config, r.frames, fallen=names))
    assert score(r.evidence(r.frames-1)).value == 100
    r.close()


def test_contact_policy_classifies_a_static_restart_candidate_without_penalizing_touch():
    chain = ['d1','d2','d3']; fallen={'d1'}
    assert contact_actions(chain,{'d2'},fallen,1,0,360)['d2'] == 'restart'
    assert contact_actions(chain,{'d3'},fallen,1,0,360)['d3'] == 'restart'



def test_moving_tile_reentry_illegal_existing_contact_continues(tmp_path):
    r = Recorder('fixture',case(),trace_path=tmp_path/'trace'); setup(r)
    r.record(frame(r.config,r.frames,contacts=[contact('robot','domino1')],responding=['domino1']))
    r.record(frame(r.config,r.frames,contacts=[contact('robot','domino1')],responding=['domino1']))
    assert r.rules.status == 'running' and r.rules.pushes == 1
    r.record(frame(r.config,r.frames,responding=['domino1']))
    r.record(frame(r.config,r.frames,contacts=[contact('robot','domino1')],responding=['domino1']))
    assert score(r.evidence(r.frames-1)).value == 0
    r.close()


def test_notice_contact_does_not_fail_and_no_setup_hold_is_required(tmp_path):
    r = Recorder('fixture',case(),trace_path=tmp_path/'trace')
    for t in range(601): r.record(frame(r.config,t))
    assert r.rules.setup_tick == 600
    r.close()
    r = Recorder('fixture',case(),trace_path=tmp_path/'notice')
    r.record(frame(r.config,0,contacts=[contact('robot','domino1')]))
    assert score(r.evidence(0)).value == 0 and r.rules.reason is None
    r.close()


def test_full_body_footprint_support_release_and_orientation():
    from mind2act_eval.tasks.PR6.mechanism.dominos import Dominos
    c = case()
    for variant in ('outside','unsupported','tilt'):
        rules = Dominos(c)
        for t in range(660):
            raw = frame(c,t)
            if variant == 'outside': raw['bodies']['domino1']['position'][1] = .04
            if variant == 'unsupported': raw['contacts'] = []
            if variant == 'touch': raw['contacts'].append(contact('robot','domino1'))
            if variant == 'tilt': raw['bodies']['domino1']['orientation_wxyz'] = [math.cos(.2),0.,math.sin(.2),0.]
            rules.sample(rule_frame(raw,c,t))
        assert rules.setup_tick is None


@pytest.mark.parametrize('mutation',['gap','nan','missing','phase','quaternion','clock','contact'])
def test_missing_or_forged_native_measurements_rejected(tmp_path,mutation):
    c = case(); r = Recorder('fixture',c,trace_path=tmp_path/'trace'); r.record(frame(c,0)); f=frame(c,1)
    if mutation == 'gap': f['tick']=2
    if mutation == 'nan': f['bodies']['bell']['linear'][0]=float('nan')
    if mutation == 'missing': del f['bodies']['domino1']
    if mutation == 'phase': f['expert_phase']='success'
    if mutation == 'quaternion': f['bodies']['bell']['orientation_wxyz']=[2,0,0,0]
    if mutation == 'clock': f['time_s']=.1
    if mutation == 'contact': f['contacts'][0]['points'][0]['normal']=[0,0,0]
    with pytest.raises(ValueError): r.record(f)
    r.close()


def test_audit_tamper_and_bounded_evidence(tmp_path):
    r=Recorder('fixture',case(),trace_path=tmp_path/'trace')
    sizes=[]
    for t in range(2000):
        r.record(frame(r.config,t))
        if t in (999,1999): sizes.append(len(json.dumps(to_json(r.evidence(t)))))
    assert abs(sizes[0]-sizes[1]) < 10
    e=r.evidence(1999); assert score(e).value==15
    r.close(); p=Path(e.values['raw_trace']['path']); data=p.read_bytes(); p.write_bytes(data.replace(b'0.008333333333333333',b'0.108333333333333333',1))
    with pytest.raises(ValueError): score(e)


@pytest.mark.parametrize('name',['domino2','domino1'])
def test_unactivated_falling_contact_zero_and_static_repair_allowed(tmp_path,name):
    r=Recorder('fixture',case(),trace_path=tmp_path/'moving');setup(r)
    # Falling target was physically moving before this new contact; no activation claim.
    r.record(frame(r.config,r.frames,fallen=[name],responding=[name]))
    r.record(frame(r.config,r.frames,fallen=[name],responding=[name],contacts=[contact('robot',name)]))
    assert score(r.evidence(r.frames-1)).value==0
    r.close()
    r=Recorder('fixture',case(),trace_path=tmp_path/'static');setup(r)
    r.record(frame(r.config,r.frames,fallen=['domino2'],contacts=[contact('robot','domino2')]))
    assert score(r.evidence(r.frames-1)).value==15 and r.rules.pushes==0
    r.close()


def test_unactivated_falling_repair_same_tick_is_zero(tmp_path):
    r=Recorder('fixture',case(),trace_path=tmp_path/'trace');setup(r)
    r.record(frame(r.config,r.frames,fallen=['domino2'],responding=['domino2'],contacts=[contact('robot','domino2')]))
    assert score(r.evidence(r.frames-1)).value==0 and not r.rules.activated
    r.close()


def test_repair_can_start_moving_a_previously_resting_fallen_tile(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    # The tile has already stopped falling before the hand begins repairing it.
    r.record(frame(r.config, r.frames, fallen=['domino2']))
    r.record(frame(r.config, r.frames, fallen=['domino2'], responding=['domino2'],
                   contacts=[contact('robot', 'domino2')]))
    assert r.rules.status == 'running' and r.rules.pushes == 0
    r.record(frame(r.config, r.frames, contacts=[contact('robot', 'domino2')]))
    r.record(frame(r.config, r.frames))
    topple(r, 5, bell=True)
    assert r.rules.pushes == 1
    assert score(r.evidence(r.frames-1)).value == 100
    r.close()


def test_notice_motion_is_allowed_and_setup_needs_no_released_dwell(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    for tick in range(600):
        r.record(frame(r.config, tick, contacts=[contact('robot', 'domino1')]))
    assert r.rules.status == 'running'
    r.record(frame(r.config, 600, contacts=[contact('robot', 'domino1')]))
    assert r.rules.setup_tick == 600
    assert score(r.evidence(600)).value == 15
    r.close()


def test_last_domino_contact_rings_without_clapper_or_bell_angle(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    topple(r, 5, bell=True, clapper=False, angle=0.)
    assert score(r.evidence(r.frames-1)).value == 100
    r.record(frame(r.config, r.frames, contacts=[contact('robot', 'bell')]))
    assert score(r.evidence(r.frames-1)).value == 0
    r.close()


def test_later_actual_push_before_predecessors_fall_is_counted_and_zero(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    r.record(frame(r.config, r.frames, responding=['domino1'],
                   contacts=[contact('robot', 'domino1')]))
    r.record(frame(r.config, r.frames, responding=['domino1', 'domino2'],
                   contacts=[contact('robot', 'domino2')]))
    assert r.rules.pushes == 2
    assert score(r.evidence(r.frames-1)).value == 0
    assert r.rules.reason == 'push_not_first_domino'
    r.close()


def test_held_authorized_push_becomes_falling_touch_and_physics_continues(tmp_path):
    from mind2act_eval.tasks.PR6.scene import DominoScene
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    touch = [contact('robot', 'domino1')]
    r.record(frame(r.config, r.frames, responding=['domino1'], contacts=touch))
    assert r.rules.pushes == 1 and r.rules.status == 'running'
    r.record(frame(r.config, r.frames, fallen=['domino1'],
                   responding=['domino1'], contacts=touch))
    assert score(r.evidence(r.frames-1)).value == 0
    assert r.rules.reason == 'domino_contact_before_repair_window'
    scene = object.__new__(DominoScene)
    scene.recorder = r
    scene.measure = lambda tick: frame(r.config, tick, fallen=['domino1'])
    assert not scene.is_success() and not scene.is_complete()
    next_tick = r.frames
    scene.after_step(next_tick)
    assert r.frames == next_tick+1 and score(r.evidence(next_tick)).value == 0
    r.close()


@pytest.mark.parametrize('quiet_contact_first', [False, True])
def test_illegal_later_push_cannot_be_recredited_by_a_later_contact_chain(tmp_path, quiet_contact_first):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    names = r.rules.chain
    r.record(frame(r.config, r.frames, responding=['domino1'],
                   contacts=[contact('robot', 'domino1')]))
    if quiet_contact_first:
        r.record(frame(r.config, r.frames, responding=['domino1'],
                       contacts=[contact('robot', 'domino2')]))
        assert r.rules.pushes == 1 and r.rules.status == 'running'
    r.record(frame(r.config, r.frames, responding=names[:2],
                   contacts=[contact('robot', 'domino2')]))
    r.record(frame(r.config, r.frames, fallen=['domino2'], responding=names[:2],
                   contacts=[contact('robot', 'domino2')]))
    r.record(frame(r.config, r.frames, fallen=names[:2], responding=names[:2],
                   contacts=[contact('domino1', 'domino2')]))
    for i in range(2, len(names)):
        contacts = [contact(names[i-1], names[i])]
        if i == len(names)-1:
            contacts.append(contact(names[i], 'bell'))
        r.record(frame(r.config, r.frames, fallen=names[:i+1], responding=[names[i]], contacts=contacts))
    assert r.rules.pushes == 2 and r.rules.status == 'failure'
    assert score(r.evidence(r.frames-1)).value == 0
    assert r.rules.reason == 'push_not_first_domino'
    r.close()


def test_repair_that_reached_upright_cannot_exempt_a_later_held_fall(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    names = r.rules.chain
    touch = [contact('robot', 'domino2')]
    r.record(frame(r.config, r.frames, fallen=['domino2']))
    r.record(frame(r.config, r.frames, fallen=['domino2'], responding=['domino2'], contacts=touch))
    # The repair reaches upright while moving, without any quiet upright frame.
    r.record(frame(r.config, r.frames, responding=['domino2'], contacts=touch))
    r.record(frame(r.config, r.frames, responding=names[:2],
                   contacts=touch+[contact('robot', 'domino1')]))
    r.record(frame(r.config, r.frames, fallen=['domino2'], responding=names[:2], contacts=touch))
    r.record(frame(r.config, r.frames, fallen=names[:2], responding=names[:2],
                   contacts=touch+[contact('domino1', 'domino2')]))
    for i in range(2, len(names)):
        contacts = touch+[contact(names[i-1], names[i])]
        if i == len(names)-1:
            contacts.append(contact(names[i], 'bell'))
        r.record(frame(r.config, r.frames, fallen=names[:i+1], responding=names[:i+1], contacts=contacts))
    assert score(r.evidence(r.frames-1)).value == 0
    assert r.rules.reason == 'domino_contact_before_repair_window'
    assert r.rules.pushes == 1
    r.close()


def test_moving_upright_repair_can_settle_then_legally_restart_after_wait(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    names = r.rules.chain
    topple(r, 1)
    touch = [contact('robot', 'domino2')]
    r.record(frame(r.config, r.frames, fallen=names[:2]))
    r.record(frame(r.config, r.frames, fallen=names[:2], responding=['domino2'], contacts=touch))
    r.record(frame(r.config, r.frames, fallen=names[:1], responding=['domino2'], contacts=touch))
    assert r.rules.status == 'running' and r.rules.pushes == 1
    r.record(frame(r.config, r.frames, fallen=names[:1], contacts=touch))
    while r.frames < r.rules.last_progress+r.config['stall_ticks']:
        r.record(frame(r.config, r.frames, fallen=names[:1], contacts=touch))
    assert r.rules.status == 'running' and r.rules.pushes == 1
    r.record(frame(r.config, r.frames, fallen=names[:1], responding=['domino2'], contacts=touch))
    assert r.rules.status == 'failure' and r.rules.reason == 'push_not_first_domino'
    assert score(r.evidence(r.frames-1)).value == 0
    r.close()


def test_repair_upright_history_ends_when_the_contact_ends(tmp_path):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    touch = [contact('robot', 'domino2')]
    r.record(frame(r.config, r.frames, fallen=['domino2']))
    r.record(frame(r.config, r.frames, fallen=['domino2'], responding=['domino2'], contacts=touch))
    r.record(frame(r.config, r.frames, responding=['domino2'], contacts=touch))
    r.record(frame(r.config, r.frames, fallen=['domino2']))
    # A fresh contact with a quiet fallen tile is again a legitimate repair.
    r.record(frame(r.config, r.frames, fallen=['domino2'], responding=['domino2'], contacts=touch))
    assert r.rules.status == 'running' and r.rules.pushes == 0
    r.record(frame(r.config, r.frames, contacts=touch))
    r.record(frame(r.config, r.frames))
    topple(r, 5, bell=True)
    assert score(r.evidence(r.frames-1)).value == 100
    r.close()


@pytest.mark.parametrize('fall_again', [False, True])
def test_quiet_activated_fallen_tile_can_be_lifted_upright_on_repair_entry(tmp_path, fall_again):
    r = Recorder('fixture', case(), trace_path=tmp_path/'trace')
    setup(r)
    names = r.rules.chain
    topple(r, 2)
    r.record(frame(r.config, r.frames, fallen=names[:2]))
    touch = [contact('robot', 'domino2')]
    r.record(frame(r.config, r.frames, fallen=names[:1], responding=['domino2'], contacts=touch))
    assert r.rules.status == 'running' and r.rules.pushes == 1
    if fall_again:
        r.record(frame(r.config, r.frames, fallen=names[:2], responding=['domino2'], contacts=touch))
        assert score(r.evidence(r.frames-1)).value == 0
        assert r.rules.reason == 'domino_contact_before_repair_window'
    else:
        r.record(frame(r.config, r.frames, fallen=names[:1], contacts=touch))
        r.record(frame(r.config, r.frames, fallen=names[:2]))
        for i in range(2, len(names)):
            contacts = [contact(names[i-1], names[i])]
            if i == len(names)-1:
                contacts.append(contact(names[i], 'bell'))
            r.record(frame(r.config, r.frames, fallen=names[:i+1], responding=[names[i]], contacts=contacts))
        assert score(r.evidence(r.frames-1)).value == 100
    assert r.rules.pushes == 1
    r.close()
