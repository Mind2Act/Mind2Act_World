from copy import deepcopy
import json
import pytest

from mind2act_eval.tasks.DR4.evidence import Recorder, reconstruct, HANDLE, GOAL, TRAY
from mind2act_eval.tasks.DR4.scorer import score


def contact(a, b):
    row = dict(pair=sorted((a, b)), impulse_ns=[0., 0., .01])
    if GOAL in a or GOAL in b:row['support_normal_z'] = 1.
    return row


def rec_radius(n):
    return {2: .030, 3: .025, 4: .020}[n]


def frame(tick, n, *, retreat=.1, gripping=False, support=True):
    z = .704 + min(tick, 40)*.002
    velocity = [0., 0., .24 if 0 < tick < 40 else 0.]
    contacts = []
    if gripping:
        contacts += [contact(HANDLE, '/World/RobotLeft/Asset/'+link+'/collision') for link in ('link7', 'link8')]
    if support:
        contacts.append(contact(TRAY+'/Shell/Collider', GOAL+'/Top'))
    return dict(tick=tick, time_s=tick/120, tray_position=[0., .15, z], tray_rotation_wxyz=[1., 0., 0., 0.],
        tray_linear_velocity=velocity, tray_angular_velocity=[0., 0., 0.],
        balls={'Ball'+str(i): dict(position=[-.04+.03*i, .15, z+.02], linear_velocity=velocity) for i in range(n)},
        tcps={'Left': [retreat, .15, z+.067], 'Right': [.3, .15, z+.067]}, contacts=contacts)


def episode(tmp_path, n=3, *, hold=120, retreat=.1, keep_grip=False, changes=None):
    level = {2: 'easy', 3: 'medium', 4: 'hard'}[n]
    config = dict(level=level, ball_count=n, radius_mm={'easy':30., 'medium':25., 'hard':20.}[level])
    geometry = dict(goal=[0., .15, .784], polygon=[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]], floor_z=0., rim_z=.03)
    rec = Recorder('DR4', 'synthetic', config, geometry, frame(0, n, support=False), trace_path=tmp_path/'raw.jsonl')
    for tick in range(1, 41+hold):
        row = frame(tick, n, retreat=retreat, gripping=tick <= 40 or keep_grip, support=tick > 40)
        if changes:
            changes(row, tick)
        rec.record(row)
    return rec


@pytest.mark.parametrize('n', [2, 3, 4])
def test_all_balls_retained_uses_the_frozen_instance_count(tmp_path, n):
    rec = episode(tmp_path, n)
    result = score(rec.evidence(rec.frames))
    facts = result.components['physical_facts']
    assert result.value == 100. and facts['all_balls_retained']
    assert facts['instance_ball_count'] == n
    assert 'both_balls_retained' not in facts


@pytest.mark.parametrize('normal', [0., -1.])
def test_delivery_needs_upward_support_not_impulse_magnitude(tmp_path, normal):
    def change(row, tick):
        if tick > 40:row['contacts'][-1]['support_normal_z'] = normal
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 0.


def test_missing_platform_support_normal_is_infra(tmp_path):
    def change(row, tick):
        if tick > 40:row['contacts'][-1].pop('support_normal_z')
    with pytest.raises(ValueError, match='support normal'):
        rec = episode(tmp_path, changes=change)
        score(rec.evidence(rec.frames))


@pytest.mark.parametrize('distance,expected', [(.0601,100.), (.060,100.), (.0599,100.)])
def test_whole_window_retreat_boundary_is_strict(tmp_path, distance, expected):
    rec = episode(tmp_path, retreat=distance)
    assert score(rec.evidence(rec.frames)).value == expected


def test_early_close_tcp_in_final_window_cannot_be_hidden_by_last_pose(tmp_path):
    def change(row, tick):
        if tick == 41:
            row['tcps']['Right'][0] = .010
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 100.


@pytest.mark.parametrize('options', [dict(hold=20), dict(keep_grip=True)])
def test_current_delivery_without_complete_stable_release_is_five(tmp_path, options):
    rec = episode(tmp_path, **options)
    assert score(rec.evidence(rec.frames)).value == 100.


@pytest.mark.parametrize('damage', ['position', 'support'])
def test_delivery_dropped_at_cutoff_does_not_keep_historical_credit(tmp_path, damage):
    def change(row, tick):
        if tick == 160:
            if damage == 'position':
                row['tray_position'][0] = .1
                for body in row['balls'].values():
                    body['position'][0] += .1
            else:
                row['contacts'] = []
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 0.


@pytest.mark.parametrize('n', [2, 3, 4])
def test_history_exit_then_full_recovery_earns_twenty(tmp_path, n):
    def change(row, tick):
        if tick == 10:
            row['balls']['Ball'+str(n-1)]['position'][0] += rec_radius(n) + .001
    rec = episode(tmp_path, n, changes=change)
    facts = rec.facts()
    assert facts['radius_violations_latched'] == [n-1] and not facts['all_balls_retained']
    assert score(rec.evidence(rec.frames)).value == {2:30.,3:22.,4:15.}[n]


@pytest.mark.parametrize('distance,expected', [(.0300,100.), (.030000001,100.),
    (.0302,100.), (.0305,100.), (.030500001,30.)])
def test_circle_uses_published_half_mm_calibration_and_recovery_tier(tmp_path, distance, expected):
    def change(row, tick):
        if tick == 10:
            row['balls']['Ball1']['position'][1] += distance
    rec = episode(tmp_path, 2, changes=change)
    assert score(rec.evidence(rec.frames)).value == expected


def test_history_exit_without_final_recovery_is_zero(tmp_path):
    def change(row, tick):
        if tick >= 10:row['balls']['Ball2']['position'][0] += .026
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 0.


@pytest.mark.parametrize('recovered', [False, True])
def test_obstacle_hit_is_zero_even_with_successful_delivery(tmp_path, recovered):
    def change(row, tick):
        if tick == 10:
            row['contacts'].append(contact(TRAY+'/Shell/Collider', '/World/PR7/Column0'))
            if recovered:row['balls']['Ball2']['position'][0] += .026
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 0.


def test_direct_ball_contact_without_circle_exit_is_allowed(tmp_path):
    def change(row, tick):
        if tick == 10:
            row['contacts'].append(contact('/World/PR7/Ball2', '/World/RobotRight/Asset/link7/col'))
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 100.


@pytest.mark.parametrize('unsupported,expected', [(11,100.),(12,100.)])
def test_goal_support_must_exceed_ninety_percent_of_the_complete_window(tmp_path, unsupported, expected):
    def change(row, tick):
        if 41 <= tick < 41+unsupported:
            row['contacts'] = []
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == expected


def test_delivery_does_not_require_grasp_and_lift_evidence(tmp_path):
    def change(row,tick):
        if tick <= 40:
            row['contacts'] = []
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 100.


@pytest.mark.parametrize('damage', ['missing_ball', 'missing_tick', 'phase', 'nan'])
def test_corrupt_measurements_are_infra_errors_not_model_failure(tmp_path, damage):
    def change(row, tick):
        if tick != 2:
            return
        if damage == 'missing_ball':
            row['balls'].pop('Ball2')
        elif damage == 'missing_tick':
            row['tick'] += 1
        elif damage == 'phase':
            row['phase'] = 'finished'
        else:
            row['tray_position'][0] = float('nan')
    with pytest.raises(ValueError):
        episode(tmp_path, changes=change)


def test_no_actions_with_native_reset_evidence_is_valid_zero(tmp_path):
    cfg = dict(level='medium', radius_mm=25., ball_count=3)
    geometry = dict(goal=[0.,.15,.784],polygon=[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],floor_z=0.,rim_z=.03)
    rec = Recorder('DR4','synthetic',cfg,geometry,frame(0,3,support=False),trace_path=tmp_path/'raw.jsonl')
    assert score(rec.evidence(0)).value == 0.


def test_evidence_is_bounded_and_replay_recovers_the_same_facts(tmp_path):
    rec = episode(tmp_path, hold=4000)
    evidence = rec.evidence(rec.frames)
    assert len(json.dumps(dict(evidence.values), default=dict)) < 5000
    assert reconstruct(evidence) == rec.facts()
    assert len((tmp_path/'raw.jsonl').read_bytes().splitlines()) == rec.frames


def test_terminal_delivery_needs_no_lift_release_retreat_or_wait(tmp_path):
    def change(row, tick):
        row['contacts'] = [contact(TRAY+'/Shell/Collider', GOAL+'/Top')]
        row['tcps']['Left'] = [0., .15, row['tray_position'][2]]
    rec = episode(tmp_path, hold=1, keep_grip=True, changes=change)
    assert score(rec.evidence(rec.frames)).value == 100.


def test_direct_ball_touch_is_not_an_independent_penalty(tmp_path):
    def change(row, tick):
        if tick == 10:
            row['contacts'].append(contact('/World/PR7/Ball2', '/World/RobotRight/Asset/link7/col'))
    rec = episode(tmp_path, changes=change)
    assert score(rec.evidence(rec.frames)).value == 100.
