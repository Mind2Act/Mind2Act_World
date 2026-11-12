"""FINAL-VERSION: final support and center membership, independent food credit."""
from copy import deepcopy

import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.tasks.CP01.evidence import Evidence
from mind2act_eval.tasks.CP01.scorer import score
def _trace(difficulty, unused):
    count = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
    setup = dict(difficulty=difficulty, items=[{'kind': k} for _ in range(count) for k in 'abc'],
                 orders=[list('abc') for _ in range(count)], tray_spec={'tray_size_m': [.36, .14]})
    base = dict(quaternion=[1, 0, 0, 0], local_center=[0, 0, 0], half_extents=[.01]*3,
                bilateral=False, touch_robot=False, touch_table=False, touch_trays=[], tilt_deg=0)
    foods = [dict(base, index=i, position=[.4*(i//3)+(i%3-1)*.04, 0, .814], touch_trays=[i//3])
             for i in range(3*count)]
    trays = [dict(base, index=3*count+j, position=[.4*j, 0, .768], touch_table=True) for j in range(count)]
    return setup, [dict(bodies=foods+trays)]


def grade(difficulty='medium', mutate=lambda setup, row: None):
    setup, rows = _trace(difficulty, 'untouched')
    row = deepcopy(rows[-1])
    mutate(setup, row)
    row.update(physics_tick=1, phase='execution')
    facts = Evidence(setup, 30).sample(row)
    return score(NativeEvidence('CP01', 'cpu-final', 1, dict(facts, difficulty=difficulty))), facts


def test_one_terminal_frame_needs_no_pickup_release_tilt_or_stability():
    def contact(setup, row):
        for b in row['bodies']:
            b.update(touch_robot=True, bilateral=False, tilt_deg=90)
    result, facts = grade(mutate=contact)
    assert result.value == 100
    assert not facts['protocol_violations']
    assert result.components['scoring_version'] == SCORING_VERSION


def test_food_credit_is_independent_of_table_ready_tray():
    def unsupported_tray(setup, row):
        row['bodies'][-1]['touch_table'] = False
    result, facts = grade(mutate=unsupported_tray)
    assert facts['credited_foods'] == 6
    assert facts['tray_steps'] == 1
    assert result.value == pytest.approx(10 * 6 / 6)


def test_center_inside_does_not_require_whole_food_bbox_or_fixed_floor_height():
    def edge(setup, row):
        row['bodies'][0]['position'] = [.175, 0, .84]
        row['bodies'][0]['half_extents'] = [.04, .04, .04]
    assert grade(mutate=edge)[0].value == 100


def test_robot_contact_without_real_tray_support_gets_no_food_credit():
    def suspended(setup, row):
        row['bodies'][0].update(touch_robot=True, touch_trays=[])
    result, facts = grade(mutate=suspended)
    assert facts['credited_foods'] == 5
    assert result.value == pytest.approx(10 * 5 / 6)


def test_orders_must_finish_showing_before_full_success_but_notice_actions_are_valid():
    setup,rows=_trace('easy',None)
    row=rows[0];row.update(physics_tick=1,phase='observation')
    e=Evidence(setup,30);first=e.sample(row)
    assert first['credited_foods']==3 and not first['task_success']
    row.update(physics_tick=2,phase='execution')
    assert e.sample(row)['task_success']
