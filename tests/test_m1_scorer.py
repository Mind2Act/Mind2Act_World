import json
from pathlib import Path

import numpy as np
import pytest

import m1_factory

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.M1.environment import compile_environment
from mind2act_eval.tasks.M1.evidence import ContactTable, Recorder
from mind2act_eval.tasks.M1.instances import layout, plain, spec_from_row
from mind2act_eval.tasks.M1.scorer import Tracker, score

LAY = layout()


def spec():
    return spec_from_row(m1_factory.row(), asset_version='t')


class Film:
    """Synthesises native rows: covered observation, uncover, extraction, then stable placements."""

    def __init__(self, route, first_cover=None, extract=True, release_fingers=False, hold_ticks=65, instance_row=None):
        s = spec_from_row(instance_row, asset_version='t') if instance_row is not None else spec()
        self.M = plain(s.config['manifest'])
        self.hold_ticks = hold_ticks
        self.route, self.extract, self.hold_fingers = route, extract, release_fingers
        self.first_cover = first_cover or self.M['target_cover_id']
        self.frames, self.arrivals, _, _ = compile_environment(self.M, LAY)
        self.start = len(self.frames) + 1
        self.rows = []
        dims = self.M['bodies']['workpiece_D']['dimensions']
        self.dims = dims
        self.initial_z = 0.684
        self.baseline = {c: 0.662 for c, body in self.M['bodies'].items() if body['kind'] == 'cover'}
        stations = self.M['submission_stations']
        self.pad_z = {k: v['support_top_z'] + dims[2] / 2 for k, v in stations.items()}

    def row(self, tick, center, finger=False, cover=None, support=None, opened=True):
        covers = {c: [0, 0, 0.662] for c in self.baseline}
        fingers = {c: False for c in self.baseline}
        if cover:
            covers[cover] = [0, 0, 0.662 + 0.06]
            fingers[cover] = True
        self.rows.append(dict(tick=tick, target_center=list(center), target_linear_velocity=[0, 0, 0],
                              target_angular_velocity=[0, 0, 0], target_finger=finger, target_robot_contact=finger,
                              cover_centers=covers,
                              cover_fingers=fingers, gripper_open=[opened, opened],
                              station_support={k: k == support for k in self.M['submission_stations']}))

    def build(self):
        t = 0
        d_xy = self.M['stations']['D']['center_xy']
        # schedule: target follows its station for arrival checks
        arrivals = {a['step']: a['target_station'] for a in self.arrivals}
        at = self.M['case']['initial_target_station']
        while t < self.start - 1:
            t += 1
            at = arrivals.get(t, at)
            xy = self.M['stations'][at]['center_xy']
            self.row(t, [xy[0], xy[1], self.initial_z])
        t += 1
        self.row(t, [*self.M['stations'][at]['center_xy'], self.initial_z], cover=self.first_cover, finger=True)  # uncover
        if self.extract:
            for _ in range(self.hold_ticks):  # held lifted with finger contact for more than 0.5 s
                t += 1
                self.row(t, [0, 0, self.initial_z + 0.06], finger=True)
        for st in self.route:
            xy = self.M['submission_stations'][st]['center_xy']
            for _ in range(5):  # lifted above pad then lowered
                t += 1
                self.row(t, [xy[0], xy[1], self.pad_z[st] + 0.08], finger=True)
            for _ in range(130):  # stable > 1.0 s, released, supported
                t += 1
                self.row(t, [xy[0], xy[1], self.pad_z[st]], finger=self.hold_fingers, support=st)
        return self

    def evidence(self):
        v = dict(ticks=self.rows, physical_start=self.start, arrivals=self.arrivals,
                 station_xy={k: s['center_xy'] for k, s in self.M['stations'].items()},
                 submission_stations=self.M['submission_stations'], target_dimensions=self.dims,
                 closed_baseline=self.baseline, target_initial_z=self.initial_z, target_cover_id=self.M['target_cover_id'],
                 expected_path=self.M['case']['expected_path'])
        return NativeEvidence('M1', 'x', len(self.rows), v)


def test_environment_compiles_for_all_tiers_with_expected_route():
    rows = (m1_factory.row(), m1_factory.row('medium', swaps=(('D', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'B'))), m1_factory.hard_row())
    for row in rows:
        M = plain(spec_from_row(row, asset_version='t').config['manifest'])
        frames, arrivals, _, _ = compile_environment(M, LAY)
        assert len(arrivals) == len(M['case']['swaps']) and frames[0][2] == {}


def test_non_baseline_physics_clock_is_rejected_at_entry_and_environment():
    from mind2act_eval.contracts import ClockSpec
    from mind2act_eval.tasks.M1.instances import require_baseline_clock
    with pytest.raises(ValueError):
        spec_from_row(m1_factory.row(), asset_version='t', clock=ClockSpec(240, 30, 30))
    with pytest.raises(ValueError):
        require_baseline_clock(ClockSpec(60, 30, 30))


def test_correct_uncover_extract_and_route_scores_100():
    s = score(Film(['D', 'B', 'C', 'A']).build().evidence())
    assert s.value == 100. and list(s.components['route']) == ['D', 'B', 'C', 'A'] and s.components['completion'] == 1.


@pytest.mark.parametrize('opened', ([False, True], [True, False], [False, False]))
@pytest.mark.parametrize('instance_row', (
    m1_factory.row(),
    m1_factory.row('medium', swaps=(('D', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'B'))),
    m1_factory.row('hard', stations='ABCDEF', target='B', spacing=.12,
                   swaps=(('B', 'E'), ('E', 'A'), ('A', 'B'), ('B', 'D'), ('D', 'B'))),
), ids=('easy', 'medium', 'hard'))
def test_released_route_scores_100_regardless_of_gripper_opening(instance_row, opened):
    expected = instance_row['config']['native_inputs']['manifest']['case']['expected_path']
    film = Film(expected, instance_row=instance_row).build()
    for row in film.rows:
        row['gripper_open'] = opened
    result = score(film.evidence())
    assert result.value == 100.
    assert list(result.components['route']) == expected


def test_wrong_cover_first_does_not_disqualify_correct_route():
    s = score(Film(['D', 'B', 'C', 'A'], first_cover='cover_B').build().evidence())
    assert s.value == 100. and not s.components['correct_first_uncover']


def test_wrong_order_with_one_wrong_submission_scores_5():
    s = score(Film(['D', 'B', 'A', 'C']).build().evidence())  # C=3, W=1, m=2
    assert s.value == 7.5 and s.components['lcs'] == 3 and s.components['missing_or_wrong'] == 2


def test_one_missing_node_scores_10_and_two_missing_5_and_three_0():
    assert score(Film(['D', 'B', 'C']).build().evidence()).value == 15
    assert score(Film(['D', 'B']).build().evidence()).value == 7.5
    assert score(Film(['D']).build().evidence()).value == 0.


def test_wrong_station_substitution_is_5():
    s = score(Film(['D', 'B', 'C', 'D']).build().evidence())  # A=DBCD vs R=DBCA: C=3, W=1, m=2
    assert s.value == 7.5 and not s.components['success']


def test_cover_premise_is_not_a_score_gate():
    assert score(Film(['D', 'B', 'C', 'A'], first_cover='cover_B').build().evidence()).value == 100.


def test_lift_hold_duration_is_not_an_eligibility_gate():
    assert score(Film(['D','B','C','A'], hold_ticks=0).build().evidence()).value == 100.


def test_missing_extraction_scores_0():
    s = score(Film([], extract=False).build().evidence())
    assert s.value == 0. and not s.components['extracted'] and list(s.components['route']) == []


def test_finger_contact_does_not_hide_real_station_support():
    s = score(Film(['D'], release_fingers=True).build().evidence())
    assert list(s.components['route']) == ['D']


def test_palm_contact_does_not_disqualify_supported_route():
    film = Film(['D', 'B', 'C', 'A']).build()
    for row in film.rows:
        if any(row['station_support'].values()):
            assert not row['target_finger'] and all(row['gripper_open'])
            row['target_robot_contact'] = True
    result = score(film.evidence())
    assert result.value == 100.
    assert list(result.components['route']) == ['D','B','C','A']


@pytest.mark.parametrize('bad_value', ('missing', None, 0, 'false'))
def test_robot_contact_annotation_is_not_scoring_evidence(bad_value):
    film = Film(['D', 'B', 'C', 'A']).build()
    if bad_value == 'missing':
        del film.rows[-1]['target_robot_contact']
    else:
        film.rows[-1]['target_robot_contact'] = bad_value
    assert score(film.evidence()).value == 100.


@pytest.mark.parametrize('interruption', ('robot_contact', 'support', 'drift', 'linear', 'angular', 'short'))
def test_only_actual_departure_creates_another_arrival_without_waits(interruption):
    film = Film(['D']).build()
    resting = [row for row in film.rows if row['station_support']['D']]
    row = resting[65]
    if interruption == 'robot_contact':
        row['target_robot_contact'] = True
    elif interruption == 'support':
        row['station_support']['D'] = False
    elif interruption == 'drift':
        row['target_center'][0] += .0031
    elif interruption == 'linear':
        row['target_linear_velocity'] = [.03, 0, 0]
    elif interruption == 'angular':
        row['target_angular_velocity'] = [0, 0, .2]
    else:
        film.rows = film.rows[:-10]  # 120 positions span less than one second.
    expected = ['D','D'] if interruption == 'support' else ['D']
    assert list(score(film.evidence()).components['route']) == expected
    # A fresh full release window can still submit; interruptions do not poison the route.
    final = film.rows[-1]
    for _ in range(121):
        film.rows.append(dict(final, tick=film.rows[-1]['tick'] + 1))
    assert list(score(film.evidence()).components['route']) == expected


def test_no_finger_extraction_proof_is_required():
    film = Film(['D', 'B', 'C', 'A']).build()
    for row in film.rows:
        row['target_finger'] = False
    result = score(film.evidence())
    assert not result.components['extracted'] and result.value == 100.


@pytest.mark.parametrize('side', ('Left', 'Right'))
@pytest.mark.parametrize('link', ('link1', 'link6', 'link7', 'link8'))
def test_recorder_measures_any_robot_link_contact_and_contact_loss(side, link):
    contacts = ContactTable()
    target, cover, pad = '/World/Workpiece', '/World/Cover', '/World/Pad'
    robot_link = '/World/Robot' + side + '/Asset/' + link + '/Collision'
    contacts.report((target + '/Collision', robot_link), True, .1, 0.)
    contacts.report((target + '/Collision', pad + '/Collision'), True, .1, 0., normals=[(0.,0.,1.)])
    contacts.report((cover + '/Collision', '/World/RobotLeft/Asset/link6/Collision'), True, .1, 0.)
    recorder = Recorder(dict(target=target, cover=cover), lambda ident: [0, 0, .7],
                        lambda: ([0, 0, 0], [0, 0, 0]), contacts, dict(D=pad), ['cover'],
                        lambda: [False, False])
    held = recorder.record(1, 'target')
    assert held['target_robot_contact']
    assert held['target_finger'] == (link in ('link7', 'link8'))
    assert held['station_support']['D'] and not held['cover_fingers']['cover']

    contacts.report((target + '/Collision', robot_link), False, 0., 1.)
    # A similarly named non-robot body must not prevent release.
    contacts.report((target + '/Collision', '/World/RobotLeftExtra/Collision'), True, .1, 0.)
    released = recorder.record(2, 'target')
    assert not released['target_robot_contact'] and not released['target_finger']
    assert released['station_support']['D'] and released['gripper_open'] == [False, False]




def test_environment_divergence_is_invalid_not_zero():
    film = Film(['D', 'B', 'C', 'A']).build()
    e = film.arrivals[0]['step']
    film.rows[e - 1]['target_center'][0] += 0.5
    with pytest.raises(ValueError):
        score(film.evidence())


def test_answer_is_truncated_to_n_submissions():
    film = Film(['D', 'B', 'C', 'A', 'B']).build()  # a fifth submission can never rescue or hurt
    t = Tracker(plain(dict(film.evidence().values)), LAY)
    for r in film.rows:
        t.observe(r)
    assert len(t.ledger.placements) == 5 and t.route == ['D', 'B', 'C', 'A']
    assert score(film.evidence()).value == 0.  # The first N stay fixed, but the terminal object left A.


def test_instance_drops_planner_data():
    s = spec()
    m = s.config['manifest']
    assert 'motion_jobs' not in m and 'source_plan' not in s.config and m['case']['expected_path'][0] == 'D'


@pytest.mark.parametrize('opened', ([True, True], [False, True], [False, False]))
def test_scene_online_tracker_matches_the_offline_scorer(monkeypatch, opened):
    """Drive the real RouteScene hooks (physical transition -> Tracker -> is_complete) with scripted rows."""
    from types import SimpleNamespace
    from mind2act_eval.tasks.M1 import mechanism
    from mind2act_eval.tasks.M1.scene import RouteScene
    film = Film(['D', 'B', 'C', 'A']).build()
    for row in film.rows:
        row['gripper_open'] = opened
    locked, dynamic = [], []
    monkeypatch.setattr(mechanism, 'set_dynamic', lambda stage, path: dynamic.append(path))
    rows = iter(film.rows)
    recorder = SimpleNamespace(ticks=film.rows, record=lambda tick, target: next(rows))
    M, B = film.M, film.M['bodies']
    s = spec()
    scene = RouteScene(s, None, dict(
        M=M, B=B, lay=LAY, frames=[], arrivals=film.arrivals, writer=SimpleNamespace(lock=locked.extend), bodies={},
        stage=None, centers=lambda ident: [0, 0, film.initial_z if ident.startswith('workpiece') else 0.662],
        covers=list(film.baseline), target='workpiece_D', cover='cover_D', recorder=recorder, subscription=None))
    scene.before_step(1)  # frames=[] -> the physical transition happens on the first step
    assert scene.physical_start == 1 and dynamic == [B['workpiece_D']['prim_path'], B['cover_D']['prim_path']]
    assert not scene.is_complete()
    for tick in range(1, len(film.rows) + 1):
        scene.after_step(tick)
    assert scene.tracker.route == ['D', 'B', 'C', 'A'] and scene.is_complete()
    assert list(scene.evidence(len(film.rows)).values['expected_path']) == M['case']['expected_path']


def test_route_needs_only_actual_supported_arrivals_not_cover_or_lift_history():
    film = Film(['D', 'B', 'C', 'A'], first_cover='cover_B', extract=False, release_fingers=True).build()
    for row in film.rows:
        row['target_linear_velocity'] = [.3, 0., 0.]
        row['target_angular_velocity'] = [0., 0., 1.]
    assert score(film.evidence()).value == 100.


def test_terminal_endpoint_support_is_required_even_after_exact_first_n():
    film = Film(['D', 'B', 'C', 'A']).build()
    film.row(len(film.rows)+1, [0.,0.,1.], support=None)
    assert score(film.evidence()).value == 0.
