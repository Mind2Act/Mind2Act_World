import json
from pathlib import Path

import numpy as np
import pytest

import f1_factory

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.F1.evidence import measure_row
from mind2act_eval.tasks.F1.instances import layout, spec_from_row
from mind2act_eval.tasks.F1.mechanism import Slider, Timeline
from mind2act_eval.tasks.F1.scorer import score

LAY = layout()
HOME = list(np.radians(LAY['home_joints_deg'])) * 2


def spec():
    return spec_from_row(f1_factory.row(), asset_version='t')


class Rig:
    """Closed loop without physics: scripted press intents drive Slider/Timeline exactly as the scene does."""

    def __init__(self, s=None, away=False, press_arm=None):
        s = s or spec()
        from mind2act_eval.tasks.F1.judge import plain
        self.case, self.protocol = plain(s.config['case']), plain(s.config['protocol'])
        self.slider = Slider(self.case, LAY, self.protocol['press_threshold_m'], self.protocol['release_threshold_m'])
        self.timeline = Timeline(self.slider, LAY, round(self.case['timing']['hold_seconds'] * 120))
        self.rows, self.tick, self.goal, self.held = [], 0, False, None
        self.hold = 10
        self.home_q = HOME
        self.press_arm = press_arm
        self.contact_arm = {b: 'Left' if xy[0] < 0 else 'Right'
                            for b, xy in zip(self.case['buttons'], self.case['layout']['button_centers_xy'])}

    def run(self, ticks, press_plan, q=None):
        """press_plan(tick, rig) -> button or None for a tick; a press holds the cap down for 10 ticks."""
        for _ in range(ticks):
            self.tick += 1
            want = press_plan(self.tick, self)
            if want is not None and self.held is None:
                self.held = (want, self.tick)
            depth = {b: 0. for b in self.case['buttons']}
            touch = {b: False for b in self.case['buttons']}
            if self.held:
                b, t0 = self.held
                if self.tick - t0 < self.hold:
                    depth[b], touch[b] = .004, True
                else:
                    self.held = None
            arm = {b: {s: ([[f'/World/Robot{s}/link7/c', '/World/Button%s/Asset/E_button_01_13/m' % b]] if touch[b] and s == (self.press_arm or self.contact_arm[b]) else [])
                       for s in ('Left', 'Right')} for b in self.case['buttons']}
            joints = list(q) if q else HOME
            events, failures = self.slider.update(self.tick, depth, touch, arm)
            row = measure_row(self.tick, self.slider.stage, joints, self.slider, depth, touch, self.goal, events, failures)
            self.rows.append(row)
            released = all(not touch[b] and depth[b] < .0008 for b in touch)
            def show():
                self.goal = True
            self.timeline.advance(self.tick, self.slider.ready_to_reset(released), show)

    def evidence(self):
        return NativeEvidence('F1', 'x', self.tick, dict(case=self.case, protocol=self.protocol, rows=self.rows))


def plan(order, route):
    state = dict(sequence=list(order) + [None] * 0)

    def f(tick, rig):
        s = rig.slider
        if s.stage == 'exploration' and not s.pending and tick % 40 == 0:
            for b in order:
                if b not in s.observed and not rig.held:
                    return b
        if s.stage == 'execution' and not rig.held and tick % 40 == 0:
            done = len([r for r in rig.rows if r['events'] and r['stage'] == 'execution'])
            if done < len(route):
                return route[done]
        return None
    return f


def test_full_success_scores_100_any_probe_order():
    for order in (['B', 'A'], ['A', 'B']):
        rig = Rig()
        rig.run(1800, plan(order, ['B']))
        s = score(rig.evidence())
        assert s.value == 100., (order, s.components['protocol_violations'])
        assert s.components['efficiency'] == dict(execution_presses=1, shortest_presses=1, path_redundancy=0, effective_probes=2)


def test_detour_arrival_is_the_ten_point_tier():
    rig = Rig()
    rig.run(2400, plan(['B', 'A'], ['A', 'B', 'B']))  # right, left, left: reaches the goal but A=3 > L*=1
    s = score(rig.evidence())
    assert s.value == 15 and s.components['tier'] == 'at_goal_released'
    assert s.components['efficiency']['path_redundancy'] == 2


def test_arrival_with_last_key_still_down_is_five():
    rig = Rig()
    pl = plan(['B', 'A'], ['B'])
    for _ in range(3000):
        rig.run(1, pl)
        if rig.slider.stage in ('reset', 'goal_reveal'):
            rig.hold = 400  # the formal press will be held down through the end of the episode
        if rig.slider.stage == 'execution' and rig.slider.pos == rig.case['execution']['target'] and rig.held \
                and rig.slider.motion_complete():
            break
    s = score(rig.evidence())
    assert s.value == 7.5 and s.components['tier'] == 'at_goal_last_key_down'


def test_cutoff_while_the_slider_is_still_moving_is_zero_not_five():
    rig = Rig()
    pl = plan(['B', 'A'], ['B'])
    for _ in range(3000):
        rig.run(1, pl)
        if rig.slider.stage == 'execution' and rig.slider.pos == rig.case['execution']['target'] and rig.held:
            break  # first tick of the formal press: logical goal reached, display not yet there
    assert not rig.slider.motion_complete()
    s = score(rig.evidence())
    assert s.value == 0. and s.components['tier'] == 'not_at_goal'


def test_not_at_goal_at_cutoff_is_zero_and_exploration_only_is_zero():
    rig = Rig()
    rig.run(1400, plan(['B', 'A'], []))  # explored and reset, then nothing
    assert score(rig.evidence()).value == 0.


def test_violation_overrides_the_low_tiers():
    rig = Rig()
    pl = plan(['B', 'A'], ['A', 'B', 'B'])
    rig.run(2400, lambda t, r: 'A' if r.slider.stage == 'reset' and t % 7 == 0 else pl(t, r))
    s = score(rig.evidence())
    assert s.value == 0. and s.components['tier'] == 'violation_or_prerequisite'


def test_repeated_probes_are_allowed_and_any_order():
    rig = Rig()
    queue = ['A', 'A', 'B']
    rest = plan(['B', 'A'], ['B'])

    def pl(t, r):
        if r.slider.stage == 'exploration' and queue and not r.slider.pending and t % 40 == 0 and not r.held:
            return queue.pop(0)
        return rest(t, r) if not queue else None
    rig.run(2200, pl)
    s = score(rig.evidence())
    assert s.components['efficiency']['effective_probes'] == 2 and 'exploration_duplicate' not in s.components['protocol_violations']


def test_shortest_is_a_bfs_over_the_real_transition_graph():
    from mind2act_eval.tasks.F1.space import shortest
    line = dict(kind='line', positions=5)
    assert shortest(line, dict(A='right', B='left'), 3, 2) == 1
    grid = dict(kind='grid', side=3)
    m = dict(A='left', B='right', C='up', D='down')
    assert shortest(grid, m, [2, 0], [0, 1]) == 3
    with pytest.raises(ValueError):
        shortest(line, dict(A='right', B='right'), 3, 0)


def test_skipping_exploration_never_resets_or_succeeds():
    rig = Rig()
    rig.run(1500, lambda t, r: 'A' if t == 40 else None)
    s = score(rig.evidence())
    assert s.value == 0. and rig.slider.stage == 'exploration'


def test_reset_never_finishing_does_not_earn_a_route_score():
    rig = Rig()
    plan_ = plan(['B', 'A'], ['B'])
    rig.run(1500, lambda t, r: 'A' if r.slider.stage == 'reset' and t % 7 == 0 else plan_(t, r))
    s = score(rig.evidence())
    assert s.value == 0. and rig.slider.stage == 'reset'


@pytest.mark.parametrize('press_arm', [None, 'Left', 'Right'])
def test_arms_away_from_home_and_either_pressing_arm_can_complete(press_arm):
    rig = Rig(press_arm=press_arm)
    away = list(HOME)
    away[0] += .3
    rig.run(1800, plan(['B', 'A'], ['B']), q=away)
    assert rig.slider.stage == 'execution'
    assert score(rig.evidence()).value == 100.


@pytest.mark.parametrize('press_arm', ['Left', 'Right'])
def test_either_arm_can_press_buttons_on_both_sides(press_arm):
    rig = Rig(press_arm=press_arm)
    rig.run(1800, plan(['B', 'A'], ['B']))
    assert score(rig.evidence()).value == 100.


def test_probe_cannot_finish_until_both_key_release_and_slider_motion_finish():
    rig = Rig()
    rig.hold = 200
    rig.run(150, lambda t, r: 'A' if t == 1 else None)
    assert rig.slider.motion_complete() and rig.slider.pending
    assert not rig.slider.ready_to_reset(False)
    rig.run(100, lambda t, r: None)
    assert not rig.slider.pending


def test_scorer_rejects_clock_gap():
    rig = Rig()
    rig.run(50, lambda t, r: None)
    ev = rig.evidence()
    rows = list(ev.values['rows'])[1:]
    with pytest.raises(ValueError):
        score(NativeEvidence('F1', 'x', 50, dict(case=ev.values['case'], protocol=ev.values['protocol'], rows=rows)))


def test_instance_drops_expert_order_and_checks_tier():
    s = spec()
    assert 'exploration' not in s.config['case'] and s.config['support_z'] > 0
    row = f1_factory.row()
    row['difficulty'] = 'hard'
    with pytest.raises(ValueError):
        spec_from_row(row, asset_version='t')


def test_non_baseline_physics_clock_is_rejected():
    from mind2act_eval.contracts import ClockSpec
    with pytest.raises(ValueError):
        spec_from_row(f1_factory.row(), asset_version='t', clock=ClockSpec(240, 30, 30))


def test_reset_starts_as_soon_as_probe_motion_and_release_complete():
    rig = Rig()
    rig.run(1, lambda t, r: None)
    rig.slider.observed = dict.fromkeys(rig.case['buttons'], True)
    rig.slider.motor_tick = rig.slider.feedback_ticks
    rig.timeline.advance(rig.tick, True, lambda: None)
    assert rig.slider.stage == 'reset'


def test_goal_appears_when_reset_motion_finishes_without_extra_delay():
    rig = Rig()
    rig.slider.begin_reset()
    rig.timeline.deadline = 10000  # a historical timer cannot postpone physical readiness
    rig.run(rig.slider.feedback_ticks, lambda t, r: None)
    assert rig.goal and rig.slider.stage == 'execution'


def test_first_released_arrival_gets_full_credit_without_post_arrival_hold():
    rig = Rig()
    pl = plan(['B', 'A'], ['B'])
    for _ in range(3000):
        rig.run(1, pl)
        if (rig.slider.stage == 'execution' and rig.slider.pos == rig.case['execution']['target']
                and rig.slider.motion_complete() and rig.held is None):
            break
    assert score(rig.evidence()).value == 100.


def test_next_probe_during_feedback_animation_is_not_a_failure():
    rig = Rig()
    formal = plan([], ['B'])
    rig.run(1500, lambda t, r: 'A' if t == 1 else 'B' if t == 20 else formal(t, r))
    result = score(rig.evidence())
    assert result.value == 100., result.components


def test_archived_feedback_wait_failure_is_ignored_but_physical_rows_replayed():
    from mind2act_eval.tasks.F1.judge import plain
    rig = Rig()
    rig.run(1500, plan(['B', 'A'], ['B']))
    values = plain(rig.evidence().values)
    for row in values['rows']:
        if row['events'] and row['stage'] == 'exploration':
            row['mechanism_failures'].append('new_probe_before_release_and_slider_completion')
        if row['stage'] == 'execution':
            row['stage'] = 'goal_reveal'
            if row['events']:
                row['mechanism_failures'].append('press_during_environment_reset')
    result = score(NativeEvidence('F1', 'x', rig.tick, values))
    assert result.value == 100., result.components


def test_button_motion_without_finger_proof_can_complete_the_task():
    from mind2act_eval.tasks.F1.judge import plain
    rig = Rig()
    rig.run(1200, plan(['A','B'], ['B']))
    values = plain(rig.evidence().values)
    for row in values['rows']:
        for key in row['button_contacts']:
            row['button_contacts'][key] = False
        for event in row['events']:
            event['finger_contact'] = False
            event['contact_colliders'] = []
    assert score(NativeEvidence('F1','x',rig.tick,values)).value == 100


def test_reset_phase_press_can_be_repaired_without_a_permanent_failure():
    rig=Rig(); state={'reset_presses':0}
    standard=plan(['A','B'],['B'])
    def inputs(t,r):
        if r.slider.stage=='reset' and r.held is None:
            if state['reset_presses']==0:
                state['reset_presses']=1
                return 'A'
            if state['reset_presses']==1:
                state['reset_presses']=2
                return 'B'
        return standard(t,r)
    rig.run(1600,inputs)
    assert score(rig.evidence()).value == 100


def test_only_the_last_formal_button_return_controls_the_terminal_tier():
    source=f1_factory.row();source['difficulty']='medium'
    case=source['config']['native_inputs']['config']
    case.update(difficulty='Medium',space=dict(kind='grid',side=3),buttons=list('ABCD'),
                mapping=dict(A='right',B='up',C='left',D='down'),exploration=list('ABCD'),
                execution=dict(start=[0,0],target=[1,1],route=['A','B']))
    case['layout']['button_centers_xy']=[[-.07,-.145],[.07,-.145],[-.07,-.22],[.07,-.22]]
    rig=Rig(spec_from_row(source,asset_version='t'))
    rig.run(2200,plan(list('ABCD'),['A','B']))
    from mind2act_eval.tasks.F1.judge import plain
    values=plain(rig.evidence().values);holding=False
    for row in values['rows']:
        if row['stage']=='execution' and any(e['button']=='A' for e in row['events']):holding=True
        if holding:
            row['button_depths']['A']=.004
            row['button_contacts']['A']=True
    assert score(NativeEvidence('F1','x',rig.tick,values)).value == 100
