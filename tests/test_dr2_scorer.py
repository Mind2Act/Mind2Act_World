import json

import pytest

import dr2_factory
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.DR2.event_clock import EventClock
from mind2act_eval.tasks.DR2.instances import spec_from_row
from mind2act_eval.tasks.DR2.scorer import score

SEQ = [0, 1, 2, 1, 2, 1, 1, 0, 2, 2]
P, R, G = 480, 120, 36  # easy: 4 s / 1 s / 0.3 s at 120 Hz


def drive(plan, seq=SEQ, length=20000, extra=0):
    """plan(clock, tick) -> (depression, contacts). Runs until the round is done (plus `extra` ticks)."""
    live = EventClock(seq, P, R, G)
    ticks, tail = [], None
    for t in range(1, length + 1):
        dep, ct = plan(live, t)
        live.advance(t, dep, ct)
        ticks.append(dict(tick=t, button_depression_m=dep, button_contacts=ct))
        if live.done and tail is None:
            tail = t
        if tail is not None and t >= tail + extra:
            break
    return ticks, live


def evidence(ticks, live, seq=SEQ):
    keys = ('index', 'button', 'cue_tick', 'press_tick', 'release_tick', 'status', 'reason')
    return NativeEvidence('DR2', 'x', len(ticks), dict(
        sequence=seq, press_ticks=P, release_ticks=R, gap_ticks=G, release_depression_m=.0008, press_depression_m=.0025,
        ticks=ticks, live_records=[{k: r[k] for k in keys} for r in live.records]))


def down(b):
    d = [0., 0., 0.]
    d[b] = .004
    return d, [i == b for i in range(3)]


UP = ([0., 0., 0.], [False] * 3)


def good(live, t, skip=None, wrong=None, late=None, hold=10):
    c = live.current
    if c and c['press_tick'] is None and t - c['cue_tick'] == 20:
        if c['index'] == skip:
            return UP
        return down(wrong(c) if wrong and c['index'] in wrong_idx(wrong) else c['button'])
    if c and c['press_tick'] is not None and t - c['press_tick'] < (late if late and c['index'] == 3 else hold):
        return down(c['button'])
    return UP


def wrong_idx(wrong):
    return {3}


def test_perfect_run_scores_100():
    ticks, live = drive(good)
    s = score(evidence(ticks, live))
    assert s.value == 100. and s.components['round_complete'] and s.components['completion']['ratio'] == 1.


def test_wrong_button_fails_that_cue_and_the_round_continues():
    ticks, live = drive(lambda l, t: good(l, t, wrong=lambda c: (c['button'] + 1) % 3))
    s = score(evidence(ticks, live))
    assert len(live.records) == 10 and s.components['completion']['k'] == 9 and s.value == 27.
    assert [f['index'] for f in s.components['failures']] == [3]


def test_timeout_fails_only_that_cue():
    ticks, live = drive(lambda l, t: good(l, t, skip=0))
    s = score(evidence(ticks, live))
    assert s.value == 27. and s.components['failures'][0]['reason'] == 'press_timeout' and len(live.records) == 10


def test_every_cue_failing_scores_zero_after_all_ten_cues():
    ticks, live = drive(lambda l, t: UP)
    s = score(evidence(ticks, live))
    assert s.value == 0. and len(live.records) == 10 and s.components['round_complete']


def test_slow_release_fails_the_cue_but_not_the_round():
    ticks, live = drive(lambda l, t: good(l, t, late=R + 30))
    s = score(evidence(ticks, live))
    assert s.value == 27. and s.components['failures'][0]['reason'] == 'release_timeout'


def test_next_cue_waits_for_all_buttons_released_plus_gap():
    held = {}

    def plan(live, t):
        c = live.current
        if c and c['index'] == 0 and c['press_tick'] is None and t - c['cue_tick'] == 20:
            held['at'] = t
        if 'at' in held and t - held['at'] < 300 and c is None or (c and c['index'] == 0 and 'at' in held and t - held['at'] < 300):
            return down(c['button'] if c else 0)
        return good(live, t)
    ticks, live = drive(plan)
    second = live.records[1]['cue_tick']
    assert second >= held['at'] + 300 + G - 1


def test_unlit_press_between_cues_fails_the_next_cue():
    def plan(live, t):
        c = live.current
        if c is None and live.records and live.records[0]['status'] == 'valid' and len(live.records) == 1 and 5 <= t - live.records[0]['release_tick'] < 15:
            return down((live.sequence[1] + 1) % 3)
        return good(live, t)
    ticks, live = drive(plan)
    s = score(evidence(ticks, live))
    assert live.records[1]['status'] == 'valid' and s.value == 30.


def test_no_closing_gap_is_added_for_a_possible_later_unlit_press():
    late_attempts = []
    def plan(live, t):
        last = live.records[-1] if len(live.records) == 10 else None
        if last and last['status'] == 'valid' and last['release_tick'] is not None and t - last['release_tick'] == 5:
            late_attempts.append(t)
            return down((last['button'] + 1) % 3)
        return good(live, t)
    ticks, live = drive(plan)
    assert live.records[-1]['status'] == 'valid' and not live.unlit_presses
    assert not late_attempts and live.last_tick == live.records[-1]['release_tick']
    assert score(evidence(ticks, live)).value == 100.


def test_holding_does_not_repeat_a_press():
    ticks, live = drive(lambda l, t: good(l, t, hold=100))
    presses = [e for e in live.events if e['kind'] == 'press']
    assert len(presses) == 10 and score(evidence(ticks, live)).value == 100.


def test_holding_longer_than_the_release_window_fails_the_cue_without_new_presses():
    ticks, live = drive(lambda l, t: good(l, t, hold=300))
    assert len([e for e in live.events if e['kind'] == 'press']) == 10 and score(evidence(ticks, live)).value == 0.


def test_press_after_the_window_closes_fails_even_on_the_boundary_tick():
    def plan(live, t):
        c = live.current
        if c and c['index'] == 0:
            if c['press_tick'] is None and t - c['cue_tick'] == P + 1:
                return down(c['button'])
            return UP
        return good(live, t)
    ticks, live = drive(plan)
    s = score(evidence(ticks, live))
    assert live.records[0]['status'] == 'failed' and live.records[0]['reason'] == 'press_timeout' and s.value == 27.
    # exactly at the limit is still on time
    ticks, live = drive(lambda l, t: down(l.current['button']) if l.current and l.current['index'] == 0 and l.current['press_tick'] is None
                        and t - l.current['cue_tick'] == P else (UP if l.current and l.current['index'] == 0 and l.current['press_tick'] is None else good(l, t, hold=10)))
    assert live.records[0]['status'] == 'valid'


def test_gap_restarts_from_the_latest_release():
    clock = EventClock([0, 1], P, R, G)
    for t in range(1, 6):
        clock.advance(t, *(down(0) if t == 3 else UP))  # cue 0 pressed tick 3, released tick 4
    assert clock.records[0]['release_tick'] == 4 and clock.next_due == 4 + G
    clock.advance(6, *UP)
    for t in range(7, 13):
        clock.advance(t, *(down(2) if t == 10 else UP))  # unlit press at 10, released at 11
    assert clock.next_due == 11 + G and len(clock.records) == 1
    for t in range(13, 11 + G + 2):
        clock.advance(t, *UP)
    assert len(clock.records) == 2 and clock.records[1]['cue_tick'] == 11 + G
    assert clock.unlit_presses == [dict(button=2, tick=10, cue=None)]


def test_replay_disagreement_is_invalid_not_scored():
    ticks, live = drive(good)
    ev = evidence(ticks, live)
    bad = dict(ev.values, live_records=[dict(ev.values['live_records'][0], cue_tick=99)] + [dict(r) for r in ev.values['live_records'][1:]])
    with pytest.raises(ValueError):
        score(NativeEvidence('DR2', 'x', ev.tick, bad))


def test_missing_tick_rejected():
    ticks, live = drive(good)
    with pytest.raises(ValueError):
        score(evidence(ticks[1:], live))


def test_instance_mapping_drops_expert_geometry_and_checks_tier():
    spec = spec_from_row(dr2_factory.row(), asset_version='t')
    assert 'expert_geometry' not in spec.config and spec.config['press_window_s'] == 4.0
    row = dr2_factory.row()
    row['difficulty'] = 'hard'
    with pytest.raises(ValueError):
        spec_from_row(row, asset_version='t')


def test_scene_health_is_typed_and_complete_only_when_round_done():
    from types import SimpleNamespace
    from mind2act_eval.contracts import Health
    from mind2act_eval.tasks.DR2.scene import ButtonScene
    clock = EventClock(SEQ, P, R, G)
    rec = SimpleNamespace(ticks=[dict(tick=1, button_depression_m=[0., 0., float('nan')], button_contacts=[False] * 3)])
    scene = ButtonScene(spec_from_row(dr2_factory.row(), asset_version='t'), None, rec, clock, None, None, None, [], {})
    assert scene.health() == Health(False, 'button depression is not finite')
    assert not scene.is_complete()
    ticks, live = drive(good)
    scene.clock = live
    assert scene.is_complete()


def test_mechanism_edge_needs_no_finger_contact():
    def plan(clock, tick):
        depression, _ = good(clock, tick)
        return depression, [False] * 3
    ticks, live = drive(plan)
    assert score(evidence(ticks, live)).value == 100.


def test_unlit_press_does_not_erase_correct_cue_credit():
    def plan(clock, tick):
        if clock.current is None and len(clock.records) == 1:
            last = clock.records[0]
            if last['release_tick'] is not None and tick - last['release_tick'] == 5:
                return down(2)
        return good(clock, tick)
    ticks, live = drive(plan)
    assert live.valid_count == 10
    assert score(evidence(ticks, live)).value == 30.


def test_last_valid_release_completes_sequence_on_that_same_tick():
    ticks, clock = drive(good)
    assert clock.current is None and len(clock.records) == 10
    assert clock.done and clock.last_tick == clock.records[-1]['release_tick']
    assert len(ticks) == clock.last_tick


def test_last_timeout_completes_without_waiting_for_an_unlit_button_release():
    def plan(clock, tick):
        if clock.current is not None and clock.current['index'] == 9:
            return down((clock.current['button'] + 1) % 3)
        if len(clock.records) == 10:
            return down((clock.records[-1]['button'] + 1) % 3)
        return good(clock, tick)
    ticks, clock = drive(plan)
    last = clock.records[-1]
    assert clock.done and clock.current is None and last['status'] == 'failed'
    assert clock.last_tick == last['cue_tick'] + P + 1
    assert not all(clock.armed)
    assert score(evidence(ticks, clock)).value == 27.


def test_timeout_does_not_complete_before_all_ten_cues_have_been_given():
    from mind2act_eval.tasks.DR2.scene import ButtonScene
    scene = object.__new__(ButtonScene)
    scene.clock = EventClock(SEQ, 3, 2, 2)
    for tick in range(1, 100):
        scene.clock.advance(tick, *UP)
        resolved = [r for r in scene.clock.records if r['status'] == 'failed']
        if len(resolved) < len(SEQ):
            assert not scene.events_complete()
        else:
            assert scene.events_complete() and scene.clock.current is None
            assert scene.clock.events[-1]['tick'] == tick
            assert scene.clock.events[-1]['kind'] == 'fail'
            break
    assert len(resolved) == 10
