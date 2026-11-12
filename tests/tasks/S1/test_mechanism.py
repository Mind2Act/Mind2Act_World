import numpy as np
import pytest
from mind2act_eval.tasks.S1.mechanism.placement import CarouselLedger, PlacementWindow, SPlacementReleaseTracker
from mind2act_eval.tasks.S1.mechanism.schedule import schedule, table_angle, rotation_end

KEYS = ["/World/Block%d" % i for i in range(4)]


def obs(supported=False, released=True, inside=True, support=True, slot="s0", match=True, after=True):
    return dict(position_local=[0, 0, 0], inside=inside, support=support, released=released, slot_id=slot,
                carousel_supported=supported, original_entity_slot_match=match, placed_after_program_finished=after)


def run(ledger, n, program_finished, **kw):
    start = (ledger.last_tick or 0) + 1
    for t in range(start, start + n):
        ledger.observe(t, {k: obs(**kw) for k in KEYS}, program_finished=program_finished)


def test_window_needs_121_consecutive_samples_for_one_second():
    w = PlacementWindow()
    for t in range(1, 121):
        w.observe(t, [0, 0, 0], inside=True, support=True, released=True, residence="a")
    assert w.snapshot()["stable_seconds"] < 1.0
    w.observe(121, [0, 0, 0], inside=True, support=True, released=True, residence="a")
    assert w.snapshot()["stable_seconds"] == 1.0


def test_window_resets_on_release_loss_and_clock_gap_is_error():
    w = PlacementWindow()
    for t in range(1, 50):
        w.observe(t, [0, 0, 0], inside=True, support=True, released=True, residence="a")
    w.observe(50, [0, 0, 0], inside=True, support=True, released=False, residence="a")
    assert w.snapshot()["stable_seconds"] == 0
    with pytest.raises(ValueError):
        w.observe(52, [0, 0, 0], inside=True, support=True, released=True, residence="a")


def test_window_spread_is_pairwise_maximum():
    w = PlacementWindow()
    for t, x in enumerate([0, .002, 0, .004], 1):
        w.observe(t, [x, 0, 0], inside=True, support=True, released=True, residence="a")
    assert w.snapshot()["position_range_m"] == pytest.approx(.004)


def test_release_on_carousel_before_program_end_latches_early_placement():
    ledger = CarouselLedger(KEYS)
    run(ledger, 3, False, supported=True, released=True)
    run(ledger, 200, True, supported=False)
    assert ledger.snapshot()["early_placement"] is True


def test_holding_on_carousel_before_program_end_is_not_early():
    ledger = CarouselLedger(KEYS)
    run(ledger, 3, False, supported=True, released=False)
    assert ledger.snapshot()["early_placement"] is False


def test_visible_cue_placement_does_not_latch_a_later_formal_failure():
    ledger = CarouselLedger(KEYS)
    ledger.observe(1, {k: obs(supported=True) for k in KEYS},
                   program_finished=False, cue_visible=True)
    run(ledger, 200, True, supported=False)
    assert ledger.snapshot()['early_placement'] is False


def test_program_completion_cannot_reverse():
    ledger = CarouselLedger(KEYS)
    run(ledger, 1, True)
    with pytest.raises(ValueError):
        run(ledger, 1, False)


def test_release_tracker_counts_only_release_strictly_after_boundary():
    tr = SPlacementReleaseTracker({"a": "/World/Block0"})
    pair = ["/World/Block0/Collision", "/World/RobotLeft/Asset/link7/Collision"]
    tr.observe(99, [dict(colliders=pair, active=True)])
    tr.observe(100, [dict(colliders=pair, active=False)])
    assert tr.placed_after_stop("a", 100) is False and tr.placed_after_stop("a", 99) is True
    assert tr.placed_after_stop("a", None) is False


def test_schedule_phases_and_rotation_end():
    cfg = dict(cue_seconds=1.0, rotation_seconds=3.0, stop_seconds=.6)
    tables = {"T": dict(rotation_degrees=[-45, -90])}
    assert schedule(0, cfg, tables)[0] == "cue"
    assert schedule(121, cfg, tables)[0] == "rotation"
    end = rotation_end(1.0, 3.0, .6, 2)
    assert schedule(int(end * 120) + 1, cfg, tables)[0] == "execution"
    assert table_angle(end, 1.0, 3.0, .6, [-45, -90]) == pytest.approx(np.radians(-135))
