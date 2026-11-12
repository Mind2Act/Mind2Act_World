import pytest
from mind2act_eval.tasks.E1.evidence import Evidence, WINDOW_TICKS
from mind2act_eval.tasks.E1.scorer import score

GOAL, INNER, Z = [0.0, 0.0], [0.1, 0.1], (0.0, 0.05)
FLOOR = lambda p, rn=1: (p + "/x", "/Batch%d/Collision/submit_floor" % rn)


def item(path, identity, at_goal, touch=False, z=0.02, x=0.0):
    center = [x, 0.0, z] if at_goal else [0.5, 0.5, z]
    return dict(path=path, identity=identity, center=center, dimensions=[0.04, 0.04, 0.04], baseline_z=0.0,
                bilateral=True, touch_robot=touch, support=at_goal)


def row(tick, rn, items, phase="batch_execute", contacts=(), deadline=False):
    return dict(physics_tick=tick, round=rn, phase=phase, items=items, goal_xy=GOAL, inner_xy=INNER, z_bounds=list(Z),
                contacts=[dict(paths=list(p)) for p in contacts], deadline_reached=deadline)


def settle(ev, rn, items_fn, start, n, contacts=()):
    sealed = False
    for t in range(start, start + n):
        sealed = ev.sample(row(t, rn, items_fn(t), contacts=contacts))
    return sealed


def test_exactly_one_correct_supported_item_passes():
    ev = Evidence({1: "red"}, "easy")
    fn = lambda t: [item("/a", "red", True), item("/b", "blue", False)]
    assert not settle(ev, 1, fn, 1, WINDOW_TICKS - 1, [FLOOR("/a")])
    assert settle(ev, 1, fn, WINDOW_TICKS, 1, [FLOOR("/a")])
    r = ev.results[0]
    assert r["pass"] and r["submitted"] == ["/a"] and r["sealed_by"] == "stable_window"
    assert score(ev.native_evidence("E1", "i", 200)).value == 15  # 1 of N=2


def test_wrong_item_seals_as_failed_and_later_batches_continue():
    ev = Evidence({1: "red", 2: "blue"}, "easy")
    wrong = lambda t: [item("/a", "red", False), item("/b", "blue", True)]
    settle(ev, 1, wrong, 1, WINDOW_TICKS, [FLOOR("/b")])
    assert ev.results[0]["pass"] is False
    right = lambda t: [item("/c", "blue", True)]
    settle(ev, 2, right, WINDOW_TICKS + 1, WINDOW_TICKS, [FLOOR("/c", 2)])
    assert [r["pass"] for r in ev.results] == [False, True]


def test_extra_registered_item_makes_batch_invalid_even_if_removed():
    ev = Evidence({1: "red"}, "easy")
    extra = lambda t: [item("/a", "red", True), item("/b", "blue", t < 5)]
    settle(ev, 1, extra, 1, WINDOW_TICKS, [FLOOR("/a"), FLOOR("/b")])
    assert ev.results[0]["pass"] is False and ev.results[0]["submitted"] == ["/a", "/b"]


def test_supported_item_waits_for_release_and_full_stability_window():
    ev = Evidence({1: "red"}, "easy")
    held = lambda t: [item("/a", "red", True, touch=True)]
    assert not settle(ev, 1, held, 1, 180, [FLOOR("/a")])
    assert ev.registered[1] == set()
    released = lambda t: [item("/a", "red", True)]
    assert not settle(ev, 1, released, 181, WINDOW_TICKS - 1, [FLOOR("/a")])
    assert settle(ev, 1, released, 181 + WINDOW_TICKS - 1, 1, [FLOOR("/a")])
    assert ev.results[0]["pass"]


def test_motion_outside_stability_range_delays_submission():
    ev = Evidence({1: "red"}, "easy")
    drift = lambda t: [item("/a", "red", True, x=0.0 if t % 2 else 0.004)]
    assert not settle(ev, 1, drift, 1, 2 * WINDOW_TICKS, [FLOOR("/a")])
    assert ev.results == []


def test_no_submission_never_seals_the_batch():
    ev = Evidence({1: "red"}, "easy")
    assert not settle(ev, 1, lambda t: [item("/a", "red", False)], 1, 5000)
    assert ev.results == []


def test_sealed_batch_is_not_sealed_twice():
    ev = Evidence({1: "red"}, "easy")
    fn = lambda t: [item("/a", "red", True)]
    settle(ev, 1, fn, 1, WINDOW_TICKS, [FLOOR("/a")])
    assert not ev.sample(row(WINDOW_TICKS + 1, 1, fn(0), contacts=[FLOOR("/a")]))
    assert len(ev.results) == 1


def test_clock_gap_and_bad_geometry_are_errors():
    ev = Evidence({1: "red"}, "easy")
    ev.sample(row(1, 1, [item("/a", "red", False)]))
    with pytest.raises(ValueError):
        ev.sample(row(3, 1, [item("/a", "red", False)]))
    bad = item("/a", "red", True)
    bad["center"][0] = float("nan")
    with pytest.raises(ValueError):
        Evidence({1: "red"}, "easy").sample(row(1, 1, [bad]))
