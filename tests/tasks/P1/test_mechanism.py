import numpy as np
import pytest
from mind2act_eval.tasks.P1.mechanism.cup_geometry import outside_cup_bounds, hull_points
from mind2act_eval.tasks.P1.mechanism.occupancy import OccupancyLedger, SAMPLE_DT
from mind2act_eval.tasks.P1.mechanism.orders import OrderClock

ORDERS = [dict(at_seconds=1.0, seconds=2.0, delta={"cupA": {"apple": 2}}),
          dict(at_seconds=5.0, seconds=2.0, delta={"cupA": {"apple": 1}, "cupB": {"pear": 1}})]


def test_order_publication_and_demand_accumulate():
    c = OrderClock(ORDERS)
    screens, published = {}, {}
    for t in range(1000):
        screens[t] = c.step(t)
        published[t] = c.all_published(t)
    assert screens[120] == "hard_order0.png" and screens[360] == "hard_idle.png"
    assert screens[600] == "hard_order1.png" and screens[840] == "hard_all_sent.png"
    assert c.published_demand() == {"cupA/apple": 3, "cupB/pear": 1}
    assert not published[599] and published[600]
    assert published[839] and published[840]
    assert [e["index"] for e in c.events] == [0, 1]


def test_bounds_early_exit_is_conservative():
    identity = [1, 0, 0, 0, 0, 0, 0]
    hull = np.array([[.01, 0, 0], [-.01, 0, 0], [0, .01, 0], [0, -.01, 0]])
    bounds = [[-.05, -.05, 0], [.05, .05, .1]]
    inside = hull_points(hull, [0, 0, .02, *identity[:4]])
    assert not outside_cup_bounds(inside, bounds)
    moved = hull_points(hull, [.2, 0, .02, 1, 0, 0, 0])
    assert outside_cup_bounds(moved, bounds)


def obj(cup, supported=True, released=True, pos=(0, 0, 0)):
    return dict(occupied_containers=[cup] if cup else [], position=pos, kind="apple",
                inside=bool(cup), support=supported, released=released)


def test_ledger_tracks_entry_tick_and_resets_on_change_of_residence():
    led = OccupancyLedger(["f0", "f1"], SAMPLE_DT)
    for t in range(1, 200):
        led.observe(t, {"f0": obj("cupA"), "f1": obj(None)}, published_demand={}, all_orders_published=False)
    snap = led.snapshot()
    assert [o["object_id"] for o in snap["occupancy"]] == ["f0"] and snap["occupancy"][0]["entered_tick"] == 1
    assert snap["occupancy"][0]["stable_seconds"] >= 1.0 - 1e-9 and snap["occupancy_complete"]
    led.observe(200, {"f0": obj("cupB"), "f1": obj(None)}, published_demand={}, all_orders_published=False)
    assert led.snapshot()["occupancy"][0]["entered_tick"] == 200


def test_two_containers_at_once_is_flagged_ambiguous():
    led = OccupancyLedger(["f0"], SAMPLE_DT)
    row = dict(obj("cupA"), occupied_containers=["cupA", "cupB"])
    led.observe(1, {"f0": row}, published_demand={}, all_orders_published=False)
    assert led.snapshot()["occupancy_complete"] is False


def test_every_food_must_be_sampled():
    led = OccupancyLedger(["f0", "f1"], SAMPLE_DT)
    with pytest.raises(ValueError):
        led.observe(1, {"f0": obj(None)}, published_demand={}, all_orders_published=False)


@pytest.mark.parametrize('cutoff,published,expected_score', [(119,False,0),(120,True,100),(240,True,100)])
def test_published_order_can_be_fulfilled_while_its_card_remains_visible(cutoff,published,expected_score):
    from mind2act_eval.contracts import NativeEvidence
    from mind2act_eval.tasks.P1.scorer import score
    clock=OrderClock([dict(at_seconds=1.,seconds=5.,delta={'cupA':{'apple':4}})])
    ledger=OccupancyLedger(['f0','f1','f2','f3'],SAMPLE_DT)
    objects={ident:obj('cupA') for ident in ('f0','f1','f2','f3')}
    screens={}
    for tick in range(cutoff+1):
        screens[tick]=clock.step(tick)
        ledger.observe(tick,objects,published_demand=clock.published_demand(),
                       all_orders_published=clock.all_published(tick))
    measured=ledger.snapshot()
    result=score(NativeEvidence('P1','synthetic-publication',cutoff,
                               dict(measured,difficulty='easy')))
    assert measured['all_orders_published'] is published
    assert result.value == expected_score
    assert result.components['success'] is published
    assert clock.last_end == 720
    if published:
        assert measured['published_demand'] == {'cupA/apple':4}
        assert screens[120] == 'hard_order0.png'
        assert all(s != 'hard_all_sent.png' for s in screens.values())
