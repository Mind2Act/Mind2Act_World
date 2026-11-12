from copy import deepcopy
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP10.evidence import ShotEvidence
from mind2act_eval.tasks.CP10.protocol import PROTOCOL_ID, RETENTION, validate
from mind2act_eval.tasks.CP10.retention import BallBounds, ContinuousRetention, contained_net
from mind2act_eval.tasks.CP10.scorer import score

CONFIG = dict(support_tolerance=.001, arena=dict(court_size=[2., 2.], court_center=[0., 0.],
    collector_length_m=dict(near=.5, far=.5), surface_z=1., net_depth_m=.2,
    net_settle_ticks=60, per_shot_timeout_ticks=2400))


def schedule(level):
    white, red = {"easy": (3, 2), "medium": (4, 4), "hard": (6, 6)}[level]
    return ("red", "white", *("white",) * white, *("red",) * red)


def ledger(level):
    return ShotEvidence(level, schedule(level), CONFIG, dict(RETENTION), PROTOCOL_ID)


def ball(net):
    y = -1.25 if net == "near" else 1.25 if net == "far" else 0.
    z = .9 if net is not None else 1.1
    return BallBounds((0., y, z), (-.02, y - .02, z - .02), (.02, y + .02, z + .02), True, (net,) if net else ())


def collect(w, net, contact):
    index = len(w.shots)
    tick = 0 if w.last_tick is None else w.last_tick + 1
    w.sample(tick, ball(net), emitted=index, paddle_contact=contact)


def correct(w):
    color = w.colors[len(w.shots)]
    collect(w, "far" if color == "white" else "near", color == "white")


def evaluate(w, level, *, termination="policy_ended", normal=True):
    return score(NativeEvidence("CP10", "synthetic-cpu", w.last_tick, dict(
        difficulty=level, collection_protocol=dict(RETENTION), rule_version=PROTOCOL_ID,
        normal_termination=normal, cp10=w.snapshot(termination))))


@pytest.mark.parametrize("level,n", [("easy", 3), ("medium", 4), ("hard", 6)])
def test_full_physical_success_uses_fixed_tier_denominator(level, n):
    w = ledger(level)
    for _ in w.colors:
        correct(w)
    s = evaluate(w, level, termination="completed")
    assert s.value == 100
    assert s.components["completion"] == {"k": n, "n": n, "ratio": 1}


def test_authoritative_medium_two_white_then_wrong_is_five():
    w = ledger("medium")
    correct(w); correct(w)  # Practice.
    correct(w); correct(w)
    collect(w, "near", False)
    assert evaluate(w, "medium").value == 5
    assert w.failure is None  # Wrong result does not stop the shot machine.
    correct(w)
    assert evaluate(w, "medium").value == 5  # Later success cannot repair prefix.


def test_formal_timeout_preserves_prior_white_and_sequence_continues():
    w=ledger('hard')
    correct(w);correct(w);correct(w)
    tick=w.last_tick+1
    w.sample(tick,ball(None),emitted=3)
    for t in range(tick+1,tick+2402):w.sample(t,ball(None))
    assert w.failure is None and len(w.shots)==4
    assert evaluate(w,'hard').value==pytest.approx(7/6)
    correct(w)
    assert evaluate(w,'hard').value==pytest.approx(7/6)


def test_authoritative_hard_six_white_then_final_wrong_red_is_seven():
    w = ledger("hard")
    for _ in w.colors[:-1]:
        correct(w)
    collect(w, "near", True)
    assert evaluate(w, "hard", termination="completed").value == 7


def test_wrong_practice_does_not_affect_success():
    w = ledger("easy")
    collect(w, "far", True)  # Wrong practice red.
    collect(w, "near", False)  # Wrong practice white.
    while len(w.shots) < len(w.colors):
        correct(w)
    assert evaluate(w, "easy", termination="completed").value == 100


def test_doing_nothing_earns_zero_even_if_reds_collect_correctly():
    w = ledger("easy")
    for _ in w.colors:
        collect(w, "near", False)
    assert evaluate(w, "easy", termination="completed").value == 0


def test_wrong_red_freezes_white_prefix_too():
    colors = ("white", "red", "white", "red", "white", "red", "white")
    w = ShotEvidence("easy", colors, CONFIG, dict(RETENTION), PROTOCOL_ID)
    correct(w); correct(w); correct(w)
    collect(w, "far", False)
    while len(w.shots) < len(w.colors):
        correct(w)
    assert evaluate(w, "easy", termination="completed").value == 5


def test_single_actual_supported_net_sample_is_collection():
    w=ledger('easy')
    w.sample(0,ball('near'),emitted=0,paddle_contact=True)
    assert len(w.shots)==1 and w.shots[0]['paddle_contact']


def test_collection_has_no_extra_dwell():
    r=ContinuousRetention()
    assert r.sample(0,'near')=='near'
    assert r.sample(1,None) is None
    assert r.sample(2,'far')=='far'


def test_next_ball_cannot_launch_before_previous_resolves():
    w=ledger('easy')
    w.sample(0,ball(None),emitted=0)
    with pytest.raises(ValueError,match='precedes collection'):w.sample(1,ball(None),emitted=1)


def test_practice_timeout_is_strict_2401_and_not_global_failure():
    w=ledger('easy')
    w.sample(0,ball(None),emitted=0)
    for tick in range(1,2401):w.sample(tick,ball(None))
    assert not w.shots
    w.sample(2401,ball(None))
    assert w.failure is None and len(w.shots)==1 and w.shots[0]['net'] is None


@pytest.mark.parametrize("flag,reason", [
    ("robot_contact", "robot_touched_ball"),
    ("unreleased_ball_moved", "unreleased_ball_moved"),
])
def test_practice_contact_and_waiting_drift_do_not_end_sequence(flag, reason):
    w=ledger('easy')
    w.sample(0,**{flag:True})
    assert w.failure is None
    w.sample(1)


def test_real_support_and_center_replace_hidden_height_bounds():
    body=BallBounds((0.,-1.25,1.1),(-.02,-1.27,.7),(.02,-1.23,1.5),True,('near',))
    assert contained_net(body,CONFIG)=='near'
    unsupported=BallBounds(body.center,body.lower,body.upper,True,())
    assert contained_net(unsupported,CONFIG) is None


@pytest.mark.parametrize("change", [None, {"consecutive_ticks": 30}, {"linear_speed_required": True},
                                   {"support_chain_required": True}, {"physics_hz": 120.0}])
def test_training_protocol_is_required_exactly(change):
    value = None if change is None else dict(RETENTION, **change)
    with pytest.raises(ValueError, match="protocol"):
        validate(value, PROTOCOL_ID)


def test_protocol_and_rule_version_cannot_disagree():
    with pytest.raises(ValueError, match="version"):
        validate(dict(RETENTION), "old-default-support-chain")


def test_missing_ticks_or_initial_frame_are_invalid():
    for first in [1, 4]:
        with pytest.raises(ValueError, match="consecutive"):
            ledger("easy").sample(first)
    w = ledger("easy")
    w.sample(0)
    with pytest.raises(ValueError, match="consecutive"):
        w.sample(4)


def test_missing_or_inconsistent_receipts_are_invalid_not_zero():
    with pytest.raises(ValueError,match='incomplete'):score(NativeEvidence('CP10','cpu',0,dict(difficulty='easy')))
    w=ledger('easy');correct(w);e=w.snapshot()
    e['shots'][0]['retention_ticks']=-1
    with pytest.raises(ValueError,match='receipt'):
        score(NativeEvidence('CP10','cpu',w.last_tick,dict(difficulty='easy',rule_version=PROTOCOL_ID,collection_protocol=dict(RETENTION),normal_termination=True,cp10=e)))


def test_full_result_does_not_need_generic_normal_termination_flag():
    w=ledger('easy')
    for _ in w.colors:correct(w)
    assert evaluate(w,'easy',termination='completed',normal=False).value==100


def test_controller_labels_and_a_forged_correct_flag_do_not_score():
    w = ledger("easy")
    for _ in w.colors:
        collect(w, "near", False)
    e = w.snapshot("completed")
    e.update(phase="success", milestones=["complete"], action_index=999)
    for row in e["shots"]:
        row["correct"] = True
    s = score(NativeEvidence("CP10", "cpu", w.last_tick, dict(difficulty="easy",
        rule_version=PROTOCOL_ID, collection_protocol=dict(RETENTION), normal_termination=True, cp10=e)))
    assert s.value == 0


def test_practice_timeout_does_not_stop_formal_sequence():
    w=ledger('easy')
    w.sample(0,ball(None),emitted=0)
    for tick in range(1,2402):w.sample(tick,ball(None))
    assert len(w.shots)==1 and w.failure is None
    correct(w)
    while len(w.shots)<len(w.colors):correct(w)
    assert evaluate(w,'easy',termination='completed').value==100
