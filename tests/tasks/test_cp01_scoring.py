import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.tasks.CP01 import evidence as ev
from mind2act_eval.tasks.CP01.scorer import score


def native(difficulty, satisfied, wrong=0, success=False):
    n = {"easy": 3, "medium": 8, "hard": 12}[difficulty]
    return NativeEvidence("CP01", "i", 1, dict(
        credited_foods=min(satisfied,{'easy': 3, 'medium': 6, 'hard': 9}[difficulty]), scoring_version=SCORING_VERSION,
        difficulty=difficulty, required_steps=n, steps_satisfied=satisfied, wrong_or_extra_items=wrong,
        task_success=success))


@pytest.mark.parametrize("difficulty,n", [("easy", 3), ("medium", 8), ("hard", 12)])
def test_success_is_100_and_partial_uses_core_cap(difficulty, n):
    core_n = {'easy': 3, 'medium': 6, 'hard': 9}[difficulty]
    cap = dict(easy=15,medium=10,hard=7)[difficulty]
    assert score(native(difficulty, n, success=True)).value == 100
    assert score(native(difficulty, 0)).value == 0
    assert score(native(difficulty, 1)).value == pytest.approx(cap / core_n)
    assert score(native(difficulty, n - 1)).value <= cap
    assert score(native(difficulty, n)).value == cap
    assert score(native(difficulty, n)).components["completion"] == {"k": core_n, "n": core_n, "ratio": 1.0}


def test_wrong_or_extra_items_subtract_and_floor_at_zero():
    assert score(native("medium", 4, wrong=1)).value == pytest.approx(10 * 3 / 6)
    assert score(native("easy", 1, wrong=3)).value == 0


def test_wrong_denominator_and_false_success_rejected():
    bad = NativeEvidence("CP01", "i", 1, dict(difficulty="hard", required_steps=9, steps_satisfied=0,
                                             wrong_or_extra_items=0, task_success=False))
    with pytest.raises(ValueError):
        score(bad)
    with pytest.raises(ValueError):
        score(native("easy", 2, success=True))
    with pytest.raises(ValueError):
        score(native("easy", 9))


def test_scorer_does_not_need_phase_labels():
    assert "phase" not in score(native("easy", 3)).components


def test_rotation_matches_scipy_convention():
    scipy_rot = pytest.importorskip("scipy.spatial.transform").Rotation
    q = np.array([0.9, 0.1, -0.3, 0.2])
    expected = scipy_rot.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    assert np.allclose(ev.rotation({"quaternion": q}), expected)


def _row(tick, count, trays, **kw):
    body = dict(position=[0, 0, 0.8], quaternion=[1, 0, 0, 0], local_center=[0, 0, 0],
                half_extents=[0.01, 0.01, 0.01], bilateral=False, touch_table=True,
                touch_robot=False, touch_trays=[], tilt_deg=0.0)
    bodies = [dict(body, index=i) for i in range(count + trays)]
    return dict(physics_tick=tick, bodies=bodies, phase="execution", **kw)


def test_clock_gap_is_an_error():
    setup = dict(difficulty="easy", items=[{"kind": "a"}], orders=[["a"]], tray_centers=[0.0],
                 tray_center_y=0.0, tray_spec={"tray_size_m": [0.36, 0.14]})
    e = ev.Evidence(setup, 30.0)
    e.sample(_row(1, 1, 1))
    with pytest.raises(ValueError):
        e.sample(_row(3, 1, 1))


class _Done:
    complete = True


def _graded(difficulty, *, ready, tray_first, food_first, inside, kinds, orders):
    setup = dict(difficulty=difficulty, items=[{"kind": k} for k in kinds], orders=orders,
                 tray_centers=[0.0] * len(orders), tray_center_y=0.0, tray_spec={"tray_size_m": [0.36, 0.14]})
    e = ev.Evidence(setup, 30.0)
    e.foods = [_Done() for _ in kinds]
    e.ready, e.inside, e.phase = list(ready), inside, "execution"
    e.tray_first_ready, e.food_first_complete = tray_first, food_first
    return e.grade()


def test_food_credit_uses_terminal_tray_readiness_in_either_order():
    kw = dict(kinds=["a", "b"], orders=[["a", "b"], []], ready=[True, False], inside=[[0, 1], []])
    late = _graded("medium", tray_first={0: 500}, food_first={0: 100, 1: 120}, **kw)
    assert late["credited_foods"] == 2 and late["tray_steps"] == 1 and late["steps_satisfied"] == 3
    ok = _graded("medium", tray_first={0: 50}, food_first={0: 100, 1: 120}, **kw)
    assert ok["credited_foods"] == 2 and ok["tray_steps"] == 1 and ok["steps_satisfied"] == 3


def test_wrong_and_extra_items_are_counted_from_the_final_state():
    g = _graded("easy", kinds=["a", "b", "c", "a"], orders=[["a", "b", "c"]], ready=[True],
                inside=[[0, 1, 2, 3]], tray_first={}, food_first={0: 1, 1: 1, 2: 1, 3: 1})
    assert g["wrong_or_extra_items"] == 1 and g["steps_satisfied"] == 3 and g["k"] == 2
    assert g["required_steps"] == 3
