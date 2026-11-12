import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.P1.scorer import score


def row(i, cup="cupA", kind="apple", tick=None, good=True, **over):
    r = dict(object_id="food%d" % i, container=cup, kind=kind, entered_tick=i if tick is None else tick,
             inside=good, support=good, released=good, stable_seconds=1.0, position_range_m=0.001)
    r.update(over)
    return r


def ev(rows, demand, difficulty="easy", published=True, complete=True, ambiguous=()):
    return NativeEvidence("P1", "P__%s__1" % difficulty, 9000, dict(
        difficulty=difficulty, occupancy=rows, occupancy_complete=complete, ambiguous_membership=list(ambiguous),
        published_demand=demand, all_orders_published=published))


D4 = {"cupA/apple": 2, "cupB/pear": 2}


def exact4():
    return [row(0), row(1), row(2, "cupB", "pear"), row(3, "cupB", "pear")]


def test_exact_fulfilment_is_100_and_success():
    s = score(ev(exact4(), D4))
    assert s.value == 100 and s.components["success"] and s.components["credited_units"] == 4


def test_one_missing_item_is_10():
    assert score(ev(exact4()[:3], D4)).value == 15


def test_two_defects_is_5_and_three_is_0():
    assert score(ev(exact4()[:2], D4)).value == 7.5
    assert score(ev(exact4()[:1], D4)).value == 0


def test_one_extra_item_after_exact_is_10_and_failure():
    s = score(ev(exact4() + [row(4)], D4))
    assert s.value == 15 and not s.components["success"]


def test_wrong_material_substituting_one_is_5():
    rows = [row(0), row(1), row(2, "cupB", "pear"), row(3, "cupB", "apple")]
    s = score(ev(rows, D4))
    assert s.value == 7.5 and s.components["credited_units"] == 3


def test_all_final_candidates_are_counted_with_wrong_items():
    rows = [row(0, "cupB", "apple", tick=0), row(1, tick=1), row(2, tick=2), row(3, "cupB", "pear", tick=3),
            row(9, "cupB", "pear", tick=99)]
    s = score(ev(rows, D4))
    assert s.components["credited_units"] == 4 and s.components["wrong"] == 1
    assert s.components["extra"] == 0 and s.value == 15


def test_supported_unstable_or_touched_item_is_credited():
    rows = exact4()
    rows[0] = row(0, stable_seconds=0.5)
    assert score(ev(rows, D4)).components["credited_units"] == 4
    rows[0] = row(0, released=False)
    assert score(ev(rows, D4)).components["credited_units"] == 4


def test_full_score_requires_all_orders_to_have_been_published():
    s = score(ev(exact4(), D4, published=False))
    assert s.components["success"] is False and s.value == 0


def test_incomplete_or_ambiguous_evidence_is_invalid():
    with pytest.raises(ValueError):
        score(ev(exact4(), D4, complete=False))
    with pytest.raises(ValueError):
        score(ev(exact4(), D4, ambiguous=[{"object_id": "food0"}]))


def test_duplicate_identity_is_invalid():
    with pytest.raises(ValueError):
        score(ev([row(0), row(0)], D4))


def test_demand_over_denominator_is_invalid():
    with pytest.raises(ValueError):
        score(ev(exact4(), {"cupA/apple": 9}))


@pytest.mark.parametrize("difficulty,n", [("easy", 4), ("medium", 5), ("hard", 7)])
def test_denominator_per_difficulty(difficulty, n):
    demand = {"cupA/apple": n}
    rows = [row(i) for i in range(n)]
    s = score(ev(rows, demand, difficulty))
    assert s.value == 100 and s.components["total_units"] == n


def test_expert_labels_do_not_matter():
    rows = [dict(r, phase="expert_release", action_index=3) for r in exact4()]
    assert score(ev(rows, D4)).value == 100


def test_all_terminal_foods_count_without_entry_order_release_or_wait():
    foods = exact4() + [row(9, 'cupC', 'wrong', tick=0)]
    foods[0].update(released=False, stable_seconds=0., position_range_m=.02)
    result = score(ev(foods, D4))
    assert result.value == 15 and result.components['credited_units'] == 4


def test_missing_wrong_and_extra_each_contribute_one_defect():
    result = score(ev(exact4()[:3] + [row(8, 'cupC', 'wrong')], D4))
    assert result.value == 7.5 and result.components['defects'] == 2
