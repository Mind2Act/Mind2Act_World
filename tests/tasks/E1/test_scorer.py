import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.E1.scorer import score

N = {"easy": 2, "medium": 4, "hard": 6}


def ev(difficulty, rounds):
    rows = [{"round": i, "pass": p} for i, p in rounds]
    return NativeEvidence("E1", "E__%s__1" % difficulty, 100, {"difficulty": difficulty, "rounds": rows})


def passes(difficulty, flags):
    return ev(difficulty, [(i + 1, f) for i, f in enumerate(flags)])


@pytest.mark.parametrize("difficulty,table", [
    ("easy", {0: 0, 1: 15, 2: 100}),
    ("medium", {0: 0, 1: 5.5, 2: 11, 3: 16.5, 4: 100}),
    ("hard", {0: 0, 1: 2.5, 2: 5, 3: 7.5, 4: 10, 5: 12.5, 6: 100}),
])
def test_snapshot_tier_tables(difficulty, table):
    n = N[difficulty]
    for c, expected in table.items():
        flags = [True] * c + [False] * (n - c)
        s = score(passes(difficulty, flags))
        assert s.value == expected and s.components["credited_units"] == c and s.components["success"] is (c == n)


def test_errors_do_not_stop_later_batches_and_credit_is_not_a_prefix():
    s = score(passes("medium", [False, True, True, True]))
    assert s.value == 16.5 and s.components["credited_units"] == 3


def test_unsealed_batches_contribute_zero_with_fixed_n():
    s = score(passes("hard", [True, True, True, True, True]))
    assert s.value == 12.5 and s.components["total_units"] == 6 and s.components["sealed_batches"] == 5


def test_invalid_indices_and_flags():
    for bad in ([(0, True)], [(3, True)], [(1, True), (1, False)]):
        with pytest.raises(ValueError):
            score(ev("easy", bad))
    with pytest.raises(ValueError):
        score(ev("easy", [(1, 1)]))
    with pytest.raises(ValueError):
        score(ev("extreme", []))


def test_expert_labels_ignored():
    e = NativeEvidence("E1", "x", 1, {"difficulty": "easy", "consumer_kind": "model",
                                      "rounds": [{"round": 1, "pass": True, "phase": "expert"}]})
    assert score(e).value == 15
