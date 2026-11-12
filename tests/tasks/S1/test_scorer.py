import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.S1.scorer import score


def block(i, good=True, **over):
    row = dict(object_id="/World/Block%d" % i, inside=good, support=good, released=good, stable_seconds=1.0,
               position_range_m=0.001, original_entity_slot_match=good, placed_after_program_finished=True)
    row.update(over)
    return row


def ev(blocks, early=False, finished=True):
    return NativeEvidence("S1", "S__easy__1", 5000,
                          {"early_placement": early, "rotation_program_finished": finished, "blocks": blocks})


@pytest.mark.parametrize("good,expected", [(4, 100), (3, 15), (2, 7.5), (1, 0), (0, 0)])
def test_count_tiers(good, expected):
    s = score(ev([block(i, i < good) for i in range(4)]))
    assert s.value == expected and s.components["credited_units"] == good
    assert s.components["success"] is (good == 4)


def test_early_placement_does_not_override_perfect_terminal_state():
    s = score(ev([block(i) for i in range(4)], early=True))
    assert s.value == 100 and s.components["success"] is True


def test_no_final_stability_wait():
    s = score(ev([block(0, stable_seconds=0.99)] + [block(i) for i in range(1, 4)]))
    assert s.value == 100


def test_no_final_drift_gate():
    s = score(ev([block(0, position_range_m=0.003)] + [block(i) for i in range(1, 4)]))
    assert s.value == 100


def test_robot_touch_does_not_disqualify_real_support():
    s = score(ev([block(0, released=False)] + [block(i) for i in range(1, 4)]))
    assert s.value == 100 and s.components["credited_units"] == 4


def test_terminal_slot_identity_is_independent_of_placement_time():
    s = score(ev([block(0, placed_after_program_finished=False)] + [block(i) for i in range(1, 4)]))
    assert s.value == 100


def test_terminal_slot_identity_does_not_wait_for_program_finish():
    assert score(ev([block(i) for i in range(4)], finished=False)).value == 100


def test_requires_four_unique_blocks():
    with pytest.raises(ValueError):
        score(ev([block(i) for i in range(3)]))
    with pytest.raises(ValueError):
        score(ev([block(0)] * 4))


def test_non_boolean_flag_is_invalid_evidence():
    with pytest.raises(ValueError):
        score(ev([block(0, released=1)] + [block(i) for i in range(1, 4)]))


def test_unknown_evidence_field_rejected():
    e = NativeEvidence("S1", "x", 1, {"difficulty":"easy", "early_placement": False, "rotation_program_finished": True,
                                      "blocks": [block(i) for i in range(4)], "carousel_touched": True})
    with pytest.raises(ValueError):
        score(e)


def test_carousel_contact_carries_no_penalty_and_expert_label_ignored():
    blocks = [block(i, phase="expert_release") for i in range(4)]
    assert score(ev(blocks)).value == 100


def test_timeline_used_is_recorded_in_score_components():
    e = NativeEvidence("S1", "x", 1, {"difficulty":"easy", "early_placement": False, "rotation_program_finished": True,
                                      "blocks": [block(i) for i in range(4)],
                                      "timeline": {"cue_seconds": 1.0, "cue_gap_s": 0.0}})
    assert score(e).components["timeline"] == {"cue_seconds": 1.0, "cue_gap_s": 0.0}


def test_final_original_slots_count_despite_early_held_unstable_placement():
    rows = [block(i, released=False, stable_seconds=0., position_range_m=.02,
                  placed_after_program_finished=False) for i in range(4)]
    assert score(ev(rows, early=True, finished=False)).value == 100
