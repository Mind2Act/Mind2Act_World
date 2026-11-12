import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP05.scorer import score

NOTES = {"easy": 8, "medium": 8, "hard": 12}
N = NOTES


def good(d):
    n = NOTES[d]
    target = [11 + i % 7 for i in range(n)]
    tool = "single" if d == "easy" else "left"
    events = [dict(index=i, phase="replay", key=k, tool_id=tool, nib_contact=True,
                   press_tick=2000 + 100 * i, release_tick=2050 + 100 * i) for i, k in enumerate(target)]
    tools = [dict(tool_id="single", returned=True, released=True)] if d == "easy" else \
        [dict(tool_id=t, returned=True, released=True) for t in ("left", "right")]
    ev = dict(triggered_play=True, demonstration_completed=True, replay_events=events, all_keys_released=True,
              tools=tools, volume="MAX" if d == "hard" else "MID", failures=[], arms_withdrawn=True,
              play_tick=500, demonstration_completed_tick=1500,
              volume_max_released_tick=300 if d == "hard" else None)
    return dict(difficulty=d, target_keys=target, normal_termination=True, cp05_evidence=ev)


def run(v):
    return score(NativeEvidence("CP05", "i", 1, v))


@pytest.mark.parametrize("d", ["easy", "medium", "hard"])
def test_success_100(d):
    s = run(good(d))
    assert s.value == 100.0 and s.components["completion"] == {"k": N[d], "n": N[d], "ratio": 1.0}


def comp(s):
    return s.components["completion"]


@pytest.mark.parametrize("d", ["easy", "medium", "hard"])
def test_t0_success_is_independent_of_legacy_termination_flag(d):
    v = good(d); v["normal_termination"] = False
    s = run(v)
    assert s.components["task_success"] is True and s.value == 100.0


def test_wrong_key_keeps_later_lcs_and_return_credit():
    v = good("easy"); v["cp05_evidence"]["replay_events"][3]["key"] = 28
    s = run(v)
    assert comp(s)["k"] == 7 and s.value == 15*7/8


def test_extra_press_prevents_full_success_without_erasing_return():
    v = good("easy"); e = v["cp05_evidence"]
    e["replay_events"].append(dict(index=8, phase="replay", key=11, tool_id="single", nib_contact=True,
                                   press_tick=5000, release_tick=5050))
    s = run(v)
    assert comp(s)["k"] == 8 and s.components["task_success"] is False


def test_notes_before_demonstration_end_do_not_count():
    v = good("medium"); v["cp05_evidence"]["demonstration_completed_tick"] = 9999
    assert comp(run(v))["k"] == 0  # Setup and rack return earn no credit.


def test_hard_max_after_play_loses_only_max_step():
    v = good("hard"); v["cp05_evidence"]["volume_max_released_tick"] = 800
    s = run(v)
    assert comp(s)["k"] == 12


def test_nib_identity_does_not_reject_actual_depth_event():
    v = good("easy"); v["cp05_evidence"]["replay_events"][2]["nib_contact"] = False
    assert run(v).value == 100


def test_supported_rack_return_does_not_require_robot_release():
    v = good("medium"); v["cp05_evidence"]["tools"][0]["released"] = False
    assert run(v).value == 100


def test_wrong_target_length_rejected():
    v = good("hard"); v["target_keys"] = v["target_keys"][:8]
    with pytest.raises(ValueError):
        run(v)


def test_final_wrong_note_uses_lcs_and_does_not_cancel_independent_return():
    v=good('easy');v['cp05_evidence']['replay_events'][3]['key']=28
    assert comp(run(v))['k']==7  # Seven matching core notes only.


def test_final_depth_notes_need_no_identity_release_or_withdrawal():
    v=good('hard');e=v['cp05_evidence'];e['arms_withdrawn']=False;e['all_keys_released']=False
    for r in e['replay_events']:
        r.update(tool_id='other',nib_contact=False,release_tick=None)
    for t in e['tools']:t['released']=False
    e['volume_max_tick']=300;e['volume_max_released_tick']=None
    e['failures']=[dict(step=100,reason='required_mallet_not_held_during_demonstration')]
    assert run(v).value==100
