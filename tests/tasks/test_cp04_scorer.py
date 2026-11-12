import copy
import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP04.scorer import score

RECIPE = {"easy": ["chili", "onion"], "medium": ["chili", "onion"], "hard": ["garlic", "ginger", "chili"]}
WINDOW = {"easy": 4.0, "medium": 3.0, "hard": 2.0}


def good(difficulty):
    recipe = RECIPE[difficulty]
    prep = difficulty != "easy"
    pours = [dict(ingredient_id=r, tick=1000 + 100 * i, whole_portion=True, entered_pot=True, cap_removed=True, terminal_portion_contained=True)
             for i, r in enumerate(recipe)]
    last = pours[-1]["tick"]
    thermal = dict(events=[dict(mode="HEAT", tick=1500, temperature_c=25.0, display_temperature_c=25.0),
                           dict(mode="KEEP_WARM", tick=1500 + 120 * 30, temperature_c=85.0, display_temperature_c=85.0),
                           dict(mode="OFF", tick=1500 + 120 * 40, temperature_c=85.0, display_temperature_c=85.0)],
                   response_window_start_tick=1500 + 120 * 30 - 60, max_temperature_after_keep_warm_c=85.0,
                   history_complete=True, hold_confirmed_tick=1500 + 120 * 35, observed_through_tick=1500 + 120 * 41)
    ev = dict(tick_hz=120, recipe_hidden_tick=420, drawer_open_tick=600 if prep else None,
              pot_lid_release_tick=last + 50 if prep else None, ingredient_events=pours, thermal=thermal,
              returns=[dict(ingredient_id=r, jar_returned=True, jar_released=True, jar_stable=True,
                            cap_in_tray=True, cap_released=True, cap_stable=True) for r in recipe],
              controls_released=True, physical_flow_valid=True)
    return dict(difficulty=difficulty, recipe=recipe, normal_termination=True, cp04_evidence=ev)


def run(values):
    return score(NativeEvidence("CP04", "i", 1, values))


def comp(s):
    return s.components["completion"]


@pytest.mark.parametrize("d,n", [("easy", 4), ("medium", 6), ("hard", 7)])
def test_full_success_is_100(d, n):
    s = run(good(d))
    assert s.value == 100.0 and comp(s) == {"k": len(RECIPE[d]), "n": len(RECIPE[d]), "ratio": 1.0}


@pytest.mark.parametrize("d,n", [("easy", 4), ("medium", 6), ("hard", 7)])
def test_t0_physical_success_does_not_need_legacy_normal_termination_flag(d, n):
    v = good(d); v["normal_termination"] = False
    s = run(v)
    assert s.components["task_success"] is True and s.value == 100.0


def test_no_events_counts_only_drawer():
    v = good("hard"); e = v["cp04_evidence"]
    e["ingredient_events"] = []; e["thermal"]["events"] = []
    e["thermal"]["response_window_start_tick"] = None; e["thermal"]["max_temperature_after_keep_warm_c"] = None
    e["thermal"]["hold_confirmed_tick"] = None; e["pot_lid_release_tick"] = None
    s = run(v)
    assert comp(s)["k"] == 0 and s.value == 0


def test_wrong_pour_keeps_lcs_and_other_correct_steps():
    v = good("hard"); v["cp04_evidence"]["ingredient_events"][1]["ingredient_id"] = "chili"
    s = run(v)
    assert comp(s)["k"] == 2 and s.components["task_success"] is False


def test_correct_pour_after_wrong_one_counts_by_lcs():
    v = good("easy"); ev = v["cp04_evidence"]["ingredient_events"]
    ev[0]["ingredient_id"], ev[1]["ingredient_id"] = ev[1]["ingredient_id"], ev[0]["ingredient_id"]
    assert comp(run(v))["k"] == 1


def test_extra_pour_prevents_success_without_freezing_later_steps():
    v = good("easy"); e = v["cp04_evidence"]
    e["ingredient_events"].append(dict(ingredient_id="garlic", tick=1300, whole_portion=True,
                                       entered_pot=True, cap_removed=True, terminal_portion_contained=True))
    s = run(v)
    assert comp(s)["k"] == 2 and s.components["task_success"] is False


def test_drawer_after_pour_does_not_erase_independent_steps():
    v = good("medium"); v["cp04_evidence"]["drawer_open_tick"] = 5000
    assert comp(run(v))["k"] == 2


def test_premature_keep_warm_is_not_valid():
    v = good("hard"); v["cp04_evidence"]["thermal"]["events"][1]["temperature_c"] = 60.0
    s = run(v)
    assert comp(s)["k"] == 3 and s.components["task_success"] is False


def test_hard_with_two_ingredient_recipe_rejected():
    v = good("hard"); v["recipe"] = ["chili", "onion"]
    with pytest.raises(ValueError):
        run(v)


def test_non_120hz_evidence_rejected():
    v = good("easy"); v["cp04_evidence"]["tick_hz"] = 30
    with pytest.raises(ValueError):
        run(v)


@pytest.mark.parametrize('difficulty,index', [
    ('easy', 0), ('easy', 1), ('medium', 0), ('medium', 1),
    ('hard', 0), ('hard', 1), ('hard', 2),
])
@pytest.mark.parametrize('heat_started', [False, True])
def test_capped_pour_does_not_erase_later_correct_ingredients(difficulty, index, heat_started):
    values = good(difficulty)
    evidence = values['cp04_evidence']
    poured = evidence['ingredient_events'][index]
    poured['cap_removed'] = False
    if not heat_started:
        evidence['thermal']['events'] = []
        evidence['thermal']['response_window_start_tick'] = None
        evidence['thermal']['max_temperature_after_keep_warm_c'] = None
        evidence['thermal']['hold_confirmed_tick'] = None

    result = run(values)

    # Credit only the drawer (when present) and earlier, uncapped pours.
    # In Medium's first-pour case, the old scorer incorrectly credited 4/6.
    expected_prefix = len(RECIPE[difficulty])-1
    assert comp(result)['k'] == expected_prefix
    assert result.value == dict(easy=15,medium=10,hard=7)[difficulty] * expected_prefix / comp(result)['n']
    assert result.components['first_error_tick'] == poured['tick']
    assert result.components['task_success'] is False


def test_later_valid_pours_earn_partial_but_extra_invalid_event_prevents_success():
    values = good('medium')
    events = values['cp04_evidence']['ingredient_events']
    invalid = copy.deepcopy(events[0])
    invalid.update(tick=900, cap_removed=False)
    events.insert(0, invalid)

    result = run(values)

    assert comp(result)['k'] == 2
    assert result.components['first_error_tick'] == 900
    assert result.value == 10
    assert not result.components['task_success']


def test_final_lcs_keeps_later_correct_pours_and_independent_thermal_finish():
    v=good('easy');events=v['cp04_evidence']['ingredient_events']
    events[0]['ingredient_id'],events[1]['ingredient_id']=events[1]['ingredient_id'],events[0]['ingredient_id']
    s=run(v)
    assert comp(s)['k']==1  # Only the LCS ingredient earns partial credit
    assert not s.components['task_success']


def test_final_no_hold_upper_temperature_exact_freeze_or_release_gate():
    v=good('hard');e=v['cp04_evidence'];t=e['thermal']
    t['hold_confirmed_tick']=None;t['max_temperature_after_keep_warm_c']=130
    t['events'][1]['temperature_c']=110;t['events'][2]['temperature_c']=109
    t['events'][2]['display_temperature_c']=108
    t['events'][2]['tick']=t['events'][1]['tick']+1
    for r in e['returns']:
        for k in ('jar_released','jar_stable','cap_released','cap_stable'):r[k]=False
    e['controls_released']=False
    assert run(v).value==100


def test_portion_final_containment_belongs_to_ingredient_not_off_return():
    v=good('easy');e=v['cp04_evidence']
    for row in e['ingredient_events']:row['terminal_portion_contained']=False
    e['physical_flow_valid']=False
    result=run(v)
    assert result.components['independent_steps']==(False,False,True,True)
    assert result.value==0
