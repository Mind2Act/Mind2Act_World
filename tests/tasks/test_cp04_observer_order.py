"""Native sample order reaches scoring without invented sub-sample timestamps."""
from mind2act_eval.contracts import NativeEvidence
import pytest
from mind2act_eval.tasks.CP04.mechanism import observer as observer_module
from mind2act_eval.tasks.CP04.mechanism.observer import IDS, Observer
from mind2act_eval.tasks.CP04.mechanism.thermal_profile497 import VERSION
from mind2act_eval.tasks.CP04.scorer import score


def measured(poured):
    jars = {name: dict(body=[.4, .2, .8], cap=[.4, .3, .85],
                      body_speed=0., body_angular_speed=0.,
                      cap_speed=0., cap_angular_speed=0., detached=True,
                      body_released=True, cap_released=True,
                      in_drawer=True, in_tray=True) for name in IDS}
    return dict(drawer=[.4, -.36, .8], jars=jars,
                food={name: [[.106139, .024681, .86]] * 3 if name in poured else []
                      for name in IDS},
                food_in_pot={name:[True]*3 if name in poured else [] for name in IDS},
                pot_lid_supported=False, pot_lid_released=False,
                pot_lid_stable=False, controls_released=True)


def observer(recipe):
    result = Observer(dict(recipe=recipe, thermal_profile_version=VERSION),
                      {name: [.4, .2, .8] for name in IDS})
    result.hidden = 420
    result.thermal_sample(900, dict(mode='OFF', temperature=25.,
                                   window_entered=None, valid_switch=False))
    return result


def scored(result, tier='easy'):
    return score(NativeEvidence('CP04', 'synthetic-order', result.tick * 8,
                 dict(difficulty=tier, recipe=result.recipe, normal_termination=True,
                      cp04_evidence=result.evidence())))


def test_same_sample_never_invents_a_future_ingredient_tick():
    result = observer(['chili', 'onion'])
    result.sample(1000, measured({'chili', 'onion'}))
    assert [row['tick'] for row in result.evidence()['ingredient_events']] == [1000, 1000]


def test_distinct_sample_order_comes_from_measurements_not_identity_iteration():
    result = observer(['onion', 'chili'])
    result.sample(1000, measured({'onion'}))
    result.sample(1001, measured({'onion', 'chili'}))
    assert [(row['ingredient_id'], row['tick']) for row in result.evidence()['ingredient_events']] == [
        ('onion', 1000), ('chili', 1001)]
    assert scored(result).components['completion']['k'] == 2


def actual_thermal_history(*, early_heat=False, early_off=False, first_pour_tick=1000,
                           simultaneous_pours=False, extra_after_warm=False,
                           dense_thermal_samples=False):
    """Feed real ThermalState snapshots and 30 Hz measurements, not thermal dicts."""
    from mind2act_eval.tasks.CP04.mechanism.thermal import ThermalState
    result = Observer(dict(recipe=['chili', 'onion'], thermal_profile_version=VERSION),
                      {name: [.4, .2, .8] for name in IDS})
    model = ThermalState('easy', thermal_profile_version=VERSION)
    dense = []
    warm = 2409 if early_heat else (2709 if early_off else 2461)
    off = warm + 125
    for tick in range(1, off + 65):
        if tick == 420:
            result.hidden = tick
        if early_heat and 700 <= tick < 800:
            powered, knob = True, 'HEAT'
        elif early_off and 1500 <= tick < 1600:
            powered, knob = True, 'HEAT'
        elif (1800 if early_off else 1500) <= tick < warm:
            powered, knob = True, 'HEAT'
        elif warm <= tick < off:
            powered, knob = True, 'KEEP WARM'
        else:
            powered, knob = False, 'HEAT'
        snapshot = model.step(1/120, powered=powered, knob_mode=knob)
        result.thermal_sample(tick, snapshot)
        dense.append(dict(mode=snapshot['mode'].replace(' ', '_'), tick=tick,
                          temperature_c=snapshot['temperature'],
                          display_temperature_c=snapshot['temperature']))
        if tick % 4 == 0:
            poured = set()
            if tick >= first_pour_tick:
                poured.add('chili')
            if tick >= (first_pour_tick if simultaneous_pours else 1100):
                poured.add('onion')
            if extra_after_warm and tick >= warm + 40:
                poured.add('garlic')
            result.sample(tick, measured(poured))
    if dense_thermal_samples:
        result.thermal = dense
    return result


def test_early_heat_does_not_erase_later_independent_steps():
    result = actual_thermal_history(early_heat=True)
    assert result.evidence()['thermal']['hold_confirmed_tick'] is not None
    assert scored(result).components['completion']['k'] == 2
    assert scored(result).value == 15
    assert not scored(result).components['task_success']


def test_early_heat_keeps_all_proven_independent_steps():
    result = actual_thermal_history(early_heat=True, first_pour_tick=600)
    assert scored(result).components['completion']['k'] == 2
    assert scored(result).value == 15


def test_later_valid_thermal_cycle_can_earn_partial_without_full_flow_success():
    result = actual_thermal_history(early_off=True)
    assert result.evidence()['thermal']['hold_confirmed_tick'] is not None
    assert scored(result).components['completion']['k'] == 2
    assert scored(result).value == 15
    assert not scored(result).components['task_success']


def test_uninterrupted_real_thermal_history_still_succeeds():
    assert scored(actual_thermal_history()).value == 100


@pytest.mark.parametrize('recipe', [['chili', 'onion'], ['onion', 'chili']])
@pytest.mark.parametrize('reverse_ids', [False, True])
def test_same_group_has_no_order_credit_in_either_identity_iteration(monkeypatch, recipe, reverse_ids):
    if reverse_ids:
        monkeypatch.setattr(observer_module, 'IDS', tuple(reversed(IDS)))
    result = observer(recipe)
    result.sample(1000, measured({'chili', 'onion'}))
    events = result.evidence()['ingredient_events']
    assert all(e['tick'] == e['sample_tick'] == 1000 and e['order_ambiguous'] for e in events)
    assert all(e['whole_portion'] and e['entered_pot'] for e in events)
    assert scored(result).components['completion']['k'] == 0


def test_ambiguous_group_preserves_prior_drawer_and_pour_prefix():
    result = observer(['chili', 'garlic', 'onion'])
    result.sample(900, measured(set()))
    result.sample(1000, measured({'chili'}))
    result.sample(1004, measured({'chili', 'garlic', 'onion'}))
    assert scored(result, 'hard').components['completion']['k'] == 1
    result.sample(2000, measured({'chili', 'garlic', 'onion'}))
    assert scored(result, 'hard').components['completion']['k'] == 1
    assert len(result.evidence()['ingredient_events']) == 3


def test_later_real_heat_cycle_cannot_credit_ambiguous_pour_group():
    result = actual_thermal_history(simultaneous_pours=True)
    assert result.evidence()['thermal']['hold_confirmed_tick'] is not None
    assert scored(result).components['completion']['k'] == 0


def test_extra_after_warm_preserves_independent_correct_steps():
    result = actual_thermal_history(extra_after_warm=True)
    assert scored(result).components['completion']['k'] == 2
    assert scored(result).value == 15
    assert not scored(result).components['task_success']


def test_initial_off_and_dense_natural_heating_samples_are_not_new_actions():
    assert scored(actual_thermal_history(dense_thermal_samples=True)).value == 100


def test_early_lid_does_not_erase_later_correct_pour():
    result = observer(['chili', 'onion'])
    result.sample(900, measured(set()))
    result.sample(1000, measured({'chili'}))
    sample = measured({'chili'})
    sample.update(pot_lid_supported=True, pot_lid_released=True, pot_lid_stable=True)
    result.sample(1050, sample)
    sample = measured({'chili', 'onion'})
    sample.update(pot_lid_supported=True, pot_lid_released=True, pot_lid_stable=True)
    result.sample(1100, sample)
    assert result.evidence()['pot_lid_release_tick'] == 1050
    assert scored(result, 'medium').components['completion']['k'] == 2
    assert scored(result, 'medium').components['first_error_tick'] == 1050


def test_lid_at_last_pour_keeps_existing_inclusive_preparation_boundary():
    result = observer(['chili', 'onion'])
    result.sample(900, measured(set()))
    result.sample(1000, measured({'chili'}))
    sample = measured({'chili', 'onion'})
    sample.update(pot_lid_supported=True, pot_lid_released=True, pot_lid_stable=True)
    result.sample(1100, sample)
    assert scored(result, 'medium').components['completion']['k'] == 2
    assert scored(result, 'medium').components['first_error_tick'] is None


def test_return_unit_does_not_require_ingredient_units_first():
    result=observer(['chili','onion'])
    result.thermal_sample(1000,dict(mode='HEAT',temperature=25.,window_entered=None,valid_switch=False))
    result.thermal_sample(1100,dict(mode='KEEP WARM',temperature=80.,window_entered=1100/120,valid_switch=True))
    result.thermal_sample(1101,dict(mode='OFF',temperature=80.,window_entered=1100/120,valid_switch=True))
    result.sample(1101,measured(set()))
    s=scored(result)
    assert not s.components['task_success']
    assert s.components['independent_steps']==(False,False,True,True)
    assert s.value==0
