"""CP04 independent steps and recipe LCS from FINAL-VERSION.md."""
from mind2act_eval.scoring import cp_score_from_completion
from collections import Counter

from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION, lcs_length

N_STEPS = {'easy': 4, 'medium': 6, 'hard': 7}
WINDOWS = {'easy': 4., 'medium': 3., 'hard': 2.}
HZ = 120


def _actions(thermal):
    actions = []
    for row in thermal['events']:
        if not actions or actions[-1]['mode'] != row['mode']:
            actions.append(row)
    return actions


def _valid_transition(thermal, difficulty):
    start = thermal['response_window_start_tick']
    for before, after in zip(_actions(thermal), _actions(thermal)[1:]):
        if (before['mode'] == 'HEAT' and after['mode'] == 'KEEP_WARM'
                and start is not None and before['tick'] <= start <= after['tick']
                and after['temperature_c'] >= 80.
                and after['tick']-start <= WINDOWS[difficulty]*HZ):
            return before['tick'], after['tick']
    return None


def score(evidence: NativeEvidence) -> Score:
    v = evidence.values
    difficulty, recipe, e = v['difficulty'], tuple(v['recipe']), v['cp04_evidence']
    if difficulty not in N_STEPS or len(recipe) != (3 if difficulty == 'hard' else 2):
        raise ValueError('CP04 difficulty/recipe count differs')
    if e['tick_hz'] != HZ or e['thermal']['history_complete'] is not True:
        raise ValueError('CP04 complete 120 Hz thermal evidence required')
    events = tuple(e['ingredient_events'])
    if any(b['tick'] < a['tick'] for a,b in zip(events,events[1:])):
        raise ValueError('CP04 ingredient evidence is not chronological')
    ambiguous = {t for t,n in Counter(r['tick'] for r in events).items() if n>1}
    def valid(row):
        return bool(row['whole_portion'] and row['entered_pot'] and row['cap_removed'] and row['terminal_portion_contained']
                    and row['tick'] not in ambiguous and not row.get('order_ambiguous', False))
    actual = tuple(r['ingredient_id'] for r in events if valid(r))
    ingredient_k = lcs_length(actual, recipe)
    exact = bool(len(events) == len(recipe) and all(valid(r) for r in events) and actual == recipe)
    prep = difficulty != 'easy'
    drawer = e['drawer_open_tick']
    drawer_ok = bool(drawer is not None and (not events or drawer < events[0]['tick']))
    thermal = e['thermal']; actions = _actions(thermal)
    heat = next((r['tick'] for r in actions if r['mode']=='HEAT'), None)
    lid = e['pot_lid_release_tick']
    lid_ok = bool(events and lid is not None and events[-1]['tick'] <= lid
                  and (heat is None or lid < heat))
    transition = _valid_transition(thermal, difficulty)
    warm_ok = transition is not None
    last_warm = next((r['tick'] for r in reversed(actions) if r['mode']=='KEEP_WARM'), None)
    off_ok = bool(actions and actions[-1]['mode']=='OFF' and last_warm is not None
                  and actions[-1]['tick'] > last_warm)
    returns = tuple(e['returns'])
    returned = (Counter(r['ingredient_id'] for r in returns) == Counter(recipe)
                and all(r['jar_returned'] and r['cap_in_tray'] for r in returns))
    finish = bool(off_ok and returned)
    steps = ([drawer_ok] if prep else []) + [True]*ingredient_k + [False]*(len(recipe)-ingredient_k)
    steps += ([lid_ok] if prep else []) + [warm_ok, finish]
    k, n = sum(steps), N_STEPS[difficulty]
    meaningful = actions[1:] if actions and actions[0]['mode']=='OFF' else actions
    flow_modes = tuple(r['mode'] for r in meaningful)
    ordered_heat = bool(transition and events and events[-1]['tick'] < transition[0]
                        and (not prep or drawer_ok and lid_ok))
    success = bool(k==n and exact and ordered_heat and flow_modes==('HEAT','KEEP_WARM','OFF'))
    bad_pours = [r['tick'] for i,r in enumerate(events)
                 if i>=len(recipe) or not valid(r) or r['ingredient_id']!=recipe[i]]
    errors = list(bad_pours)
    if prep and events and not drawer_ok: errors.append(events[0]['tick'])
    if prep and lid is not None and events and lid < events[-1]['tick']: errors.append(lid)
    if heat is not None and events and heat <= events[-1]['tick']: errors.append(heat)
    if prep and heat is not None and not lid_ok: errors.append(heat)
    return cp_score_from_completion(success, ingredient_k, len(recipe), difficulty=difficulty, components={
        'scoring_version':SCORING_VERSION, 'independent_steps':steps,
        'recipe_lcs':ingredient_k, 'recipe_exact':exact,
        'first_error_tick':min(errors) if errors else None})
