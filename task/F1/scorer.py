"""F1: exploration and shortest formal route, with terminal arrival and last-key return."""
from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION

from .instances import layout
from .judge import Judge


def score(evidence: NativeEvidence) -> Score:
    if evidence.instance_id.startswith('VAL-F1-reuse-') or 'gap_composition_spec' in evidence.values:
        from mind2act_eval.gap_composition_inputs import validate_evidence
        validate_evidence(evidence)
    v = evidence.values
    rows = v['rows']
    if not rows:
        raise ValueError('Empty F measured trace')
    judge = Judge(v['case'], v['protocol'], layout())
    grade = None
    for row in rows:
        grade = judge.sample(row)
    space = v['case']['space']
    level = 'easy' if space['kind']=='line' else {3:'medium',4:'hard'}[space['side']]
    cap = {'easy':30.,'medium':22.,'hard':15.}[level]
    if grade['protocol_violations'] or not grade['prerequisites']:
        value, tier = 0., 'violation_or_prerequisite'
    elif grade['task_success']:
        value, tier = 100., 'success'
    elif grade['at_target']:
        value, tier = (cap/2, 'at_goal_released') if grade['final_released'] else (cap/4, 'at_goal_last_key_down')
    else:
        value, tier = 0., 'not_at_goal'
    return Score(value, dict(scoring_version=SCORING_VERSION, success=grade['task_success'], tier=tier, efficiency={k: grade[k] for k in (
        'execution_presses', 'shortest_presses', 'path_redundancy', 'effective_probes')},
        protocol_violations=grade['protocol_violations']))
