"""CP07 binary actual driven arrival at T0, FINAL-VERSION.md."""
from math import isfinite
from types import SimpleNamespace
from mind2act_eval.contracts import NativeEvidence, score_from_completion
from mind2act_eval.scoring import SCORING_VERSION

# Geometry inventory only; no bridge milestones enter the final score.
COUNTS={'easy':(8,8),'medium':(13,18),'hard':(13,28)}
SCHEMA='cp07.physical.final'


def score(evidence:NativeEvidence):
    if evidence.task_id!='CP07':raise ValueError('CP07 task identity mismatch')
    v=evidence.values
    if v.get('schema')!=SCHEMA or v.get('scoring_version')!=SCORING_VERSION:
        raise ValueError('CP07 measured final car evidence required; old aggregate cannot be rescored')
    if 'validation_source_spec' in v:
        from .validation import validate_evidence
    else:
        from mind2act_eval.cp07_training_inputs import validate_evidence
    validate_evidence(evidence)
    car=v.get('terminal_car')
    if not isinstance(car,dict) and not hasattr(car,'keys'):raise ValueError('CP07 actual arrival measurements required')
    if not {'p_m','far_bank_supported','far_bank_center_inside'}<=car.keys():raise ValueError('CP07 actual arrival measurements required')
    if len(car['p_m'])!=3 or not all(isfinite(x) for x in car['p_m']):
        raise ValueError('CP07 finite terminal vehicle pose required')
    for value in (v['drive_launched'],car['far_bank_supported'],car['far_bank_center_inside']):
        if type(value) is not bool:raise ValueError('CP07 actual arrival measurements required')
    success=bool(v['drive_launched'] and car['far_bank_supported'] and car['far_bank_center_inside'])
    return score_from_completion(success,int(success),1,components={
        'scoring_version':SCORING_VERSION,'binding_mode':v['binding_mode'],
        'solutions_scope':v['solutions_scope'],'terminal_car':dict(car)})
