"""Independent CP06 source-bound raw audit replay into authority 0–100 scores."""
from mind2act_eval.scoring import cp_score_from_completion
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.contracts import to_json
from .instances import validate_core
from .audit import read
from .physical import evaluate


def score(evidence):
    if evidence.task_id!='CP06' or type(evidence.tick) is not int or evidence.tick<0:
        raise ValueError('CP06 source physics evidence required')
    values=to_json(evidence.values)
    core=validate_core(values['source_task'],values['difficulty'])
    samples=read(values['audit'],evidence.tick)
    initial=samples[0]['raw']
    ticks=[dict(physics_tick=r['tick'],simulation_time_s=r['raw']['simulation_time_s'],raw=r['raw']) for r in samples[1:]]
    result=evaluate(core,ticks,initial)
    core_n = {'easy':3,'medium':4,'hard':5}[values['difficulty']]
    core_k = len(result['placed_fruits']) + min(2, len(result['placed_weights']))
    return cp_score_from_completion(result['success'],core_k,core_n,difficulty=values['difficulty'],
        components=dict(scoring_version=SCORING_VERSION,physical_report=result,source_spec_sha256=values.get('source_spec_sha256')))
