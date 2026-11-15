"""Independent CP06 source-bound raw audit replay into authority 0–100 scores."""
from mind2act_eval.scoring import cp_score_from_completion
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.contracts import to_json
from .instances import selected, validate_core
from .audit import read
from .physical import evaluate


def score(evidence):
    if evidence.task_id!='CP06' or type(evidence.tick) is not int or evidence.tick<0:
        raise ValueError('CP06 source physics evidence required')
    values=to_json(evidence.values)
    if 'validation_source_spec' in values:
        from .validation import validate_evidence
        spec=validate_evidence(evidence)
        core=to_json(spec.config['source_task'])
        row=dict(source_spec_sha256=spec.config['source_spec_sha256'])
    elif 'training_source_spec' in values:
        from mind2act_eval.contracts import InstanceSpec
        from mind2act_eval.cp06_training_inputs import validate as validate_training
        spec=InstanceSpec.from_dict(values['training_source_spec'])
        validate_training(spec)
        if ('test_source' in values or spec.variation.mode!='clean'
                or spec.task_id!=evidence.task_id or spec.instance_id!=evidence.instance_id
                or spec.difficulty!=values['difficulty']
                or values['source_spec_sha256']!=spec.config['source_spec_sha256']
                or values['source_task_sha256']!=spec.config['declared_task_sha256']
                or values['source_task']!=to_json(spec.config['source_task'])):
            raise ValueError('CP06 training scoring source identity/condition differs')
        core=to_json(spec.config['source_task'])
        row=dict(source_spec_sha256=spec.config['source_spec_sha256'])
    elif 'test_source' in values:
        from types import SimpleNamespace
        from mind2act_eval.original_test_inputs import read_audit
        from .instances import original_core,signature
        spec=SimpleNamespace(task_id='CP06',instance_id=evidence.instance_id,difficulty=values['difficulty'],
            seed=values['test_source']['source_seed'],config=dict(test_source=values['test_source']))
        raw=read_audit(spec,values)
        if (raw['task_id']!='CP06' or raw['split']!='eval'
                or values['source_spec_sha256']!=values['test_source']['source_configuration_sha256']
                or values['source_task_sha256']!=raw['binding']['config_sha256']
                or signature(values['source_task'])!=signature(original_core(raw))):
            raise ValueError('CP06 original test scoring core differs')
        core=original_core(raw);row=dict(source_spec_sha256=values['source_spec_sha256'])
    else:
        row=selected(values['difficulty'])
        if (evidence.instance_id!=row['instance_id'] or values['source_spec_sha256']!=row['source_spec_sha256']
                or values['source_task_sha256']!=row['declared_task_sha256']):
            raise ValueError('CP06 original selected H5/spec binding differs')
        core=validate_core(values['source_task'],values['difficulty'])
    samples=read(values['audit'],evidence.tick)
    initial=samples[0]['raw']
    ticks=[dict(physics_tick=r['tick'],simulation_time_s=r['raw']['simulation_time_s'],raw=r['raw']) for r in samples[1:]]
    result=evaluate(core,ticks,initial)
    core_n = {'easy':3,'medium':4,'hard':5}[values['difficulty']]
    core_k = len(result['placed_fruits']) + min(2, len(result['placed_weights']))
    return cp_score_from_completion(result['success'],core_k,core_n,difficulty=values['difficulty'],
        components=dict(scoring_version=SCORING_VERSION,physical_report=result,source_spec_sha256=row['source_spec_sha256']))
