"""Recompute CP08 chronological physical prefix from complete raw measurements."""
from mind2act_eval.scoring import cp_score_from_completion
from mind2act_eval.contracts import to_json
from .physical import evaluate, CONTACT_SCHEMA
from .replay import Replay
from .projection import validate as validate_projection
from .audit import read
from .instances import selected, signature


def score(evidence):
    if evidence.task_id != 'CP08' or type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError('CP08 requires a nonnegative physics tick')
    values = to_json(evidence.values)
    config = values['source_case']
    if 'validation_source_spec' in values:
        from .validation import validate_evidence
        validate_evidence(evidence)
    elif 'training_source_spec' in values:
        from mind2act_eval.contracts import InstanceSpec
        from mind2act_eval.cp08_training_inputs import validate as validate_training
        from .projection import validate_training as validate_training_projection
        spec=InstanceSpec.from_dict(values['training_source_spec'])
        validate_training(spec)
        if ('test_source' in values or spec.variation.mode!='clean'
                or spec.task_id!=evidence.task_id or spec.instance_id!=evidence.instance_id
                or spec.difficulty!=config['level']
                or config!=to_json(spec.config['source_case'])
                or values['source_projection']!=to_json(spec.config['source_projection'])):
            raise ValueError('CP08 training scoring source identity/condition differs')
        validate_training_projection(values['source_projection'],spec.difficulty,config)
    elif 'test_source' in values:
        from types import SimpleNamespace
        from mind2act_eval.original_test_inputs import read_audit
        spec=SimpleNamespace(task_id='CP08',instance_id=evidence.instance_id,difficulty=config['level'],
            seed=values['test_source']['source_seed'],config=dict(test_source=values['test_source']))
        raw=read_audit(spec,values)
        if (raw['task_id']!='CP08' or raw['split']!='eval'
                or signature(config)!=signature(raw['config']['source_case'])):
            raise ValueError('CP08 original test scoring condition differs')
        validate_projection(values['source_projection'],config['level'],original_case=raw['config']['source_case'])
    else:
        bound = selected(config['level'])
        if evidence.instance_id != bound['instance_id'] or signature(config) != signature(bound['source_case']):
            raise ValueError('CP08 scoring source instance differs')
        validate_projection(values['source_projection'], config['level'])
    samples = read(values['audit'], evidence.tick)
    initial, raw = samples[0], samples[1:]
    if initial.get("contact_measurement_schema") != CONTACT_SCHEMA:
        raise ValueError("CP08 initial owned contact evidence incomplete")
    if len(raw) != evidence.tick:
        raise ValueError('CP08 raw tick coverage incomplete')
    replay = Replay(config, initial,
                    source_state_tick=values['source_projection']['source_state_tick'])
    for row in raw:
        replay.sample(row)
    result = evaluate(replay.rows, config, initial)
    core_ids = ['B','C','D','E','F'] if config['level']=='hard' else ['B','A','C'] + (['D'] if config['level']=='medium' else [])
    k = sum(result['block_checks'][name] for name in core_ids)
    return cp_score_from_completion(result['success'], k, len(core_ids), difficulty=config['level'],
        components={'scoring_version': 'mind2act-final-scoring-20261105', 'physical_report': result, 'source_projection': values['source_projection']})
