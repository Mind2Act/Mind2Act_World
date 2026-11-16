"""Recompute CP08 chronological physical prefix from complete raw measurements."""
from mind2act_eval.scoring import cp_score_from_completion
from mind2act_eval.contracts import to_json
from .physical import evaluate, CONTACT_SCHEMA
from .replay import Replay
from .audit import read


def score(evidence):
    if evidence.task_id != 'CP08' or type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError('CP08 requires a nonnegative physics tick')
    values = to_json(evidence.values)
    config = values['source_case']
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
