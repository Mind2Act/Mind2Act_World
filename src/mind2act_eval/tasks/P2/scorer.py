"""P2: terminal supported bottom prefix, independently replayed from native physics."""
from mind2act_eval.contracts import NativeEvidence, Score, to_json
from mind2act_eval.scoring import SCORING_VERSION
from .binding import VERSION, SCORING_VERSION as RUBRIC_VERSION, SOURCE_VERSION, DISMANTLING_RULE, EVALUATION_MODE, digest, validate_source
from .evidence import read_rows
from .progress import Progress
from .stages import stage_value


def score(evidence: NativeEvidence) -> Score:
    if evidence.task_id != 'P2':
        raise ValueError('P2 scorer received another task')
    v = to_json(evidence.values)
    source = v['source_instance']
    if v['input_protocol_version'] not in (VERSION, SOURCE_VERSION):
        raise ValueError('P2 unknown input protocol')
    validate_source(source)
    if digest(source) != v['source_record_sha256'] or v['dismantling_rule_version'] != DISMANTLING_RULE:
        raise ValueError('P2 source/ruling receipt mismatch')
    progress = Progress(source, v['initial'])
    for row in read_rows(v['measurements']):
        progress.sample(row)
    if progress.facts.tick != evidence.tick:
        raise ValueError('P2 terminal native tick differs from sealed evidence')
    result = progress.result()
    value = stage_value(difficulty=source['difficulty'], **{key: result[key] for key in
        ('valid_disassembly', 'restored_prefix', 'target_count', 'final_conditions_met')})
    return Score(float(value), dict(result, rubric=RUBRIC_VERSION, scoring_version=SCORING_VERSION, source_input_version=VERSION, dismantling_rule_version=DISMANTLING_RULE,
                                   evaluation_mode=EVALUATION_MODE, formal_eligible=True,
                                   accepted_expert_episode=False,
                                   input_protocol_version=v['input_protocol_version']))
