"""Adapt a verified native-terminal prefix to the independent receipt scorer."""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from mind2act_eval.contracts import InstanceSpec,NativeEvidence,to_json
from .prefix import TRACE_CONTRACT, verify_prefix

TRAINING_CONFIG_SHA256 = MappingProxyType({level: row['source_config_canonical_sha256']
    for level, row in json.loads((Path(__file__).parent/'config/source.json').read_text())['training_cohorts'].items()})


def scoring_evidence(evidence):
    values = evidence.values
    validation_case=evidence.instance_id.startswith('VAL-CP10-') or 'validation_source' in values
    if validation_case:
        from mind2act_eval.cp10_validation_inputs import validate_evidence
        validate_evidence(evidence)
    if any(key in values for key in ('physical_prefix_receipt', 'scored_until_tick',
            'native_terminal_tick', 'frozen_native_snapshot')):
        if 'trace_contract' not in values or 'source_config' not in values:
            raise ValueError('CP10 native prefix requires its source and trace contract')
    if 'original_test_inputs' in values or values.get('input_binding_kind')=='original_test':
        from .original_inputs import read_audit
        bound=to_json(values['original_test_inputs'])
        spec=InstanceSpec.from_dict(bound['spec'])
        if (values.get('input_binding_kind')!='original_test' or spec.task_id!='CP10'
                or spec.instance_id!=evidence.instance_id or spec.difficulty!=values['difficulty']
                or 'trace_contract' not in values
                or to_json(spec.config['collection_protocol'])!=to_json(values['collection_protocol'])
                or spec.config['rule_version']!=values['rule_version']):
            raise ValueError('CP10 original test evidence identity/protocol differs')
        config=read_audit(spec,bound['audit'])
        if to_json(values['source_config'])!=config:
            raise ValueError('CP10 original test scoring source differs')
    elif 'source_config' in values:
        # Removing optional original-test labels cannot select a weaker path.
        # The other native path is the same exact code-owned training cohort
        # accepted by scene.validate_binding, not arbitrary receipt metadata.
        raw=json.dumps(to_json(values['source_config']),sort_keys=True,
            separators=(',', ':'),allow_nan=False).encode()
        from mind2act_eval.cp10_training_inputs import admitted_config_sha256
        if validation_case:
            from mind2act_eval.cp10_validation_inputs import scoring_config_sha256
            expected=scoring_config_sha256(evidence)
        else:
            expected=admitted_config_sha256(evidence.instance_id,values['difficulty'])
        if expected is None:expected=TRAINING_CONFIG_SHA256.get(values['difficulty'])
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('CP10 native prefix lacks original audit or selected training source')
    if 'trace_contract' not in values:
        return evidence
    if values['trace_contract'] != TRACE_CONTRACT:
        raise ValueError('CP10 unknown physical prefix envelope')
    if type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError('CP10 real outer clock is invalid')
    if values['source_config']['difficulty'] != values['difficulty']:
        raise ValueError('CP10 source and scoring difficulties differ')
    snapshot = verify_prefix(values, evidence.tick)
    return NativeEvidence(evidence.task_id, evidence.instance_id, snapshot['last_tick'], values)
