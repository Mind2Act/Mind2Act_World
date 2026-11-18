"""Adapt a verified native-terminal prefix to the independent receipt scorer."""

from mind2act_eval.contracts import NativeEvidence
from .prefix import TRACE_CONTRACT, verify_prefix


def scoring_evidence(evidence):
    values = evidence.values
    if any(key in values for key in ('physical_prefix_receipt', 'scored_until_tick',
            'native_terminal_tick', 'frozen_native_snapshot')):
        if 'trace_contract' not in values or 'source_config' not in values:
            raise ValueError('CP10 native prefix requires its source and trace contract')
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
