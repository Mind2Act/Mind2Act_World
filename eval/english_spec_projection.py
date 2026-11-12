"""Explicit instruction-only projection; original source identities remain evidence.

A projected spec is a candidate, not a release: catalogs, source bindings and
variation manifests must be rebuilt and verified by their respective owners.
"""
from copy import deepcopy
from types import SimpleNamespace

from mind2act_eval.model_text import require_english_model_text
from mind2act_eval.tasks._instructions import public_instruction

OLD15 = frozenset(('E1', 'F1', 'P1', 'S1', 'M1', 'DR1', 'DR2', 'PR1', 'PR2',
                   'PR3', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05'))


def project_instruction(spec: dict) -> dict:
    """Return an independent spec; change only old15 config.instruction."""
    projected = deepcopy(spec)
    if spec['task_id'] in OLD15:
        instruction = public_instruction(SimpleNamespace(
            task_id=spec['task_id'], difficulty=spec['difficulty'], config=spec.get('config', {})))
        projected.setdefault('config', {})['instruction'] = require_english_model_text(instruction)
    return projected
