"""Closed shared capabilities for the inspected fixed50 CP09/PR4 inputs.

These records add no geometry or native settings. They retain exact original
source declarations and the compiled-scene pins already admitted by task code.
Task validators still authenticate raw inputs and actual geometry separately.
"""
from functools import lru_cache
from importlib import resources
import json

from mind2act_eval.contracts import freeze, to_json


@lru_cache(maxsize=1)
def _bindings():
    import hashlib
    raw=resources.files(__package__).joinpath('new15_source_bindings.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='78d66f1c0c3d7f4e6a2de470a3368b7131aab17cc5611e85dc2b96644eac0a18':
        raise ValueError('public shared capability bytes differ')
    value=json.loads(raw)
    if (value['schema']!='mind2act_eval.new15_fixed50_source_capabilities.v1'
            or len(value['bindings'])!=0):
        raise ValueError('fixed50 shared source capability inventory differs')
    return freeze(value['bindings'])


def matches_original_test_binding(spec):
    """Match every code-owned declaration; never accept a test-source flag alone."""
    row=_bindings().get(spec.instance_id)
    if row is None:
        return False
    if any(getattr(spec,key)!=row[key] for key in
           ('task_id','instance_id','difficulty','source_id','config_version','protocol_version','seed')):
        return False
    if to_json(spec.clock)!=to_json(row['clock']):
        return False
    for key,value in row['config'].items():
        actual=spec.config.get(key)
        if to_json(actual)==to_json(value):
            continue
        if spec.task_id=='CP09' and key=='geometry_decision':
            from mind2act_eval.robot.task_collision_overlay import canonical_geometry_decision
            try:
                if canonical_geometry_decision(actual,value)==value:
                    continue
            except ValueError:
                pass
        return False
    return True
