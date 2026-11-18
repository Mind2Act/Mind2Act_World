"""Exact admission of the selected public PR4 train and validation layouts."""
from mind2act_eval.contracts import InstanceSpec


def validate_case_source(spec):
    if not isinstance(spec, InstanceSpec) or spec.task_id != 'PR4':
        raise ValueError('PR4 case source binding requires a PR4 InstanceSpec')
    from mind2act_eval.public_layouts import validate
    validate(spec)
