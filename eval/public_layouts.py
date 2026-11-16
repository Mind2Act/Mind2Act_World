"""Development layout metadata and optional offline release verification."""
from importlib.resources import files
import hashlib
import json
from functools import lru_cache


@lru_cache(maxsize=1)
def catalogue():
    data = json.loads(files('mind2act_eval.env_cfg').joinpath('public_layouts.json').read_bytes())
    if data.get('schema') != 'mind2act.public_clean_layouts.v1':
        raise ValueError('unsupported public layout catalogue schema')
    records = {(r['task_id'], r['difficulty'], r['instance_id']): r for r in data['records']}
    if len(records) != len(data['records']):
        raise ValueError('duplicate public layout identity')
    return records


def validate(spec):
    """Check the development input boundary without pinning a seed or file digest.

    Task loaders validate their required parameters and scene assets. The shipped
    catalogue describes the bundled examples; it is not a runtime whitelist.
    """
    if spec.variation.mode != 'clean' or 'test_source' in spec.config or 'fixed50_origin' in spec.config:
        raise ValueError('public development inputs must be clean train/validation layouts')
    records = catalogue()
    if spec.task_id not in {key[0] for key in records}:
        raise ValueError('unsupported development task: ' + spec.task_id)
    split = 'validation' if spec.config.get('layout_split') == 'validation' else 'train'
    return records.get((spec.task_id, spec.difficulty, spec.instance_id),
                       dict(task_id=spec.task_id, difficulty=spec.difficulty,
                            instance_id=spec.instance_id, split=split))


def verify(spec):
    """Offline: verify an unmodified example against the released catalogue."""
    from mind2act_eval.spec_identity import spec_digest
    raw = files('mind2act_eval.env_cfg').joinpath('public_layouts.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'b26de0d1057fe9d9ac1dfbdd11451a907912bb27d68e612deec44524e6c0f1d9':
        raise ValueError('public layout authority bytes differ')
    validate(spec)
    row = catalogue().get((spec.task_id, spec.difficulty, spec.instance_id))
    if row is None or spec_digest(spec) != row['spec_sha256']:
        raise ValueError('layout differs from the released example')
    if (row['split'] == 'validation') != (spec.config.get('layout_split') == 'validation'):
        raise ValueError('public layout split differs')
    return row
