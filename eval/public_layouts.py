"""Finite full-spec admission for the public clean development subset."""
from importlib.resources import files
import hashlib,json
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def catalogue():
    raw=files('mind2act_eval.env_cfg').joinpath('public_layouts.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='36ef9d56928734d7339ae2b6f48eb3eef78828445df0d3ac564978bfc177c0f5':
        raise ValueError('public layout authority bytes differ')
    data=json.loads(raw)
    if data.get('schema')!='mind2act.public_clean_layouts.v1' or len(data['records'])!=2700:
        raise ValueError('public layout inventory differs')
    records={(r['task_id'],r['difficulty'],r['instance_id']):r for r in data['records']}
    if len(records)!=2700:raise ValueError('duplicate public layout identity')
    return records

def validate(spec):
    from mind2act_eval.spec_identity import spec_digest
    if spec.variation.mode!='clean' or 'test_source' in spec.config or 'fixed50_origin' in spec.config:
        raise ValueError('public development code accepts only selected clean train/validation layouts')
    row=catalogue().get((spec.task_id,spec.difficulty,spec.instance_id))
    if row is None or spec_digest(spec)!=row['spec_sha256']:
        raise ValueError('layout is absent from or differs from the complete public spec authority')
    if (row['split']=='validation')!=(spec.config.get('layout_split')=='validation'):
        raise ValueError('public layout split differs')
    return row
