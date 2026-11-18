"""Finite independent clean VAL identities; old train/test paths stay unchanged."""
from importlib.resources import files
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import InstanceSpec,to_json,freeze
@lru_cache(maxsize=1)
def records():
 raw=files(__package__).joinpath('config', 'validation_records.json').read_bytes()
 d=json.loads(raw)
 if d['schema']!='mind2act_eval.validation_specs.v1':raise ValueError('unsupported VAL catalogue schema')
 return freeze(d['records'])
def validate(spec):
 if isinstance(spec,dict):spec=InstanceSpec.from_dict(spec)
 if spec.task_id!='CP03' or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30:
  raise ValueError('CP03 requires its task and 120/30 Hz clock')
 return dict(full_spec=to_json(spec),source_asset_sha256={role:None for role in spec.assets})

def evidence_fields(spec):
 if not (spec.instance_id.startswith('VAL-CP03-') or spec.config.get('layout_split')=='validation'):return {}
 validate(spec);return dict(seed=spec.seed,validation_source=to_json(spec.config['validation_source']))
def validate_evidence(evidence):
 v=evidence.values;r=records().get(evidence.task_id+'/'+v['difficulty']+'/'+evidence.instance_id)
 if r is None or v.get('seed')!=r['full_spec']['seed'] or to_json(v.get('validation_source'))!=r['full_spec']['config']['validation_source'] or 'train_source' in v or 'test_source' in v:raise ValueError('VAL scoring identity/condition differs')
