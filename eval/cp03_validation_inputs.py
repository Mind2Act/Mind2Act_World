"""Finite independent clean VAL identities; old train/test paths stay unchanged."""
from importlib.resources import files
import hashlib,json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import InstanceSpec,to_json,freeze
from mind2act_eval.spec_identity import spec_digest
CATALOGUE_SHA256='429370039fb595fda1efff3f12b5cf28823a00cd1ec21d2f4ffb997df833911b'
@lru_cache(maxsize=1)
def records():
 raw=files('mind2act_eval.env_cfg').joinpath('cp03_validation_records.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=CATALOGUE_SHA256:raise ValueError('VAL catalogue bytes differ')
 d=json.loads(raw)
 if d['schema']!='mind2act_eval.validation_specs.v1' or len(d['records'])!=30:raise ValueError('VAL catalogue schema/count differs')
 return freeze(d['records'])
def validate(spec):
 if isinstance(spec,dict):spec=InstanceSpec.from_dict(spec)
 r=records().get(spec.task_id+'/'+spec.difficulty+'/'+spec.instance_id)
 if r is None or spec_digest(spec)!=r['full_spec_sha256']:raise ValueError('VAL full spec/identity differs')
 if (spec.config.get('layout_split')!='validation' or spec.config.get('layout_profile')!='clean' or spec.variation.mode!='clean' or 'train_source' in spec.config or 'test_source' in spec.config):raise ValueError('independent clean validation identity required')
 return r
def evidence_fields(spec):
 if not (spec.instance_id.startswith('VAL-CP03-') or spec.config.get('layout_split')=='validation'):return {}
 validate(spec);return dict(seed=spec.seed,validation_source=to_json(spec.config['validation_source']))
def validate_evidence(evidence):
 v=evidence.values;r=records().get(evidence.task_id+'/'+v['difficulty']+'/'+evidence.instance_id)
 if r is None or v.get('seed')!=r['full_spec']['seed'] or to_json(v.get('validation_source'))!=r['full_spec']['config']['validation_source'] or 'train_source' in v or 'test_source' in v:raise ValueError('VAL scoring identity/condition differs')
