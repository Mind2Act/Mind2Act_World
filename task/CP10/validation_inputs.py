"""Finite independent clean VAL identities; old train/test paths stay unchanged."""
from importlib.resources import files
import hashlib,json
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
 if spec.task_id!='CP10' or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30:
  raise ValueError('CP10 requires its task and 120/30 Hz clock')
 return dict(full_spec=to_json(spec))

def evidence_fields(spec):
 if not (spec.instance_id.startswith('VAL-CP10-') or spec.config.get('layout_split')=='validation'):return {}
 validate(spec);return dict(seed=spec.seed,validation_source=to_json(spec.config['validation_source']))
def validate_evidence(evidence):
 v=evidence.values;r=records().get(evidence.task_id+'/'+v['difficulty']+'/'+evidence.instance_id)
 if r is None or v.get('seed')!=r['full_spec']['seed'] or to_json(v.get('validation_source'))!=r['full_spec']['config']['validation_source'] or 'train_source' in v or 'test_source' in v:raise ValueError('VAL scoring identity/condition differs')

 from mind2act_eval.tasks.CP10.prefix import TRACE_CONTRACT
 if v.get('trace_contract')!=TRACE_CONTRACT:raise ValueError('VAL scoring requires complete physical prefix')
 if (to_json(v.get('source_config'))!=to_json(r['full_spec']['config']['source_config']) or to_json(v.get('collection_protocol'))!=to_json(r['full_spec']['config']['collection_protocol']) or v.get('rule_version')!=r['full_spec']['config']['rule_version']):raise ValueError('VAL scoring source/protocol differs')
def validate_assets(spec,assets):
 validate(spec)
 from mind2act_eval.tasks.CP10.original_inputs import _static_geometry,_validate_projection,STATIC_POOL_SHA256,STATIC_POOL_GRAPH_SHA256
 c=to_json(spec.config['source_config']);_static_geometry(c)
 scene=assets.resolve('scene');pool=assets.resolve('validation_static_pool')
 if assets.resolve('station')!=scene:raise ValueError('VAL station and scene differ')
 raw=scene.read_bytes();_validate_projection(raw,scene,pool,c)
 return c
def scoring_config_sha256(evidence):
 validate_evidence(evidence)
 return hashlib.sha256(json.dumps(to_json(evidence.values['source_config']),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
