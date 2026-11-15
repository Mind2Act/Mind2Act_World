"""Finite independent clean VAL identities; old train/test paths stay unchanged."""
from importlib.resources import files
import hashlib,json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import InstanceSpec,to_json,freeze
from mind2act_eval.spec_identity import spec_digest
CATALOGUE_SHA256='5a9b15dea94eae370204603537680a2bd276e64c6bac6cd9b66a0bcbac90d948'
@lru_cache(maxsize=1)
def records():
 raw=files('mind2act_eval.env_cfg').joinpath('cp10_validation_records.json').read_bytes()
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
 from mind2act_eval.original_asset_graph import verify_graph
 from mind2act_eval.verified_files import verified_bytes
 c=to_json(spec.config['source_config']);_static_geometry(c)
 scene=assets.resolve('scene');pool=assets.resolve('validation_static_pool')
 if assets.resolve('station')!=scene:raise ValueError('VAL station and scene differ')
 raw=verified_bytes(scene,spec.config['scene_sha256']);verified_bytes(pool,STATIC_POOL_SHA256[spec.difficulty]);verify_graph(pool,assets.root,STATIC_POOL_GRAPH_SHA256[spec.difficulty]);_validate_projection(raw,scene,pool,c)
 return c
def scoring_config_sha256(evidence):
 validate_evidence(evidence)
 return hashlib.sha256(json.dumps(to_json(evidence.values['source_config']),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
