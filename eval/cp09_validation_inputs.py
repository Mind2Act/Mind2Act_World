"""Finite new clean validation compositions, not an arbitrary-seed factory."""
from importlib.resources import files
import hashlib,json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import InstanceSpec,to_json,freeze
from mind2act_eval.spec_identity import spec_digest
CATALOGUE_SHA256='8b34f4a4c1797aed8caba12e0c3d5c981dc852ee6ef0623e2f3acec6750dfe5f'
@lru_cache(maxsize=1)
def records():
 raw=files('mind2act_eval.env_cfg').joinpath('cp09_validation_records.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=CATALOGUE_SHA256:raise ValueError('CP09 VAL catalogue bytes differ')
 d=json.loads(raw)
 if d.get('schema')!='mind2act_eval.validation_specs.v1' or len(d['records'])!=30:raise ValueError('CP09 VAL catalogue inventory differs')
 return freeze(d['records'])
def validate(spec):
 if isinstance(spec,dict):spec=InstanceSpec.from_dict(spec)
 r=records().get(spec.task_id+'/'+spec.difficulty+'/'+spec.instance_id)
 if r is None or spec_digest(spec)!=r['full_spec_sha256']:raise ValueError('CP09 VAL complete spec/identity differs')
 if spec.config.get('layout_split')!='validation' or spec.config.get('layout_profile')!='clean' or spec.variation.mode!='clean' or spec.config.get('composition_kind')!='new_validation_composition' or 'train_source' in spec.config or 'test_source' in spec.config:raise ValueError('explicit clean independent composition identity required')
 return r
@lru_cache(maxsize=256)
def _file_sha(path,inode,size,mtime_ns,ctime_ns):
 p=Path(path);return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_assets(spec,assets):
 r=validate(spec)
 from mind2act_eval.sim.source_assets import verify_source_file
 # Every actual runtime dependency is code-owned, including MDL imports/textures;
 # source_assets direct-SHA readers additionally enforce their usual boundaries.
 root=Path(assets.root).resolve()
 for item in r['closure']:
  path=(root/item['path']).resolve(strict=True)
  if not path.is_relative_to(root) or _file_sha(str(path),path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)!=item['sha256']:raise ValueError('CP09 VAL dependency closure differs')
 for role,expected in r['runtime_asset_sha256'].items():verify_source_file(assets,role,expected)
 case=json.loads(assets.resolve('source_case').read_text())
 from mind2act_eval.tasks.CP09.instances import signature
 if signature(case)!=spec.config['source_case_sha256'] or case['difficulty']!=spec.difficulty or case['seed']!=spec.seed or case['case_id']!=spec.instance_id or case['composition_kind']!='new_validation_composition':raise ValueError('CP09 VAL source case differs')
 proof=json.loads(assets.resolve('validation_geometry_proof').read_text())
 if proof['task_sha256']!=r['runtime_asset_sha256']['validation_task'] or proof['source_case_sha256']!=spec.config['source_case_sha256'] or proof['instance_id']!=spec.instance_id or proof['difficulty']!=spec.difficulty:raise ValueError('CP09 VAL geometry proof differs')
 return r['runtime_row']
def validate_evidence(evidence,case):
 v=evidence.values;r=records().get(evidence.task_id+'/'+v['difficulty']+'/'+evidence.instance_id)
 if r is None or v.get('seed')!=r['full_spec']['seed'] or to_json(v.get('validation_source'))!=r['full_spec']['config']['validation_source'] or 'train_source' in v or 'test_source' in v:raise ValueError('CP09 VAL scoring identity differs')
 from mind2act_eval.tasks.CP09.instances import signature
 if signature(case)!=r['full_spec']['config']['source_case_sha256']:raise ValueError('CP09 VAL audit case differs')
