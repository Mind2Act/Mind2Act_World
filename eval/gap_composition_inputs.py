"""Finite new physical compositions and explicit development training-condition reuse."""
from importlib.resources import files
import hashlib,json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import InstanceSpec,to_json
from mind2act_eval.spec_identity import spec_digest
@lru_cache(maxsize=1)
def catalogue():
 raw=files('mind2act_eval.env_cfg').joinpath('gap_composition_records.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!='4c6237c8643fe9e2fc112fc1c0d4d9b7bea351acf927c2817fcc3729de3811d4':raise ValueError('gap composition table differs')
 d=json.loads(raw)
 if d['schema']!='mind2act.gap_composition.v1' or len(d['records'])!=22:raise ValueError('gap composition table identity differs')
 return d
@lru_cache(maxsize=1024)
def file_sha(path,size,mtime,ctime):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(spec,assets=None):
 from mind2act_eval.public_layouts import validate as public_validate
 public_validate(spec);d=catalogue();matches=[r for r in d['records'] if (r['task_id'],r['difficulty'],r['instance_id'])==(spec.task_id,spec.difficulty,spec.instance_id)]
 if len(matches)!=1 or matches[0]['spec_sha256']!=spec_digest(spec) or spec.config.get('gap_composition_authority')!='gap-composition-v1' or spec.config.get('layout_split')!='validation' or spec.variation.mode!='clean' or any(k in spec.config for k in ['train_source','test_source']):raise ValueError('finite gap composition identity/condition differs')
 if assets is not None:
  for row in d['files_by_task'][spec.task_id]:
   p=(assets.root/row['path']).resolve(strict=True)
   if not p.is_relative_to(assets.root):raise ValueError('gap dependency escapes asset root')
   s=p.stat()
   if s.st_size!=row['size'] or file_sha(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=row['sha256']:raise ValueError('gap dependency bytes differ')
 return matches[0]
def evidence_fields(spec):
 if spec.config.get('gap_composition_authority')!='gap-composition-v1':return {}
 validate(spec);return dict(gap_composition_spec=to_json(spec))
def validate_evidence(evidence):
 v=to_json(evidence.values);spec=InstanceSpec.from_dict(v['gap_composition_spec']);validate(spec)
 if type(evidence.tick) is not int or evidence.tick<0 or (evidence.task_id,evidence.instance_id)!=(spec.task_id,spec.instance_id):raise ValueError('gap evidence identity/tick differs')
 if spec.task_id=='F1':
  if v['case']!=to_json(spec.config['case']) or v['protocol']!=to_json(spec.config['protocol']):raise ValueError('gap F1 evidence condition differs')
 else:
  if v['source_case']!=to_json(spec.config['source_case']) or v['input_protocol_version']!=spec.config['input_protocol_version'] or v['source_start_tick']!=0 or v['source_origin_receipt'] is not None:raise ValueError('gap F2 evidence condition differs')
 return spec
