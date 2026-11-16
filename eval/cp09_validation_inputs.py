"""Finite new clean validation compositions, not an arbitrary-seed factory."""
from importlib.resources import files
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import InstanceSpec,to_json,freeze
@lru_cache(maxsize=1)
def records():
 raw=files('mind2act_eval.env_cfg').joinpath('cp09_validation_records.json').read_bytes()
 d=json.loads(raw)
 if d.get('schema')!='mind2act_eval.validation_specs.v1':raise ValueError('CP09 VAL catalogue inventory differs')
 return freeze(d['records'])
def validate(spec):
 if isinstance(spec,dict):spec=InstanceSpec.from_dict(spec)
 if spec.task_id!='CP09' or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30:
  raise ValueError('CP09 requires its task and 120/30 Hz clock')
 return dict(full_spec=to_json(spec),runtime_row=to_json(spec.config))

def validate_assets(spec,assets):
 r=validate(spec)
 from mind2act_eval.sim.source_assets import source_file
 for role in ('station', 'validation_task', 'source_case', 'x5'):source_file(assets, role)
 case=json.loads(assets.resolve('source_case').read_text())
 from mind2act_eval.tasks.CP09.instances import signature
 if case['difficulty']!=spec.difficulty:raise ValueError('CP09 VAL source case differs')
 return r['runtime_row']
def validate_evidence(evidence,case):
 v=evidence.values;r=records().get(evidence.task_id+'/'+v['difficulty']+'/'+evidence.instance_id)
 if r is None or v.get('seed')!=r['full_spec']['seed'] or to_json(v.get('validation_source'))!=r['full_spec']['config']['validation_source'] or 'train_source' in v or 'test_source' in v:raise ValueError('CP09 VAL scoring identity differs')
 from mind2act_eval.tasks.CP09.instances import signature
 if signature(case)!=r['full_spec']['config']['source_case_sha256']:raise ValueError('CP09 VAL audit case differs')
