"""Code-pinned training admission; raw inputs and H5 lineage remain mandatory."""
from importlib.resources import files
import hashlib,json,re
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import to_json,freeze
CATALOGUE_SHA256 = '95214fad77c74bcc3246cf03a56021e8052b9538d9d89bbde094cfc01566d490'

@lru_cache(maxsize=1)
def _records():
    raw=files('mind2act_eval.env_cfg').joinpath('cp10_training_source_records.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CATALOGUE_SHA256:
        raise ValueError('training admission catalogue SHA differs')
    value=json.loads(raw)
    if value['schema']!='mind2act_eval.training_source_records.v1':
        raise ValueError('training admission schema differs')
    return freeze(value['records'])

def validate(spec):
    source=to_json(spec.config.get('train_source'))
    if not isinstance(source,dict) or source.get('schema')!='mind2act_eval.training_source.v1' or source.get('split')!='train':
        raise ValueError('explicit training source identity required')
    if 'test_source' in spec.config:
        raise ValueError('training and test identities cannot be combined')
    key=spec.task_id+'/'+spec.difficulty+'/'+spec.instance_id
    record=_records().get(key)
    if record is None:raise ValueError('training instance has no reviewed admission record')
    for k in ('task_id','difficulty','instance_id','seed','source_id','config_version','asset_version','protocol_version'):
        if record['identity'][k]!=getattr(spec,k):raise ValueError('training identity differs: '+k)
    if record['identity']['clock']!=to_json(spec.clock):raise ValueError('training clock differs')
    config=to_json(spec.config)
    if hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=record['config_sha256']:
        raise ValueError('training condition/configuration SHA differs')
    if source!=record['train_source'] or to_json(spec.assets)!=record['assets']:
        raise ValueError('training original input/H5/assets binding differs')
    for k in ('input_sha256','h5_sha256'):
        if re.fullmatch('[0-9a-f]{64}',source.get(k,'')) is None:raise ValueError('training hash missing')
    return record['runtime_row']

def validate_assets(spec,assets):
    row=validate(spec)
    record=_records()[spec.task_id+'/'+spec.difficulty+'/'+spec.instance_id]
    raw=assets.resolve('train_source_raw').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec.config['train_source']['input_sha256']:
        raise ValueError('training raw input SHA differs')
    document=json.loads(raw)
    if document.get('case_id')!=record['train_source']['native_case_id'] or document['config']['source_config']!=to_json(spec.config['source_config']):
        raise ValueError('CP10 training raw condition differs')
    if document.get('difficulty')!=spec.difficulty or document.get('split')!='train':
        raise ValueError('training raw task/tier/split differs')
    # Source task aliases (e.g. BRIDGE_CAR) must be individually reviewed in the catalogue.
    if document.get('task_id')!=record['raw_task_id']:
        raise ValueError('training raw task differs')
    for role,expected in record['runtime_asset_sha256'].items():
        if hashlib.sha256(assets.resolve(role).read_bytes()).hexdigest()!=expected:
            raise ValueError('training runtime asset SHA differs: '+role)
    return row


def admitted_config_sha256(instance_id,difficulty):
    records=_records()
    row=records.get('CP10/'+difficulty+'/'+instance_id)
    if row is None:
        if any('CP10/'+tier+'/'+instance_id in records for tier in ('easy','medium','hard')):
            raise ValueError('CP10 training scoring identity/tier differs')
        return None
    return row['runtime_row']['source_config_canonical_sha256']
