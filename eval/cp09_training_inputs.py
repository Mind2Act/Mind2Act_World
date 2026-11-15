"""Code-pinned training admission; raw inputs and H5 lineage remain mandatory."""
from importlib.resources import files
import hashlib,json,re
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import to_json,freeze
CATALOGUE_SHA256 = '031c8e55c728e1b3590f7faa21521b15a9bd979b5d02f260bcee3320b2ba3bc6'

@lru_cache(maxsize=1)
def _records():
    raw=files('mind2act_eval.env_cfg').joinpath('cp09_training_source_records.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CATALOGUE_SHA256:
        raise ValueError('training admission catalogue SHA differs')
    value=json.loads(raw)
    if value['schema']!='mind2act_eval.training_source_records.v1':
        raise ValueError('training admission schema differs')
    return freeze(value['records'])

def validate(spec):
    from mind2act_eval.public_layouts import validate as public_validate
    public_validate(spec)
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
    if record.get('public_runtime_only') is True:
        # Private original input/H5/condition SHA remains code-owned lineage.
        # Public verification covers these exact runtime assets and receipts;
        # omitted collector wrappers are deliberately not reconstructed here.
        for role,expected in record['runtime_asset_sha256'].items():
            if hashlib.sha256(assets.resolve(role).read_bytes()).hexdigest()!=expected:
                raise ValueError('public runtime asset SHA differs: '+role)
        proof=json.loads(assets.resolve('training_geometry_proof').read_text())
        source=spec.config['train_source']
        if (proof['task_id']!=spec.task_id or proof['difficulty']!=spec.difficulty
                or proof['instance_id']!=spec.instance_id
                or proof['source_input_sha256']!=source['input_sha256']):
            raise ValueError('public runtime/private source lineage differs')
        from mind2act_eval.tasks.CP09.instances import signature
        case=json.loads(assets.resolve('source_case').read_text())
        if (case['difficulty']!=spec.difficulty or case['seed']!=spec.seed
                or signature(case)!=spec.config['source_case_sha256']
                or proof['source_case_sha256']!=spec.config['source_case_sha256']
                or proof['h5_sha256']!=source['h5_sha256']
                or proof['authored_task_source_sha256']!=spec.config['authored_task_source_sha256']
                or proof['export_reopen_equal'] is not True or proof['initial_tools_unattached'] is not True
                or proof['initial_caps_closed'] is not True):
            raise ValueError('public CP09 runtime source case/Task receipt differs')
        return row
    raise ValueError('public runtime-only admission marker required')


def validate_evidence(evidence,source_case):
    values=evidence.values
    record=_records().get(evidence.task_id+'/'+values['difficulty']+'/'+evidence.instance_id)
    if (record is None or record['identity']['seed']!=values['seed']
            or 'test_source' in values or values.get('train_source')!=record['train_source']):
        raise ValueError('CP09 training scoring source identity differs')
    from mind2act_eval.tasks.CP09.instances import signature
    if signature(source_case)!=record['runtime_row']['source_case_sha256']:
        raise ValueError('CP09 training scoring effective case differs')
