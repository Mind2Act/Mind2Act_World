"""Training runtime metadata and source-condition readers."""
from importlib.resources import files
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import to_json,freeze

@lru_cache(maxsize=1)
def _records():
    raw=files('mind2act_eval.env_cfg').joinpath('cp07_training_source_records.json').read_bytes()
    value=json.loads(raw)
    if value['schema']!='mind2act_eval.training_source_records.v1':
        raise ValueError('training admission schema differs')
    return freeze(value['records'])

def validate(spec):
    """Resolve runtime metadata and keep the scene's time/protocol alignment."""
    source = spec.config.get('train_source')
    if source is None or source.get('schema') != 'mind2act_eval.training_source.v1' or source.get('split') != 'train':
        raise ValueError('training layout requires its source metadata')
    if 'test_source' in spec.config:
        raise ValueError('training and test inputs cannot be combined')
    key = spec.task_id + '/' + spec.difficulty + '/' + spec.instance_id
    record = _records().get(key)
    if record is None:
        raise ValueError('missing runtime metadata for ' + key)
    if record['identity']['clock'] != to_json(spec.clock):
        raise ValueError('training clock differs from the scene configuration')
    if (spec.task_id!='CP07' or spec.config_version!='cp07-source-binding-v2'
            or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30 or spec.clock.rgb_hz!=30
            or spec.config.get('cp07_mode')!='diagnostic_training'
            or list(spec.config['material_mdl_search_asset_ids'])!=['portable-mdl-0','portable-mdl-1','portable-mdl-2']):
        raise ValueError('CP07 training task protocol differs')
    return record['runtime_row']

def validate_assets(spec, assets):
    """Require the resources used by the scene, without provenance receipts."""
    from mind2act_eval.sim.source_assets import source_file
    row = validate(spec)
    for role in ('station', 'cp07_binding', 'x5'):
        source_file(assets, role)
    return row


def validate_evidence(evidence):
    v=evidence.values;record=_records().get(evidence.task_id+'/'+v['difficulty']+'/'+evidence.instance_id)
    if (record is None or record['identity']['seed']!=v.get('seed')
            or to_json(v.get('train_source'))!=record['train_source'] or 'test_source' in v
            or v.get('source_layout_sha256')!=record['runtime_row']['source_layout_sha256']
            or v.get('binding_mode')!='diagnostic_training' or v.get('solutions_scope')!='known-source-solutions'):
        raise ValueError('CP07 training scoring source identity differs')
