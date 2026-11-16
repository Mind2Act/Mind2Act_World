"""Training runtime metadata and source-condition readers."""
from importlib.resources import files
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import to_json,freeze

@lru_cache(maxsize=1)
def _records():
    raw=files('mind2act_eval.env_cfg').joinpath('cp08_training_source_records.json').read_bytes()
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
    return record['runtime_row']

def validate_assets(spec, assets):
    """Require the resources used by the scene, without provenance receipts."""
    from mind2act_eval.sim.source_assets import source_file
    row = validate(spec)
    for role in ('station', 'cue_plan', 'cue_idle', 'x5'):
        source_file(assets, role)
    return row
