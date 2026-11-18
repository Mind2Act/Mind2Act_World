"""Training runtime metadata and source-condition readers."""
from importlib.resources import files
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import to_json,freeze

@lru_cache(maxsize=1)
def _records():
    raw=files(__package__).joinpath('config', 'training_source_records.json').read_bytes()
    value=json.loads(raw)
    if value['schema']!='mind2act_eval.training_source_records.v1':
        raise ValueError('training admission schema differs')
    return freeze(value['records'])

def validate(spec):
    """Resolve runtime metadata and keep the scene's time/protocol alignment."""
    if spec.task_id != 'CP09' or spec.clock.physics_hz != 120 or spec.clock.control_hz != 30:
        raise ValueError('CP09 requires its task and 120/30 Hz clock')
    return to_json(spec.config)


def validate_assets(spec, assets):
    """Require the resources used by the scene, without provenance receipts."""
    from mind2act_eval.sim.source_assets import source_file
    row = validate(spec)
    for role in ('station', 'training_task', 'source_case', 'x5'):
        source_file(assets, role)
    return row


def validate_evidence(evidence,source_case):
    values=evidence.values
    record=_records().get(evidence.task_id+'/'+values['difficulty']+'/'+evidence.instance_id)
    if (record is None or record['identity']['seed']!=values['seed']
            or 'test_source' in values or values.get('train_source')!=record['train_source']):
        raise ValueError('CP09 training scoring source identity differs')
    from mind2act_eval.tasks.CP09.instances import signature
    if signature(source_case)!=record['runtime_row']['source_case_sha256']:
        raise ValueError('CP09 training scoring effective case differs')
