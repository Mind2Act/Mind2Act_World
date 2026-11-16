"""Training runtime metadata and source-condition readers."""
from importlib.resources import files
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.contracts import to_json,freeze

@lru_cache(maxsize=1)
def _records():
    raw=files('mind2act_eval.env_cfg').joinpath('cp10_training_source_records.json').read_bytes()
    value=json.loads(raw)
    if value['schema']!='mind2act_eval.training_source_records.v1':
        raise ValueError('training admission schema differs')
    return freeze(value['records'])

def validate(spec):
    """Resolve runtime metadata and keep the scene's time/protocol alignment."""
    if spec.task_id != 'CP10' or spec.clock.physics_hz != 120 or spec.clock.control_hz != 30:
        raise ValueError('CP10 requires its task and 120/30 Hz clock')
    return to_json(spec.config)


def validate_assets(spec, assets):
    """Require the resources used by the scene, without provenance receipts."""
    from mind2act_eval.sim.source_assets import source_file
    row = validate(spec)
    for role in ('scene', 'x5'):
        source_file(assets, role)
    return row


def admitted_config_sha256(instance_id,difficulty):
    records=_records()
    row=records.get('CP10/'+difficulty+'/'+instance_id)
    if row is None:
        if any('CP10/'+tier+'/'+instance_id in records for tier in ('easy','medium','hard')):
            raise ValueError('CP10 training scoring identity/tier differs')
        return None
    return row['runtime_row']['source_config_canonical_sha256']
