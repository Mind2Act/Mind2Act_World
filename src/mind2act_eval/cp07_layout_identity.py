"""Public spec/index authentication; contains no solution or raw geometry reads."""
import hashlib
import json
from pathlib import Path

MODE = 'train_layout_test'
SOURCE = 'cp07-training-layout-v1'
PROTOCOL = 'mind2act-common_input-coding-cp07-training-v3'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_identity(spec):
    from mind2act_eval.contracts import to_json
    authority = json.loads(Path(__file__).parent.joinpath('tasks/CP07/config/train-layout-authority.json').read_text())
    row = authority['cases'].get(spec.instance_id)
    if (spec.task_id != 'CP07' or spec.source_id != SOURCE or spec.config.get('cp07_mode') != MODE
            or row is None or spec.seed != row['seed'] or spec.difficulty != row['difficulty']
            or digest(to_json(spec.config)) != row['config_sha256']
            or digest(dict(spec.assets)) != row['assets_sha256']):
        raise ValueError('CP07 training-layout source identity differs from frozen authority')
    return authority, row
