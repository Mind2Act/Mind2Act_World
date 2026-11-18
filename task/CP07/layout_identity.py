"""Public spec/index authentication; contains no solution or raw geometry reads."""
import hashlib
import json
from pathlib import Path

MODE = 'train_layout_test'
SOURCE = 'cp07-training-layout-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_identity(spec):
    raise ValueError('CP07 has no admitted public clean layouts in this candidate')
