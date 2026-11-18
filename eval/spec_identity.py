"""Canonical identity of an admitted development specification."""
import hashlib
import json
from mind2act_eval.contracts import to_json

def spec_digest(spec):
    return hashlib.sha256(json.dumps(to_json(spec), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
