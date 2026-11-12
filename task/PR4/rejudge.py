"""Versioned offline scoring of preserved native audit evidence; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path
from mind2act_eval.contracts import NativeEvidence, to_json
from .instances import RULE_VERSION
from .scorer import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.evidence.read_bytes()
    evidence = NativeEvidence(**json.loads(raw))
    result = score(evidence)
    directory = Path(__file__).parent
    modules = ['scorer.py', 'evidence.py', 'instances.py', 'mechanism/geometry.py',
               'mechanism/fold_shapes.py', 'mechanism/fold_layout.py',
               'mechanism/hexagon.py', 'mechanism/clearance.py']
    record = dict(kind='offline_native_audit_rejudge', simulation_rerun=False,
                  source_evidence_sha256=hashlib.sha256(raw).hexdigest(),
                  rule_version=RULE_VERSION,
                  scorer_sha256=hashlib.sha256(Path(__file__).with_name('scorer.py').read_bytes()).hexdigest(),
                  implementation_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                         for name in modules},
                  score=to_json(result))
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
