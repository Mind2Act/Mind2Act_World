from mind2act_eval.scoring import SCORING_VERSION
"""PR5 highest eligible stage from native physical measurements, on a 0–100 scale.

Authority: jclin 2026-11-04 16:10: imprint plus terminal stamp return only.
The source imprint and pose thresholds remain pinned in config/source.json.
"""
from mind2act_eval.contracts import Score
from .evidence import reconstruct

RULE_VERSION = SCORING_VERSION


def score(evidence):
    facts = reconstruct(evidence)
    cap = {'easy':30.,'medium':22.,'hard':15.}[evidence.values['source_case']['level']]
    if not all(facts[key] for key in ('single', 'inside', 'clean')):
        value = 0.
    elif not facts['heading']:
        value = cap/2
    elif facts['success']:
        value = 100.
    else:
        value = cap
    return Score(value, dict(task_success=value == 100., rule_version=RULE_VERSION, scoring_version=RULE_VERSION,
                             stage=value, physical_facts=facts))
