"""PR4 §6, with benchmark maintainer 2026-11-01 23:30 shape/release amendment."""
from mind2act_eval.contracts import Score
from .instances import RULE_VERSION
from .evidence import reconstruct


def score(evidence):
    facts = reconstruct(evidence)
    return Score(100. if facts['success'] else 0., dict(
        task_success=facts['success'], rule_version=RULE_VERSION, scoring_version=RULE_VERSION,
        decision='benchmark maintainer2026-11-01/23:30', physical_facts=facts))
