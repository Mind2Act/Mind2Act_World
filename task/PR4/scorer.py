"""PR4 terminal cloth shape, support and release scoring."""
from mind2act_eval.contracts import Score
from .instances import RULE_VERSION
from .evidence import reconstruct


def score(evidence):
    facts = reconstruct(evidence)
    return Score(100. if facts['success'] else 0., dict(
        task_success=facts['success'], rule_version=RULE_VERSION, scoring_version=RULE_VERSION,
        decision='Measured cloth coverage and corner placement', physical_facts=facts))
