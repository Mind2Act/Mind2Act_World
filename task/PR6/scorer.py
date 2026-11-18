"""Current PR6 §6, highest eligible stage; no ring is a valid numeric failure."""
from mind2act_eval.contracts import Score
from .evidence import reconstruct
RULE_VERSION = 'mind2act-pr6-scoring-20261118-stable-setup-authorized-push-v1'


def score(evidence):
    facts = reconstruct(evidence)
    cap = {'easy':30.,'medium':22.,'hard':15.}[evidence.values['config']['difficulty']]
    value = {0:0.,10:cap/2,20:cap,100:100.}[facts['score']]
    return Score(value,dict(task_success=facts['success'],rule_version=RULE_VERSION,scoring_version=RULE_VERSION,physical_facts=facts))
