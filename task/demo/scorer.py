from mind2act_eval.contracts import Score


def score(evidence):
    return Score(100. if evidence.values["reached"] else 0.)
