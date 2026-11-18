"""E1: all N correct = 100; otherwise the tier cap times the correct batch fraction."""
from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION, partial_score

N_BY_DIFFICULTY = {"easy": 2, "medium": 4, "hard": 6}


def score(evidence: NativeEvidence) -> Score:
    values = evidence.values
    difficulty = values["difficulty"]
    if difficulty not in N_BY_DIFFICULTY:
        raise ValueError("E1 unknown difficulty: %r" % (difficulty,))
    n = N_BY_DIFFICULTY[difficulty]
    seen, credited = set(), 0
    for row in values["rounds"]:
        index = row["round"]
        if type(index) is not int or not 1 <= index <= n or index in seen:
            raise ValueError("E1 sealed batch index must be unique and within 1..%d" % n)
        if type(row["pass"]) is not bool:
            raise ValueError("E1 batch pass must be boolean")
        seen.add(index)
        credited += row["pass"]
    missing = n - credited
    success = credited == n
    value = 100 if success else partial_score(credited, n, {'easy':30.,'medium':22.,'hard':15.}[difficulty])
    return Score(float(value), {"scoring_version": SCORING_VERSION, "credited_units": credited, "total_units": n, "success": success,
                                "sealed_batches": len(seen), "difficulty": difficulty})
