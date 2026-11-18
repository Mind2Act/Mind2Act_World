"""P1: compare all terminal supported ingredients with accumulated published demand."""
from collections import Counter
from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION

N_BY_DIFFICULTY = {"easy": 4, "medium": 5, "hard": 7}
MIN_STABLE_S = 1.0
MAX_RANGE_M = .003


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError("P1 %s must be boolean" % name)
    return value


def _int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("P1 %s must be a non-negative integer" % name)
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        raise ValueError("P1 %s must be a non-negative number" % name)
    return float(value)


def _valid(row):
    for key in ("inside", "support", "released"):
        _bool(row[key], key)
    stable = _number(row["stable_seconds"], "stable_seconds")
    spread = _number(row["position_range_m"], "position_range_m")
    return row["inside"] and row["support"]


def score(evidence: NativeEvidence) -> Score:
    e = evidence.values
    difficulty = e["difficulty"]
    if difficulty not in N_BY_DIFFICULTY:
        raise ValueError("P1 unknown difficulty: %r" % (difficulty,))
    n = N_BY_DIFFICULTY[difficulty]
    demand = dict(e["published_demand"])
    for key, value in demand.items():
        if type(key) is not str or len(key.split("/")) != 2:
            raise ValueError("P1 container/type demand key required")
        _int(value, "cumulative demand")
    if sum(demand.values()) > n:
        raise ValueError("P1 published demand exceeds fixed total")
    if not _bool(e["occupancy_complete"], "occupancy_complete") or e.get("ambiguous_membership"):
        raise ValueError("P1 occupancy evidence is incomplete or ambiguous")
    published = _bool(e["all_orders_published"], "all_orders_published")
    if published and sum(demand.values()) != n:
        raise ValueError("P1 published cumulative denominator mismatch")
    rows, seen = list(e["occupancy"]), set()
    for row in rows:
        ident = row.get("object_id")
        if type(ident) is not str or not ident or ident in seen:
            raise ValueError("P1 duplicate/missing object identity")
        seen.add(ident)
        _int(row["entered_tick"], "entered_tick")
        if type(row.get("container")) is not str or type(row.get("kind")) is not str:
            raise ValueError("P1 occupancy identity missing")
        _valid(row)  # validate every row, including excess ones
    counts = Counter(row['container'] + '/' + row['kind'] for row in rows if _valid(row))
    c = sum(min(counts.get(k, 0), v) for k, v in demand.items())
    missing = n-c
    wrong = sum(v for k, v in counts.items() if k not in demand)
    extra = sum(max(0, v-demand[k]) for k, v in counts.items() if k in demand)
    defects = missing + wrong + extra
    success = bool(published and demand and defects == 0)
    cap = {'easy':30.,'medium':22.,'hard':15.}[difficulty]
    value = 100 if success else {1: cap/2, 2: cap/4}.get(defects, 0)
    return Score(float(value), dict(scoring_version=SCORING_VERSION, credited_units=c,
        total_units=n, success=success, missing=missing, wrong=wrong,
        extra=extra, defects=defects, all_orders_published=published))
