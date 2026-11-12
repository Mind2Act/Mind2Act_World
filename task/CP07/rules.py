"""Source-specific physical criteria, independent of policy or expert timing."""
from math import isfinite

L_ROAD_RULE = "source_l_road_max_0.10rad"
NO_ANGLE_RULE = "not_applicable"
L_ROAD_MAX_RAD = .10


class IncompleteEvidence(ValueError):
    """A required physical fact is unknown, so no numerical score is justified."""


def rule_for_kind(kind):
    if not isinstance(kind, str) or not kind:
        raise ValueError("CP07 source piece kind missing")
    return L_ROAD_RULE if kind == "L_road" else NO_ANGLE_RULE


def orientation_matches(row):
    kind = row.get("kind")
    rule = row.get("orientation_rule")
    if kind is None or rule is None:
        raise IncompleteEvidence("CP07 source orientation applicability missing")
    expected = rule_for_kind(kind)
    if rule != expected:
        raise ValueError("CP07 orientation rule differs from source kind")
    error = row.get("orientation_error_rad")
    if rule == NO_ANGLE_RULE:
        if error is not None:
            raise ValueError("CP07 non-L_road must explicitly have no angular criterion")
        return True  # Explicitly not applicable; not an unknown predicate.
    if error is None:
        raise IncompleteEvidence("CP07 required L_road angular measurement missing")
    if type(error) not in (int, float) or not isfinite(error) or error < 0:
        raise ValueError("CP07 invalid L_road angular measurement")
    return error <= L_ROAD_MAX_RAD


def placed(row):
    """Conjunction with missing-evidence propagation and known-failure dominance.

    Unknown never supplies success or zero. A separately proven failed criterion
    already proves this step failed, without requiring an unrelated missing fact.
    """
    known, unknown = [], []
    if not isinstance(row.get("piece_id"), str) or not row["piece_id"]:
        raise IncompleteEvidence("CP07 assigned piece identity missing")
    for key, threshold, above in (("lift_m", .035, True),
                                  ("bilateral_fraction", .80, True),
                                  ("position_error_m", .012, False)):
        value = row.get(key)
        if value is None:
            unknown.append(key)
            continue
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("CP07 invalid physical measurement: " + key)
        if key == "bilateral_fraction" and value > 1:
            raise ValueError("CP07 invalid bilateral fraction")
        known.append(value > threshold if above else value < threshold)
    for key in ("released", "supported"):
        value = row.get(key)
        if value is None:
            unknown.append(key)
        elif type(value) is not bool:
            raise ValueError("CP07 physical criterion must be boolean: " + key)
        else:
            known.append(value)
    try:
        known.append(orientation_matches(row))
    except IncompleteEvidence as error:
        unknown.append(str(error))
    if False in known:
        return False
    if unknown:
        raise IncompleteEvidence("CP07 required placement evidence missing: " + ", ".join(unknown))
    return True
