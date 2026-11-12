"""Shared contracts from reviews/final-version-20261105/FINAL-VERSION.md.

Callers supply measured support and actual authored container bounds. This
module adds no geometric tolerance and makes no inference from gripper release.
"""
from dataclasses import dataclass
from math import isfinite

SCORING_VERSION = 'mind2act-final-scoring-20261105'

# FINAL-VERSION: result tasks stop after a complete model action succeeds;
# event tasks stop when their public sequence is finished, including errors.
EVENT_SEQUENCE_TASKS = frozenset({'DR2', 'E1', 'E2', 'F2', 'M1', 'CP04', 'CP05', 'CP10'})
RESULT_TASKS = frozenset({
    'PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'DR1', 'DR3', 'DR4',
    'S1', 'S2', 'P1', 'P2', 'F1', 'M2', 'CP01', 'CP02', 'CP03',
    'CP06', 'CP07', 'CP08', 'CP09',
})


def container_contains(*, supported, center, lower, upper):
    """Real target support plus center in the container's actual cavity bounds.

    Use XY for an open tray, or XYZ when an authored cavity has vertical bounds.
    Non-rectangular cavities must supply their own exact center-inside test.
    """
    if not len(center) == len(lower) == len(upper) or len(center) == 0:
        raise ValueError('matching nonempty center and bounds required')
    if not all(isfinite(v) for values in (center, lower, upper) for v in values):
        raise ValueError('finite physical coordinates required')
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError('container bounds must be ordered')
    return bool(supported) and all(lo <= p <= hi for p, lo, hi in zip(center, lower, upper))


def lcs_length(actual, expected):
    """Count matching sequence items without changing their relative order.

    Full success must separately require the exact full sequence, not just LCS.
    """
    row = [0] * (len(expected) + 1)
    for event in actual:
        previous = 0
        for j, target in enumerate(expected, 1):
            saved = row[j]
            row[j] = previous + 1 if event == target else max(row[j], row[j - 1])
            previous = saved
    return row[-1]


def partial_score(correct, total, ceiling=20):
    """Unrounded partial credit; full success is a separate task decision."""
    if not all(isfinite(v) for v in (correct, total, ceiling)):
        raise ValueError('finite score counts required')
    if total <= 0 or not 0 <= correct <= total or not 0 <= ceiling <= 100:
        raise ValueError('invalid partial score counts or ceiling')
    return ceiling * correct / total


@dataclass
class PressEdge:
    """One event on crossing the mechanism threshold; rearm after return."""
    threshold: float
    pressed: bool = False

    def __post_init__(self):
        if not isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('positive finite mechanism threshold required')

    def update(self, displacement):
        if not isfinite(displacement):
            raise ValueError('finite mechanism displacement required')
        now = displacement >= self.threshold
        edge = now and not self.pressed
        self.pressed = now
        return edge


CP_PARTIAL_CAPS = {'easy': 15, 'medium': 10, 'hard': 7}


def cp_score_from_completion(success, k, n, *, difficulty, components=None):
    """CP partial credit counts core units; setup gates remain in success."""
    from mind2act_eval.contracts import Score, score_from_completion
    checked = score_from_completion(success, k, n, components=components)
    ceiling = CP_PARTIAL_CAPS[difficulty]
    return Score(100.0 if success else partial_score(k, n, ceiling),
                 dict(checked.components, partial_ceiling=ceiling, core_k=k, core_N=n))
