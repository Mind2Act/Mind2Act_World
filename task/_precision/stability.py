"""Released-hold rule shared by PR1 (ohm, 0.5 s) and PR2 (volt, 1 s).

Authoritative text: Feishu snapshot 20261101 (PR1.md / PR2.md section 5-6, 2026-10-29 correction): the measured
value is within the tolerance of the target and the gripper no longer touches the device, continuously for
the hold time; time spent holding before release does not count. Binary 100/0. The window must run up to
the last sample (the episode ends as soon as it is met) and its length is the tick difference over tick_hz,
never a sample count.
"""
from math import ceil, isfinite

from mind2act_eval.contracts import NativeEvidence, Score

KEYS = frozenset(('tick_hz', 'target_value', 'first_tick', 'values', 'touching', 'consecutive'))


def window_ticks(tick_hz: int, hold_seconds: float) -> int:
    """Samples that span hold_seconds (endpoints included); evidence may carry just this trailing window."""
    return ceil(hold_seconds * tick_hz) + 1


def released_hold(values, touching, target, tolerance, tick_hz, hold_seconds) -> tuple[bool, float]:
    """Length (s) of the trailing run that is in tolerance and untouched, and whether it reaches hold_seconds."""
    start = None
    for index, (value, touched) in enumerate(zip(values, touching)):
        if abs(value - target) <= tolerance + 1e-12 and not touched:
            start = index if start is None else start
        else:
            start = None
    duration = (len(values) - 1 - start) / tick_hz if start is not None else 0.
    return duration >= hold_seconds - 1e-12, duration


def held_window(evidence: NativeEvidence, tolerance: float, hold_seconds: float) -> tuple[bool, float]:
    v = evidence.values
    if set(v) != KEYS:
        raise ValueError('precision evidence needs exactly ' + ', '.join(sorted(KEYS)))
    hz, target, values, touching = v['tick_hz'], v['target_value'], v['values'], v['touching']
    if type(hz) is not int or hz <= 0 or not isinstance(target, (int, float)) or not isfinite(target):
        raise ValueError('invalid tick_hz/target_value')
    if v['consecutive'] is not True:
        raise ValueError('measurements must cover consecutive physics ticks')
    if not values or len(values) != len(touching) or not all(isinstance(x, (int, float)) and isfinite(x) for x in values):
        raise ValueError('missing or nonfinite measurement')
    if not all(t in (0, 1, True, False) for t in touching):
        raise ValueError('touch flags must be boolean')
    return released_hold(values, touching, target, tolerance, hz, hold_seconds)


def binary_score(evidence: NativeEvidence, tolerance: float, hold_seconds: float) -> Score:
    goal, duration = held_window(evidence, tolerance, hold_seconds)
    return Score(100. if goal else 0., dict(released_stable_sim_seconds=duration, required_sim_seconds=hold_seconds,
                                            tolerance=tolerance, credited_units=int(goal), total_units=1))
