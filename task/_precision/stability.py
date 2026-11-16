"""Measurement keys and released-hold termination helpers for PR1/PR2.

The scorer reads the final T0 value; these hold checks only control scene termination.
"""
from math import ceil


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
