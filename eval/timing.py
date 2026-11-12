"""Bounded monotonic wall-time counters; no device synchronization or I/O."""
from contextlib import contextmanager
import time


class PhaseTimings:
    def __init__(self):
        self._phases = {}

    @contextmanager
    def measure(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            row = self._phases.setdefault(name, dict(calls=0, wall_s=0., max_s=0.))
            row['calls'] += 1
            row['wall_s'] += elapsed
            row['max_s'] = max(row['max_s'], elapsed)

    def snapshot(self):
        return {name: dict(row) for name, row in self._phases.items()}


def phases_for(owner):
    phases = getattr(owner, '_phase_timings', None)
    if phases is None:
        phases = owner._phase_timings = PhaseTimings()
    return phases
