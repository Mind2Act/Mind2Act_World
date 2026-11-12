"""Per-tick measurement record shared by PR1 and PR2 (the value is read after each physics step).

The runner samples evidence on every tick, so evidence carries only the trailing `window` measurements the
rule needs (a released hold ending at the last tick); the full series stays in the recorder and is
written to the attempt directory by `dump`. This keeps per-tick evidence O(window), not O(episode).
"""
import json
from pathlib import Path
from typing import Callable

from mind2act_eval.contracts import NativeEvidence


class Recorder:
    """One measured-value sample per physics tick; a skipped tick makes the record non-consecutive."""

    def __init__(self, read_value: Callable[[], float], read_touch: Callable[[], bool],
                 task_id: str, instance_id: str, target_value: float, tick_hz: int, window: int):
        self._read, self._touch = read_value, read_touch
        self._task, self._instance, self._target, self._hz, self._window = task_id, instance_id, target_value, tick_hz, window
        self._first = None
        self._values: list[float] = []
        self._touching: list[bool] = []
        self._consecutive = True

    def record(self, tick: int) -> float:
        """Idempotent per tick; ticks already recorded return the stored reading."""
        if self._first is None:
            self._first = tick
        index = tick - self._first
        if 0 <= index < len(self._values):
            return self._values[index]
        if index != len(self._values):
            self._consecutive = False
            raise ValueError(f"tick {tick} skips the measurement record")
        value = float(self._read())
        self._values.append(value)
        self._touching.append(bool(self._touch()))
        return value

    @property
    def last(self) -> float:
        return self._values[-1]

    def evidence(self, tick: int) -> NativeEvidence:
        if self._first is not None and self._first + len(self._values) - 1 != tick:
            self._consecutive = False  # asked for a tick that was not measured
        skipped = max(0, len(self._values) - self._window)
        return NativeEvidence(self._task, self._instance, tick, dict(
            tick_hz=self._hz, target_value=self._target, first_tick=(self._first or 0) + skipped,
            values=tuple(self._values[skipped:]), touching=tuple(self._touching[skipped:]),
            consecutive=self._consecutive))

    def released_hold_met(self, tolerance: float, hold_seconds: float) -> bool:
        """True once the trailing run (in tolerance, gripper off the device) spans hold_seconds."""
        from .stability import released_hold, window_ticks
        n = window_ticks(self._hz, hold_seconds)
        if len(self._values) < n or not self._consecutive:
            return False
        return released_hold(self._values[-n:], self._touching[-n:], self._target, tolerance, self._hz, hold_seconds)[0]

    def dump(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(task=self._task, instance=self._instance, tick_hz=self._hz,
                                        target_value=self._target, first_tick=self._first or 0,
                                        consecutive=self._consecutive, values=self._values, touching=self._touching)) + "\n")
