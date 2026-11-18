"""Fresh per-world key calibration. Does not write physical poses or materials."""
from collections import deque
import math

PRESS_M = .002
RELEASE_M = .0005


class KeyPressPolicy:
    def __init__(self, keys, *, window=120, minimum_ticks=180, maximum_ticks=360, noise_limit_m=.0001):
        self.keys = frozenset(keys)
        if self.keys != frozenset(range(29)):
            raise ValueError('Expected all 29 visible white keys')
        if not 1 <= window <= minimum_ticks < maximum_ticks or not 0 < noise_limit_m < RELEASE_M:
            raise ValueError('Invalid calibration window')
        self.window = deque(maxlen=window)
        self.minimum_ticks = minimum_ticks
        self.maximum_ticks = maximum_ticks
        self.noise_limit_m = noise_limit_m
        self.samples_seen = 0
        self.baseline_depths_m = None
        self.noise_ranges_m = None
        self.calibrated_at_sample = None

    @property
    def ready(self):
        return self.baseline_depths_m is not None

    def checked(self, authored):
        if set(authored) != self.keys:
            raise ValueError('Missing or extra key measurement')
        values = {key: float(value) for key, value in authored.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Nonfinite key depth')
        return values

    def observe_idle(self, authored, *, clear):
        values = self.checked(authored)
        if self.ready:
            return
        self.samples_seen += 1
        if clear:
            self.window.append(values)
        else:
            self.window.clear()
        if self.samples_seen >= self.minimum_ticks and len(self.window) == self.window.maxlen:
            ranges = {key: max(row[key] for row in self.window) - min(row[key] for row in self.window)
                      for key in self.keys}
            means = {key: sum(row[key] for row in self.window) / len(self.window) for key in self.keys}
            if max(ranges.values()) <= self.noise_limit_m and all(-.001 <= value <= .0015 for value in means.values()):
                self.baseline_depths_m = means
                self.noise_ranges_m = ranges
                self.calibrated_at_sample = self.samples_seen
                self.window.clear()
        if not self.ready and self.samples_seen >= self.maximum_ticks:
            raise RuntimeError('Fresh idle key baseline did not qualify within 360 ticks')

    def relative_depths(self, authored):
        values = self.checked(authored)
        if not self.ready:
            raise RuntimeError('Key baseline is not calibrated')
        return {key: value - self.baseline_depths_m[key] for key, value in values.items()}

    def snapshot(self):
        # Export JSON object keys without changing integer-indexed calibration.
        return dict(ready=self.ready, samples_seen=self.samples_seen,
                    calibrated_at_sample=self.calibrated_at_sample,
                    baseline_depths_m=(None if self.baseline_depths_m is None else
                                       {str(key): value for key, value in self.baseline_depths_m.items()}),
                    idle_noise_ranges_m=(None if self.noise_ranges_m is None else
                                        {str(key): value for key, value in self.noise_ranges_m.items()}),
                    press_displacement_m=PRESS_M, release_displacement_m=RELEASE_M,
                    training_depth_only_green_aligned=False)
