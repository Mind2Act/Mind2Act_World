"""Bounded online equivalent of the pinned ImprintTracker.

The batch tracker shares the corrected duration comparison as the CPU oracle. Keep the first
force maximum (source max() tie behavior), displacement/twist from the first
segment sample, both endpoint times and only the first two marks: one versus
multiple is the task's zero-priority distinction. The exact count is retained.
"""
import math

from .imprint import Imprint, ImprintTracker, _ang, duration_at_least


class StreamingImprints:
    def __init__(self):
        self.limits = ImprintTracker()
        self.first = self.last = self.peak = None
        self.slide = self.twist = 0.
        self.count = 0
        self.first_two = []

    def _mark(self):
        if self.first is None or not duration_at_least(self.first.t,self.last.t,self.limits.t_min):
            return None
        return Imprint(self.peak.x, self.peak.y, self.peak.yaw_deg, self.last.t - self.first.t,
                       self.slide > self.limits.smear_m or self.twist > self.limits.smear_yaw_deg)

    def update(self, frame):
        if frame.force_n >= self.limits.f_min and frame.tilt_deg <= self.limits.tilt_max_deg:
            if self.first is None:
                self.first = self.peak = frame
                self.slide = self.twist = 0.
            elif frame.force_n > self.peak.force_n:
                self.peak = frame
            self.last = frame
            self.slide = max(self.slide, math.hypot(frame.x - self.first.x, frame.y - self.first.y))
            self.twist = max(self.twist, _ang(frame.yaw_deg, self.first.yaw_deg))
            return []
        mark = self._mark()
        self.first = self.last = self.peak = None
        if mark is None:
            return []
        self.count += 1
        if len(self.first_two) < 2:
            self.first_two.append(mark)
        return [mark]

    def preview(self):
        """Source finish() at this cutoff, without ending an open contact segment."""
        mark = self._mark()
        count = self.count + int(mark is not None)
        first_two = list(self.first_two)
        if mark is not None and len(first_two) < 2:
            first_two.append(mark)
        return count, first_two
