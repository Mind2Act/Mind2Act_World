"""Autonomous contact/release mechanism on the source 120 Hz physics clock.

Source mechanism digest: 2d5a516d. Durations/angles preserve that mechanism;
section 5 now requires a complete physical press-release before selection.
Construction happens after the source two-second initialization settle. There
is no external enable flag, teacher, expert phase, or policy-side state access.
"""
from copy import deepcopy
from math import acos, isfinite


def angle_error_deg(q, target):
    def unit(values):
        if len(values) != 4 or not all(type(x) in (int, float) and isfinite(x) for x in values):
            raise ValueError("M2 physical quaternion invalid")
        length = sum(x*x for x in values) ** .5
        if abs(length - 1.) > 1e-4:
            raise ValueError("M2 physical quaternion is not normalized")
        return [x/length for x in values]
    a, b = unit(q), unit(target)
    return 2 * acos(min(1., abs(sum(x*y for x, y in zip(a, b))))) * 180 / 3.141592653589793


class MemoryMechanism:
    def __init__(self, identities):
        self._identities = tuple(identities)
        self.count = len(identities)
        self.desired = [178.] * self.count
        self._dwell = [0] * self.count
        self._latched = [False] * self.count
        self._eligible = [False] * self.count
        self.open = []
        self.matched = set()
        self.pairs = []
        self.events = []
        self.flip_until = 0
        self.pair_deadline = self.mismatch_deadline = None
        self.tick = 0
        self.max_unmatched_faceup = 0

    def ready(self):
        return self.pair_deadline is None and self.mismatch_deadline is None and self.tick >= self.flip_until

    def _select(self, i):
        accepted = self.ready() and i not in self.matched and i not in self.open
        self.events.append(dict(kind="press_release", tick=self.tick, card=i, accepted=accepted))
        if not accepted:
            return
        self.desired[i] = 2.
        self.flip_until = self.tick + 84
        self.open.append(i)
        if len(self.open) == 2:
            if self._identities[self.open[0]] == self._identities[self.open[1]]:
                self.pair_deadline = self.tick + 108
            else:
                self.mismatch_deadline = self.tick + 540

    def step(self, tick, depths_m, fingers, faceup, unmatched_faceup):
        if tick != self.tick + 1:
            raise ValueError("M2 requires every consecutive physics tick")
        if any(len(values) != self.count for values in (depths_m, fingers, faceup)):
            raise ValueError("M2 physical card/button evidence missing")
        if (any(type(x) not in (int, float) or not isfinite(x) for x in depths_m)
                or any(type(x) is not bool for x in (*fingers, *faceup))
                or type(unmatched_faceup) is not int or not 0 <= unmatched_faceup <= self.count):
            raise ValueError("M2 physical samples invalid")
        self.tick = tick
        self.max_unmatched_faceup = max(self.max_unmatched_faceup, unmatched_faceup)
        # A completed flip/result is ready on this tick, before release events.
        if self.mismatch_deadline is not None and tick >= self.mismatch_deadline:
            for i in self.open:
                self.desired[i] = 178.
            self.events.append(dict(kind="mismatch_close", tick=tick, cards=self.open[:]))
            self.open.clear()
            self.mismatch_deadline = None
            self.flip_until = tick + 84
        if (self.pair_deadline is not None and tick >= self.pair_deadline
                and all(faceup[i] for i in self.open)):
            pair = dict(cards=self.open[:], valid_press_release=True,
                        same_picture=True, physical_faceup=True)
            self.pairs.append(pair)
            self.matched.update(self.open)
            self.events.append(dict(kind="pair_matched", tick=tick, cards=self.open[:]))
            self.open.clear()
            self.pair_deadline = None

        for i in range(self.count):
            depth, touch = depths_m[i], fingers[i]
            self._dwell[i] = self._dwell[i] + 1 if depth > .001 else 0
            if self._dwell[i] >= 1 and not self._latched[i]:
                self._latched[i] = True
                self._eligible[i] = i not in self.matched and i not in self.open
            if depth < .0003:
                if self._latched[i] and self._eligible[i]:
                    self._select(i)
                self._latched[i] = self._eligible[i] = False

    def pair_receipts(self):
        return deepcopy(self.pairs)


def hinge_angle_deg(body_q, frame0=(1.,0.,0.,0.), frame1=(1.,0.,0.,0.)):
    """Actual Y hinge twist between authored joint frames, not world orientation."""
    from scipy.spatial.transform import Rotation
    from math import atan2, degrees
    def rot(q):
        angle_error_deg(q, q)
        return Rotation.from_quat([*q[1:], q[0]])
    relative = rot(frame0).inv() * rot(body_q) * rot(frame1)
    x, y, z, w = relative.as_quat()
    return (degrees(2*atan2(y, w))+180) % 360-180
