"""PR5 stamp imprint tracking from per-frame contact readings; reusable by CP12.

Simulator-agnostic: the Isaac side feeds one Frame per physics step.
"""
from dataclasses import dataclass, field
import math


@dataclass
class Frame:
    t: float            # seconds
    force_n: float      # stamp-face normal force on the card
    x: float            # stamp-face center in card frame, metres
    y: float
    yaw_deg: float      # stamp arrow direction in card frame
    tilt_deg: float     # angle between stamp-face normal and card normal


@dataclass
class Imprint:
    x: float
    y: float
    yaw_deg: float
    duration_s: float
    smeared: bool


def _ang(a, b):
    return abs((a - b + 180) % 360 - 180)


def duration_at_least(first_s, last_s, minimum_s):
    """Compare sampled durations without losing the nominal 0.1 s endpoint.

    One nanosecond only absorbs clock subtraction roundoff, far below one
    120 Hz interval. It does not admit a segment that is one sample short.
    """
    elapsed = last_s-first_s
    return elapsed >= minimum_s or math.isclose(elapsed,minimum_s,rel_tol=0.,abs_tol=1e-9)


@dataclass
class ImprintTracker:
    f_min: float = 5.0
    tilt_max_deg: float = 10.0
    t_min: float = 0.1
    smear_m: float = 0.003
    smear_yaw_deg: float = 5.0
    imprints: list = field(default_factory=list)
    _seg: list = field(default_factory=list)

    def update(self, f: Frame):
        if f.force_n >= self.f_min and f.tilt_deg <= self.tilt_max_deg:
            self._seg.append(f)
        else:
            self._close()

    def finish(self):
        self._close()
        return self.imprints

    def _close(self):
        seg, self._seg = self._seg, []
        if not seg or not duration_at_least(seg[0].t,seg[-1].t,self.t_min):
            return  # light touch leaves no imprint
        peak = max(seg, key=lambda f: f.force_n)
        slide = max(math.hypot(f.x - seg[0].x, f.y - seg[0].y) for f in seg)
        twist = max(_ang(f.yaw_deg, seg[0].yaw_deg) for f in seg)
        self.imprints.append(Imprint(peak.x, peak.y, peak.yaw_deg, seg[-1].t - seg[0].t,
                                     slide > self.smear_m or twist > self.smear_yaw_deg))


def judge(imprints, box_center, box_wh, face_side, target_yaw_deg, yaw_tol_deg,
          stamp_home, arm_clear):
    """PR5 success: exactly one clean imprint fully inside box with correct heading."""
    out = dict(imprint_count=len(imprints), single=len(imprints) == 1,
               inside=False, heading=False, clean=False,
               stamp_home=bool(stamp_home), arm_clear=bool(arm_clear))
    if out['single']:
        m = imprints[0]
        r, c = math.radians(m.yaw_deg), [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        h = face_side / 2
        corners = [(m.x + h * (sx * math.cos(r) - sy * math.sin(r)),
                    m.y + h * (sx * math.sin(r) + sy * math.cos(r))) for sx, sy in c]
        out['inside'] = all(abs(px - box_center[0]) <= box_wh[0] / 2 and
                            abs(py - box_center[1]) <= box_wh[1] / 2 for px, py in corners)
        out['heading'] = _ang(m.yaw_deg, target_yaw_deg) <= yaw_tol_deg
        out['clean'] = not m.smeared
    out['success'] = all(out[k] for k in ('single', 'inside', 'heading', 'clean', 'stamp_home', 'arm_clear'))
    return out
