"""Source CP09 bilateral, >40 mm, 24 consecutive physics-tick pickup proof.

This observes measured contacts and displacement; it never attaches or moves a
tool and never changes robot geometry, friction, or drive parameters.
"""
from math import isfinite


class PickupWitness:
    def __init__(self, tool, color=0):
        if tool not in ("eraser", "pen"):
            raise ValueError("unknown CP09 pickup tool")
        if type(color) is not int or (tool == "pen" and color < 1) or (tool == "eraser" and color != 0):
            raise ValueError("CP09 pickup colour mismatch")
        self.tool, self.color = tool, color
        self.last_tick = None
        self.consecutive = 0
        self.verified = False

    def sample(self, tick, finger_contacts, lift_m, *, attachment_before_lift=False):
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick + 1):
            raise ValueError("CP09 pickup requires contiguous physics ticks")
        if (len(finger_contacts) != 2 or any(type(v) is not bool for v in finger_contacts)
                or type(attachment_before_lift) is not bool or not isfinite(lift_m)):
            raise ValueError("CP09 pickup measurements missing")
        self.last_tick = tick
        good = all(finger_contacts) and lift_m > .04 and not attachment_before_lift
        self.consecutive = self.consecutive + 1 if good else 0
        if self.verified or self.consecutive < 24:
            return None
        self.verified = True
        return dict(tick=tick, tool=self.tool, color_id=self.color,
                    two_finger_contact=True, lift_m=float(lift_m),
                    consecutive_lift_ticks=self.consecutive,
                    attachment_joint_created_before_lift=False)
