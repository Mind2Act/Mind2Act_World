"""CP03 pure evidence helpers: stable-placement tracker and dish target predicate.

Source: shared coupling module (physco_eval.coupling_v2 StablePlacement / dish_target), the
tracker as used by cp03-em-b-20261028 cases_unified/CP03/world.py.
"""
from collections import deque

import numpy as np


class StablePlacement:
    def __init__(self, ticks=120, tolerance=.003, require_manipulation=True):
        self.ticks=ticks;self.tolerance=tolerance;self.positions=deque(maxlen=ticks)
        self.grasped=False;self.lifted=False;self.require_manipulation=require_manipulation

    def update(self, position, *, bilateral, elevated, released, supported, correct):
        self.grasped |= bool(bilateral)
        # Actual free gripper transport is measured by the scene; height alone
        # cannot turn a supported push into a carried plate.
        self.positions.clear()
        if supported and correct:
            self.positions.append(np.asarray(position).copy())

    @property
    def complete(self):
        return bool(self.positions)


def dish_target(relative_x, height, normal_x, slot_center, slot_width, seated_height):
    # Real cradle support is checked separately. The receiving gap bounds the
    # center along the rack axis; it adds no height or orientation tolerance.
    return bool(abs(relative_x-slot_center) <= slot_width/2)


def quat_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_xyz_deg(q):
    """scipy Rotation.as_euler('xyz', degrees=True) for non-gimbal poses."""
    r = quat_matrix(q)
    sy = float(np.clip(-r[2, 0], -1., 1.))
    b = np.arcsin(sy)
    a = np.arctan2(r[2, 1], r[2, 2])
    c = np.arctan2(r[1, 0], r[0, 0])
    return np.degrees([a, b, c])
