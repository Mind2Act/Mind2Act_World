"""Native M1 contact table and per-tick rows (source: attach() contacts + development world `measured`)."""
import math
from collections import defaultdict

import numpy as np

from mind2act_eval.contracts import FrozenHistory

CONTACT_EPS = 1e-7


def under(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


from mind2act_eval.sim.contacts import ContactLedger


class ContactTable(ContactLedger):
    def report(self, pair, active, impulse, separation, normals=None):
        if active and normals == [] and tuple(sorted(pair)) in self.pairs:
            return
        self.record(*pair, active=active, impulse_ns=impulse, separation_m=separation,
                    normals_on_collider0=normals)


class Recorder:
    """Builds one row per physics tick from handles injected by the scene (so tests can use fakes)."""

    def __init__(self, paths, centers, target_velocity, contacts, support_pads, covers, gripper_open):
        self.paths, self.centers, self.velocity = paths, centers, target_velocity
        self.contacts, self.pads, self.covers, self.open = contacts, support_pads, covers, gripper_open
        self.ticks = FrozenHistory()

    def _finger(self, ident):
        return any(self.contacts.touch(self.paths[ident], other='/World/Robot' + s, link=f)
                   for s in ('Left', 'Right') for f in ('link7', 'link8'))

    def record(self, tick, target):
        lin, ang = self.velocity()
        row = dict(tick=tick, target_center=[float(x) for x in self.centers(target)],
                   target_linear_velocity=[float(x) for x in lin], target_angular_velocity=[float(x) for x in ang],
                   target_finger=self._finger(target),
                   target_robot_contact=any(self.contacts.touch(self.paths[target], other='/World/Robot' + s)
                                            for s in ('Left', 'Right')),
                   cover_centers={c: [float(x) for x in self.centers(c)] for c in self.covers},
                   cover_fingers={c: self._finger(c) for c in self.covers},
                   gripper_open=[bool(x) for x in self.open()],
                   station_support={k: self.contacts.supported(self.paths[target], other=pad) for k, pad in self.pads.items()})
        if not np.isfinite(row['target_center']).all():
            raise ValueError('target pose is not finite')
        self.ticks = self.ticks.append(row)
        return row
