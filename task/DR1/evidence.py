"""Native DR1 measurements: parcel poses/velocities and PhysX contact classes, one row per physics tick.

Ported from source `cases_unified/FR1/evidence.py` (source_id fr1-retained-train-x5-120hz-v1, module
sha f890d88c8a1f family). The source classified 'unexpected_robot_contact' using the expert's current
target; that label is dropped. Contacts are recorded raw and classified only by geometry/link names.
"""
import re
from math import isfinite
from mind2act_eval.contracts import FrozenHistory

ITEM = re.compile(r'/World/FullStream/Goods_(\d\d)(?:/|$)')
FINGER_LINKS = ('link7', 'link8')


def item_ids(pair):
    return {int(m.group(1)) for path in pair if (m := ITEM.search(path))}


def has_link(path, link):
    return '/' + link + '/' in path or path.endswith('/' + link)


class ContactTable:
    """Accumulates PhysX contact reports for robot/parcel pairs; pure Python so CPU tests can drive it."""

    def __init__(self, package_roots):
        self.roots = tuple(package_roots)
        self.pairs = {}
        self.tick = 0

    def report(self, pair, active, impulse, separation, support_normal_z=None):
        pair = tuple(sorted(pair))
        if not any('/Robot' in p or '/World/FullStream/Goods_' in p for p in pair):
            return
        row = self.pairs.setdefault(pair, dict(active=False, max_impulse_Ns=0., min_separation_m=1.,
                                               first_tick=self.tick, last_tick=self.tick))
        row.update(active=bool(active), last_tick=self.tick,
                   max_impulse_Ns=max(row['max_impulse_Ns'], impulse),
                   min_separation_m=min(row['min_separation_m'], separation))
        if support_normal_z is not None:
            if not isfinite(support_normal_z):
                raise ValueError('nonfinite DR1 owned support normal')
            row['support_normal_z'] = float(support_normal_z)

    def active(self, index, *, token=None, fingers=False):
        root = self.roots[index]
        for pair, row in self.pairs.items():
            if not row['active'] or not any(root in p for p in pair):
                continue
            if token is not None and not any(token in p for p in pair):
                continue
            if fingers and not any('/Robot' in p and any(has_link(p, l) for l in FINGER_LINKS) for p in pair):
                continue
            return True
        return False

    def supports(self, index):
        root = self.roots[index]
        rows = [row for pair, row in self.pairs.items() if row['active']
                and any(p == root or p.startswith(root + '/') for p in pair)
                and any('/FR1/InspectionTray/' in p for p in pair)]
        if any('support_normal_z' not in row for row in rows):
            raise ValueError('missing DR1 owned tray support normal')
        return any(row['support_normal_z'] > 0. for row in rows)

    def summary(self):
        return [dict(pair=list(pair), **row) for pair, row in self.pairs.items()]


class Recorder:
    """Per-tick rows consumed by scorer.score. `bodies` expose get_world_pose/get_linear_velocity."""

    def __init__(self, bodies, dims, contacts, activity, read_extent, belt_velocity):
        self.bodies, self.dims, self.contacts = bodies, dims, contacts
        self.active, self._extent, self._belt = activity, read_extent, belt_velocity
        self.ticks = FrozenHistory()

    def record(self, tick):
        self.contacts.tick = tick
        parcels = []
        for i, body in enumerate(self.bodies):
            position, quaternion = body.get_world_pose()
            parcels.append(dict(
                active=bool(self.active[i]), position=[float(x) for x in position],
                extent=[float(x) for x in self._extent(quaternion, self.dims[i])],
                velocity=[float(x) for x in body.get_linear_velocity()],
                finger_contact=self.contacts.active(i, fingers=True),
                belt_contact=self.contacts.active(i, token='BeltContactSurface'),
                tray_contact=self.contacts.active(i, token='/FR1/InspectionTray/'),
                tray_supported=self.contacts.supports(i)))
        self.ticks = self.ticks.append(dict(tick=tick, belt_velocity=[float(x) for x in self._belt()], parcels=parcels))
