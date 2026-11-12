"""Native contact measurements extracted from the selected CP01 backend.

source_id: cp01-shared-backend-legacy-run925de9 / shared backend world.py.
Retains its impulse/separation thresholds and directed flange tilt convention;
excludes the old attach/world/controller wrapper and all expert hooks.
"""
from collections import defaultdict
import math


class ContactLedger:
    def __init__(self):
        self.pairs = {}
        self.index = defaultdict(set)
        self.normals = {}

    def record(self, collider0, collider1, *, active, impulse_ns, separation_m, normals_on_collider0=None):
        pair = tuple(sorted((collider0, collider1)))
        if pair not in self.pairs:
            for path in pair:
                parts = path.split("/")
                for end in range(2, len(parts) + 1):
                    self.index["/".join(parts[:end])].add(pair)
        self.pairs[pair] = (bool(active), float(impulse_ns), float(separation_m))
        if normals_on_collider0 is not None:
            sign = 1. if pair[0] == collider0 else -1.
            normals = tuple(tuple(sign*float(x) for x in normal) for normal in normals_on_collider0)
            if any(len(normal) != 3 or not all(math.isfinite(x) for x in normal) for normal in normals):
                raise ValueError('invalid native support normal')
            self.normals[pair] = normals
        if not active:
            self.normals.pop(pair, None)

    def touch(self, root, other=None, robot_only=False, link=None):
        def matches(path, prefix):
            return path == prefix or path.startswith(prefix + "/")
        for pair in self.index.get(root, ()):
            active, impulse, separation = self.pairs[pair]
            if not active or not (impulse > 1e-7 or separation < 0):
                continue
            for path, opposite in (pair, pair[::-1]):
                if not matches(path, root):
                    continue
                if other is not None and not matches(opposite, other):
                    continue
                if robot_only and "/Robot" not in opposite:
                    continue
                if link is not None and not ("/" + link + "/" in opposite or opposite.endswith("/" + link)):
                    continue
                return True
        return False

    def supported(self, root, other=None):
        """Actual upward target contact; no impulse minimum or gripper credit.

        Normals are measured on the original collider0 and retain ownership
        when pair order is canonicalized. Missing normals never imply support.
        """
        def matches(path, prefix):
            return path == prefix or path.startswith(prefix + '/')
        for pair in self.index.get(root, ()):
            active, impulse, separation = self.pairs[pair]
            if not active or not (impulse > 0. or separation <= 0.):
                continue
            for index, (path, opposite) in enumerate((pair, pair[::-1])):
                if not matches(path, root) or matches(opposite, root) or opposite.startswith('/World/Robot'):
                    continue
                if other is not None and not matches(opposite, other):
                    continue
                sign = 1. if index == 0 else -1.
                normals = self.normals.get(pair)
                if not normals:
                    raise ValueError('active target contact lacks a measured support normal')
                if any(sign*normal[2] > 0. for normal in normals):
                    return True
        return False

    def bilateral(self, root):
        return any(self.touch(root, other="/World/Robot" + side, link="link7") and
                   self.touch(root, other="/World/Robot" + side, link="link8")
                   for side in ("Left", "Right"))

    def clear(self):
        self.pairs.clear()
        self.index.clear()
        self.normals.clear()


class IsaacContacts(ContactLedger):
    def __init__(self, stage):
        super().__init__()
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        self.stage = stage
        self._path_cache = {}
        def path(value):
            if value not in self._path_cache:
                self._path_cache[value] = str(PhysicsSchemaTools.intToSdfPath(value))
            return self._path_cache[value]
        def receive(headers, data):
            for header in headers:
                points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
                collider0, collider1 = path(header.collider0), path(header.collider1)
                active = header.type != ContactEventType.CONTACT_LOST
                # Sleeping manifolds stay valid until a native CONTACT_LOST.
                if active and len(points) == 0 and tuple(sorted((collider0, collider1))) in self.pairs:
                    continue
                self.record(collider0, collider1,
                            active=active,
                            impulse_ns=max((math.hypot(*p.impulse) for p in points), default=0.),
                            separation_m=min((float(p.separation) for p in points), default=1.),
                            normals_on_collider0=[tuple(float(p.normal[j]) for j in range(3)) for p in points])
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(receive)

    def held_tilt(self, root):
        from pxr import UsdGeom, Usd
        import numpy as np
        values = []
        for side in ("Left", "Right"):
            if any(self.touch(root, other="/World/Robot" + side, link=link) for link in ("link7", "link8")):
                prim = self.stage.GetPrimAtPath("/World/Robot" + side + "/Asset/link6")
                if not prim:
                    raise ValueError("native flange transform missing")
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                rotation = np.asarray(matrix)[:3, :3].T
                values.append(round(float(np.degrees(np.arccos(np.clip(-rotation[2, 0], -1., 1.)))), 10))
        return max(values, default=0.)

    def close(self):
        self._subscription = None
        self.clear()
