"""One summed impulse vector per collider pair, as in the selected PR5 source."""
import numpy as np


class ContactReadings:
    def __init__(self):
        self.pairs = {}
        self.support_normals = {}

    def record(self, pair, impulses, support_normal_z=None):
        value = np.asarray(impulses, dtype=float)
        if value.size == 0:
            vector = np.zeros(3)
        elif value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
            raise ValueError('invalid native PR5 contact impulses')
        else:
            vector = value.sum(axis=0)
        key = tuple(sorted(pair))
        self.pairs[key] = vector.tolist()
        if support_normal_z is not None:
            self.support_normals[key] = float(support_normal_z)

    def lost(self, pair):
        key = tuple(sorted(pair))
        self.pairs.pop(key, None)
        self.support_normals.pop(key, None)

    def clear(self):
        self.pairs.clear()

    def snapshot(self):
        # Sleeping contacts retain their physical manifold until CONTACT_LOST;
        # their old impulses are never reused as fresh stamping force.
        result = []
        for pair in sorted(set(self.pairs) | set(self.support_normals)):
            row = dict(pair=list(pair), impulse_ns=list(self.pairs.get(pair, [0., 0., 0.])))
            if pair in self.support_normals:
                row['support_normal_z'] = self.support_normals[pair]
            result.append(row)
        return result


def subscribe(readings):
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    from omni.physx.bindings._physx import ContactEventType

    def receive(headers, data):
        for header in headers:
            pair = [str(PhysicsSchemaTools.intToSdfPath(p)) for p in (header.collider0, header.collider1)]
            if header.type == ContactEventType.CONTACT_LOST:
                readings.lost(pair)
                continue
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            sign = 1. if pair[0].startswith('/World/Task/StampA') else -1.
            normal_z = max((sign * float(point.normal[2]) for point in points), default=None)
            readings.record(pair, [[float(point.impulse[i]) for i in range(3)] for point in points], normal_z)
    return get_physx_simulation_interface().subscribe_contact_report_events(receive)
