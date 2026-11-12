"""Native contact manifold ownership, including ongoing contacts and loss."""
from copy import deepcopy


class Contacts:
    def __init__(self):
        self.rows = {}

    def receive(self, headers, data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        for h in headers:
            ordered = tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0,h.collider1))
            pair = tuple(sorted(ordered))
            if not any('/World/Task/' in p for p in pair):
                continue
            active = h.type != ContactEventType.CONTACT_LOST
            if not active:
                self.rows.pop(pair,None)
                continue
            points = data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
            sign = 1 if pair[0] == ordered[0] else -1
            if points:
                self.rows[pair] = dict(pair=list(pair),active=True,points=[dict(
                    position=[float(p.position[j]) for j in range(3)],
                    normal=[sign*float(p.normal[j]) for j in range(3)],separation=float(p.separation)) for p in points])

    def snapshot(self):
        return deepcopy([self.rows[p] for p in sorted(self.rows)])


def subscribe(contacts):
    from omni.physx import get_physx_simulation_interface
    return get_physx_simulation_interface().subscribe_contact_report_events(contacts.receive)
