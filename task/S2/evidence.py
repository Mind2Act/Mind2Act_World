"""Append-only native samples and task-local contact manifolds, no expert inputs."""
import hashlib
import json
from pathlib import Path

SLIDER = "/World/Task/Objects/slider"
CONTROLS = "/World/Task/Design/Controls/"


class Rows:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("xb")
        self.hash = hashlib.sha256()
        self.count, self.size = 0, 0

    def append(self, frame):
        if frame["tick"] != self.count + 1:
            raise ValueError("S2 measurements require consecutive ticks from 1")
        raw = (json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.file.write(raw)
        self.hash.update(raw)
        self.count += 1
        self.size += len(raw)

    def receipt(self):
        self.file.flush()
        return {"schema": "s2.native-measurements.v1", "path": str(self.path),
                "sha256": self.hash.hexdigest(), "rows": self.count, "bytes": self.size}

    def close(self):
        self.file.close()


def read_rows(receipt, *, strict=False):
    if receipt["schema"] != "s2.native-measurements.v1":
        raise ValueError("S2 native measurement schema mismatch")
    digest, size, count = hashlib.sha256(), 0, 0
    with Path(receipt["path"]).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            size += len(raw)
            count += 1
            yield json.loads(raw)
    if (size, count) != (receipt['bytes'], receipt['rows']) or (strict and digest.hexdigest() != receipt['sha256']):
        raise ValueError("S2 native measurement SHA/size/count mismatch")


class Manifolds:
    """Preserve the source's current contact batch, active pairs and normals."""
    def __init__(self):
        self.events, self.active, self.points = [], {}, {}
        self.subscription = None

    def subscribe(self):
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType

        def report(headers, data):
            for h in headers:
                ordered = tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
                if not any("/World/Task/" in p for p in ordered):
                    continue
                pair = tuple(sorted(ordered))
                active = h.type != ContactEventType.CONTACT_LOST
                rows = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
                sign = 1 if pair[0] == ordered[0] else -1
                points = [{"position": list(map(float, p.position)),
                           "normal": [sign * float(v) for v in p.normal],
                           "separation": float(p.separation), "impulse": list(map(float, p.impulse))} for p in rows]
                self.active[pair] = active
                if active and points:
                    self.points[pair] = points
                elif not active:
                    self.points.pop(pair, None)
                self.events.append({"pair": list(pair), "active": active, "contact_count": len(points),
                                    "min_separation_m": min((p["separation"] for p in points), default=1.),
                                    "points": points})
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(report)

    def contacts(self, path):
        return sorted({other for pair, active in self.active.items() if active
                       for own, other in (pair, pair[::-1]) if own == path or own.startswith(path + "/")})

    def supports(self):
        result = []
        for pair, points in self.points.items():
            if not self.active.get(pair):
                continue
            for own, other, sign in ((pair[0], pair[1], 1), (pair[1], pair[0], -1)):
                if own != SLIDER and not own.startswith(SLIDER + "/"):
                    continue
                for point in points:
                    normal = [sign * v for v in point["normal"]]
                    if normal[2] >= .5:
                        result.append({"path": other, "position": point["position"],
                                       "normal": normal, "separation": point["separation"]})
        return result

    def close(self):
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None


def measure(tick, c, slider, buttons, robots, manifolds, target, world):
    import numpy as np
    from scipy.spatial.transform import Rotation
    p, q = slider.get_world_pose()
    half = np.abs(Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()) @ (np.asarray(c["objects"]["slider"]["size"]) / 2)
    body = {"lower": (p - half).tolist(), "upper": (p + half).tolist(),
            "position": p.tolist(), "orientation_wxyz": q.tolist(),
            "linear": slider.get_linear_velocity().tolist(), "angular": slider.get_angular_velocity().tolist(),
            "contacts": manifolds.contacts(SLIDER), "support_contacts": manifolds.supports()}
    caps = {}
    for key, button in buttons.items():
        position, orientation = button.get_world_pose()
        path = CONTROLS + key + "/Cap"
        touch = any(e["active"] and e["contact_count"] > 0 and e["min_separation_m"] <= .0001
                    and any(own.startswith(path + "/") and other.startswith("/World/Robot")
                            for own, other in (e["pair"], e["pair"][::-1])) for e in manifolds.events)
        caps[key] = {"depth_m": float(c["direction_control"]["buttons"][key]["center"][2] - position[2]),
                     "robot_contact": touch, "position_m": position.tolist(), "orientation_wxyz": orientation.tolist(),
                     "linear_velocity_m_s": button.get_linear_velocity().tolist(),
                     "angular_velocity_rad_s": button.get_angular_velocity().tolist()}
    return {"tick": tick, "bodies": {"slider": body}, "direction_buttons": caps,
            "stage_drive_target_m": list(target), "contact_events": list(manifolds.events),
            "q": {side: r.get_joint_positions().tolist() for side, r in robots.items()},
            "simulation_time_s": float(world.current_time)}
