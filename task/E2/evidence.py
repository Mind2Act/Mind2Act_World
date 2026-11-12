"""Task-specific projection of the shared native contact ledger."""
import math
import hashlib
import json
from pathlib import Path


class Measurements:
    """Append native rows once; per-tick evidence contains a constant-size receipt."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("xb")
        self.digest = hashlib.sha256()
        self.count = 0
        self.bytes = 0

    def append(self, row):
        if row["tick"] != self.count + 1:
            raise ValueError("E2 measurement writer requires every consecutive tick")
        raw = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.count += 1
        self.bytes += len(raw)

    def receipt(self):
        self.stream.flush()
        return {"schema": "e2.native-measurements.v1", "path": str(self.path),
                "sha256": self.digest.hexdigest(), "rows": self.count, "bytes": self.bytes}

    def close(self):
        self.stream.close()


def read_measurements(receipt):
    """Score only the sealed byte range and reject any missing or changed bytes."""
    if receipt["schema"] != "e2.native-measurements.v1":
        raise ValueError("E2 measurement schema mismatch")
    digest, count, size = hashlib.sha256(), 0, 0
    with Path(receipt["path"]).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            size += len(raw)
            count += 1
            yield json.loads(raw)
    if (count != receipt["rows"] or size != receipt["bytes"]
            or digest.hexdigest() != receipt["sha256"]):
        raise ValueError("E2 measurement receipt differs from native bytes")


def finger_contact_sides(ledger, cap):
    """Original E2 world.py thresholds, distinct from generic CP01 touch()."""
    sides = set()
    for pair in ledger.index.get(cap, ()):
        active, impulse, separation = ledger.pairs[pair]
        if not active or not (impulse > 1e-8 or separation < .00003):
            continue
        for path, other in (pair, pair[::-1]):
            if not path.startswith(cap + "/"):
                continue
            for side in ("Left", "Right"):
                if any(other.startswith(f"/World/Robot{side}/Asset/{link}/") for link in ("link7", "link8")):
                    sides.add(side)
    return sorted(sides)


def finger_contact(ledger, cap):
    return bool(finger_contact_sides(ledger, cap))


def measure(tick, case, caps, ledger, grippers):
    travel, contacts, contact_sides = [0.0] * 9, [False] * 9, [[] for _ in range(9)]
    occupied = set()
    cap_z = .768 + .034
    for entity, cap in enumerate(caps):
        position = tuple(float(v) for v in cap.get_world_pose()[0])
        if not all(math.isfinite(v) for v in position):
            raise ValueError("E2 cap pose is non-finite")
        slot = min(range(9), key=lambda i: sum((case["slot_xy_m"][i][j] - position[j])**2 for j in (0, 1)))
        if slot in occupied:
            raise ValueError("E2 multiple caps occupy the same slot")
        occupied.add(slot)
        travel[slot] = position[2] - cap_z
        contact_sides[slot] = finger_contact_sides(ledger, f"/World/EKeyboard/Key{entity}/Cap")
        contacts[slot] = bool(contact_sides[slot])
    return {"tick": tick, "travel_m": travel, "fingertip_contacts": contacts,
            "fingertip_contact_sides": contact_sides,
            "gripper_positions": list(grippers)}
