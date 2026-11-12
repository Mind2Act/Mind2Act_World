"""Native evidence stays bounded per tick and retains independently verifiable bytes."""
import json

import pytest

from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.E2.evidence import Measurements, read_measurements


def test_per_tick_evidence_size_does_not_grow_with_the_physical_trace(tmp_path):
    writer = Measurements(tmp_path / "measurements.jsonl")
    sizes = []
    for tick in range(1, 3001):
        writer.append({"tick": tick, "travel_m": [0.] * 9,
                       "fingertip_contacts": [False] * 9, "gripper_positions": [0., 0.]})
        native = NativeEvidence("E2", "synthetic", tick, {"measurements": writer.receipt()})
        sizes.append(len(json.dumps(to_json(native))))
    receipt = writer.receipt()
    writer.close()
    assert max(sizes) - min(sizes) < 40
    assert len(list(read_measurements(receipt))) == 3000
    assert receipt["bytes"] > 500_000


@pytest.mark.parametrize("corruption", ["append", "truncate", "edit"])
def test_missing_or_changed_native_bytes_reject_the_receipt(tmp_path, corruption):
    writer = Measurements(tmp_path / "measurements.jsonl")
    writer.append({"tick": 1})
    receipt = writer.receipt()
    writer.close()
    raw = writer.path.read_bytes()
    writer.path.write_bytes(raw + raw if corruption == "append" else b"" if corruption == "truncate" else raw.replace(b"1", b"2"))
    with pytest.raises(ValueError, match="receipt"):
        list(read_measurements(receipt))
