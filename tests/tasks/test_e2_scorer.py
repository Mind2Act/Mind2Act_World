import hashlib
import itertools
import json
from pathlib import Path

import pytest

from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.E2.scorer import score


TRACES = itertools.count()


def seal(tmp_path, rows):
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    path = tmp_path / f"measurements-{next(TRACES)}.jsonl"
    path.write_bytes(raw)
    return {"schema": "e2.native-measurements.v1", "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows), "bytes": len(raw)}


def evidence(tmp_path, target=(1, 2), entered=None, *, level="easy", release=True, extra_down=False, normal=True, idle=330, transform=None):
    rows = []
    def add(count, slot=None, contact=False, depth=-.003):
        for _ in range(count):
            travel, contacts = [0.] * 9, [False] * 9
            if slot is not None:
                travel[slot], contacts[slot] = depth, contact
            if extra_down:
                travel[8] = -.002
            rows.append({"tick": len(rows) + 1, "travel_m": travel, "fingertip_contacts": contacts,
                         "fingertip_contact_sides": [["Left"] if value else [] for value in contacts],
                         "gripper_positions": [0., 0.]})
    add(idle)
    for i, digit in enumerate(target if entered is None else entered):
        add(3, digit - 1, True)
        if release or i != len(target) - 1:
            add(4)
    if transform:
        transform(rows)
    return NativeEvidence("E2", "synthetic", len(rows), {
        "difficulty": level, "target_digits": list(target),
        "timing": {"learn_seconds": 2., "shuffle_seconds": .5, "reasoning_seconds": .25},
        "measurements": seal(tmp_path, rows), "normal_completion_tick": len(rows) if normal else None})


@pytest.mark.parametrize("level,target", [("easy", [1, 2]), ("medium", [4, 4, 3]), ("hard", [8, 2, 1]), ("hard", [8, 7, 2, 1])])
def test_all_tiers_exact_sequence_earns_100(tmp_path, level, target):
    result = score(evidence(tmp_path, target, level=level))
    assert result.value == 100
    assert result.components["task_success"] is True
    assert result.components["rubric"] == "e2-spec-20261030-v1"


@pytest.mark.parametrize("kwargs,expected", [({"entered": [2, 1]},0), ({"entered": [1]},15), ({"entered": [1, 2, 3]},30), ({"release": False},30)])
def test_wrong_missing_extra_and_unreleased_keep_only_position_credit(tmp_path, kwargs, expected):
    result = score(evidence(tmp_path, **kwargs))
    assert result.value == expected and result.components["task_success"] is False


def test_mechanical_input_needs_no_contact_or_supplied_success_labels(tmp_path):
    def change(rows):
        for row in rows:
            row["fingertip_contacts"] = [False] * 9
            row["fingertip_contact_sides"] = [[] for _ in range(9)]
            row.update(success=True, action_index=100, milestone="finished")
    result = score(evidence(tmp_path, transform=change))
    assert result.value == 100


def test_incomplete_and_nonconsecutive_native_coverage_raises(tmp_path):
    native = evidence(tmp_path)
    for change in (lambda rows: rows.pop(), lambda rows: rows[5].update(tick=8)):
        v = to_json(native.values)
        rows = [json.loads(line) for line in Path(v["measurements"]["path"]).read_text().splitlines()]
        change(rows)
        v["measurements"] = seal(tmp_path, rows)
        with pytest.raises(ValueError):
            score(NativeEvidence("E2", "synthetic", native.tick, v))


def test_digits_completed_before_reasoning_wait_ends_cannot_score(tmp_path):
    result = score(evidence(tmp_path, idle=300))
    assert result.value == 0
    assert result.components["entered_digits"] == ()


def test_sealed_measurement_bytes_cannot_be_edited_after_receipt(tmp_path):
    native = evidence(tmp_path)
    path = Path(native.values["measurements"]["path"])
    path.write_bytes(path.read_bytes().replace(b'"gripper_positions": [0.0, 0.0]',
                                             b'"gripper_positions": [1.0, 1.0]', 1))
    with pytest.raises(ValueError, match="receipt"):
        score(native)


def test_approved_source_contact_rule_counts_open_grippers_with_real_finger_contact(tmp_path):
    # Approved 2026-11-01 source-parity ruling: collection has no closure gate.
    # Keep the instruction wording discrepancy visible in task documentation.
    def opening(rows):
        for row in rows:
            row["gripper_positions"] = [1., 1.]
    assert score(evidence(tmp_path, transform=opening)).value == 100


@pytest.mark.parametrize('target', [(1, 2), (2, 1)])
def test_report_overlap_counterexample_cannot_score_by_release_reordering(tmp_path, target):
    native = evidence(tmp_path, target=target)
    rows = [json.loads(line) for line in Path(native.values['measurements']['path']).read_text().splitlines()]
    rows = rows[:330]
    # First press 1, then 2; release 2, then 1. Source contact remains valid at any opening.
    for down_slots, count in [([0], 3), ([0, 1], 3), ([0], 4), ([], 4)]:
        for _ in range(count):
            rows.append(dict(tick=len(rows) + 1, travel_m=[-.003 if i in down_slots else 0. for i in range(9)],
                             fingertip_contacts=[i in down_slots for i in range(9)], gripper_positions=[1., 1.]))
    v = to_json(native.values)
    v.update(measurements=seal(tmp_path, rows), normal_completion_tick=len(rows))
    result = score(NativeEvidence('E2', 'synthetic', len(rows), v))
    assert list(result.components['entered_digits']) == [1]
    assert result.value == (15 if target[0]==1 else 0)


@pytest.mark.parametrize('kwargs', [{'extra_down': True}, {'normal': False}])
def test_terminal_input_needs_no_unrelated_release_or_completion_latch(tmp_path, kwargs):
    assert score(evidence(tmp_path, **kwargs)).value == 100
