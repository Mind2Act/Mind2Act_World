import json
from pathlib import Path

import pytest

from mind2act_eval.contracts import (
    ConsumerKind, InstanceSpec, Result, Termination, to_json,
)
from mind2act_eval.sealing import seal_worker_outputs, validate_worker_seal


def _successful_attempt(root: Path, *, pid=12345, task_id="demo", artifacts=None):
    spec = InstanceSpec(task_id, "seal-001", "easy", "synthetic-public-v1", "v1", "none", "v1")
    (root / "spec.json").write_text(json.dumps(to_json(spec)))
    (root / "evidence.json").write_text(json.dumps({"task_id": "demo", "tick": 4}))
    (root / "score.json").write_text(json.dumps({"value": 25.0}))
    (root / "video.mp4").write_bytes(b"encoded-video")
    result = Result(task_id, "seal-001", 25.0, Termination.PHYSICS_BUDGET, True,
                    ConsumerKind.DIAGNOSTIC, False, 4,
                    artifacts={"video": "video.mp4"} if artifacts is None else artifacts)
    (root / "result.json").write_text(json.dumps(to_json(result)))
    rows = (
        ("run-episode-returned.json", "run_episode_returned", {"infra_valid": True}),
        ("worker-returned.json", "worker_returned", {"exit_code": 0}),
        ("cli-returned.json", "cli_returned", {"exit_code": 0}),
    )
    for filename, stage, details in rows:
        (root / filename).write_text(json.dumps({
            "schema": "mind2act_eval.worker_lifecycle.v1", "stage": stage,
            "pid": pid, **details,
        }))
    return spec, result


def test_worker_seal_fsync_manifest_covers_result_evidence_and_video(tmp_path):
    spec, expected = _successful_attempt(tmp_path)

    seal = seal_worker_outputs(tmp_path)
    actual, checked = validate_worker_seal(tmp_path)

    assert checked == seal
    assert checked["status"] == "sealed"
    assert checked["teardown"] == "os_exit"
    assert actual == expected
    names = {row["path"] for row in seal["files"]}
    assert {"spec.json", "result.json", "evidence.json", "score.json", "video.mp4"} <= names


def test_worker_seal_rejects_a_changed_video(tmp_path):
    _successful_attempt(tmp_path)
    seal_worker_outputs(tmp_path)
    (tmp_path / "video.mp4").write_bytes(b"altered-video")

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_worker_seal(tmp_path)


def test_worker_seal_cannot_be_written_before_a_required_evidence_file(tmp_path):
    _successful_attempt(tmp_path)
    (tmp_path / "evidence.json").unlink()

    with pytest.raises(ValueError, match="required result artifact missing"):
        seal_worker_outputs(tmp_path)


def test_native_result_requires_closed_camera_and_video_producer_outputs(tmp_path):
    _successful_attempt(tmp_path, task_id="pr1", artifacts={})

    with pytest.raises(ValueError, match="native result must reference every camera video"):
        seal_worker_outputs(tmp_path)


def test_native_result_seals_only_after_all_camera_videos_and_producers_close(tmp_path):
    artifacts = {
        "video_head": "head.mp4",
        "video_left_wrist": "left_wrist.mp4",
        "video_right_wrist": "right_wrist.mp4",
        "native_camera_receipt": "camera-receipt.json",
        "native_shutdown_receipt": "shutdown-receipt.json",
    }
    _, expected = _successful_attempt(tmp_path, task_id="pr1", artifacts=artifacts)
    for name in ("head", "left_wrist", "right_wrist"):
        (tmp_path / (name + ".mp4")).write_bytes(b"encoded " + name.encode())
    cameras = {name: {"valid": True} for name in ("head", "left_wrist", "right_wrist")}
    (tmp_path / "camera-receipt.json").write_text(json.dumps({
        "schema": "mind2act_eval.native_camera_receipt.v1",
        "status": "verified", "initial_frame_encoded": True, "cameras": cameras,
    }))
    (tmp_path / "shutdown-receipt.json").write_text(json.dumps({
        "schema": "mind2act_eval.shutdown_receipt.v1",
        "status": "os_exit_prepared", "teardown": "os_exit",
        "phases": [{"name": "recorder", "status": "complete"},
                   {"name": "app.close", "status": "skipped"}],
    }))
    (tmp_path / "result.json").write_text(json.dumps(to_json(expected)))

    seal = seal_worker_outputs(tmp_path)
    result, _ = validate_worker_seal(tmp_path)

    assert result.infra_valid
    assert set(artifacts) <= set(result.artifacts)
    assert seal["status"] == "sealed"


def test_worker_seal_rejects_seal_path_symlink(tmp_path):
    _successful_attempt(tmp_path)
    seal_worker_outputs(tmp_path)
    external = tmp_path.parent / (tmp_path.name + "-external-seal.json")
    (tmp_path / "worker-seal.json").rename(external)
    (tmp_path / "worker-seal.json").symlink_to(external)

    with pytest.raises(ValueError, match="worker seal must be a regular in-attempt file"):
        validate_worker_seal(tmp_path)
