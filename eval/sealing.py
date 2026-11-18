"""Durable worker completion seals for supervised episode children."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from mind2act_eval.contracts import ConsumerKind, InstanceSpec, Result

SEAL_NAME = "worker-seal.json"
_IGNORED_DIRS = {"cache", "tmp", "native-exit-debug"}
_OPTIONAL_ARTIFACTS = {"cleanup_traceback"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_kit(root: Path) -> bool:
    """Only the fresh whole-Kit layout opts out; old seals keep their inventory."""
    owner = root / 'scratch-owner.json'
    if not owner.exists():
        return False
    if owner.is_symlink() or not owner.is_file():
        raise ValueError('scratch ownership marker must be a regular file')
    row = json.loads(owner.read_text())
    return (isinstance(row, dict) and row.get('schema') == 'mind2act_eval.scratch_owner.v1'
            and [p.get('path') for p in row.get('paths', [])] == ['cache', 'tmp', 'kit'])


def _excluded(relative: Path, *, temporary_kit=False) -> bool:
    parts = relative.parts
    if temporary_kit and parts and parts[0] == 'kit':
        return True
    if any(part in _IGNORED_DIRS for part in parts):
        return True
    if any(parts[index:index + 2] == ("kit", "data") for index in range(len(parts) - 1)):
        return True
    return relative.as_posix() in {SEAL_NAME, "batch-exit.json", "scratch-ledger.json", "scratch-cleanup.json"}


def _inventory(root: Path) -> list[Path]:
    files = []
    temporary_kit = _temporary_kit(root)
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _excluded(relative, temporary_kit=temporary_kit):
            continue
        if path.is_symlink():
            raise ValueError("worker output tree contains a symlink: " + relative.as_posix())
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("worker output tree contains a non-regular file: " + relative.as_posix())
        if relative.name.endswith(".tmp"):
            raise ValueError("worker output tree contains an unfinished temporary file")
        files.append(relative)
    return sorted(files, key=lambda p: p.as_posix())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directories(root: Path) -> None:
    temporary_kit = _temporary_kit(root)
    dirs = [path for path in root.rglob("*") if path.is_dir()
            and not _excluded(path.relative_to(root), temporary_kit=temporary_kit)]
    for directory in sorted(dirs, key=lambda p: len(p.parts), reverse=True) + [root]:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _lifecycle_marker(root: Path, name: str, stage: str) -> dict:
    row = json.loads((root / name).read_text())
    if (not isinstance(row, dict) or row.get("schema") != "mind2act_eval.worker_lifecycle.v1"
            or row.get("stage") != stage or type(row.get("pid")) is not int):
        raise ValueError("invalid worker lifecycle marker: " + name)
    return row


def _read_seal(path: Path) -> dict:
    """Open only an in-attempt regular seal file; never follow an external link."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("worker seal must be a regular in-attempt file") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("worker seal must be a regular in-attempt file")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            return json.load(stream)
    finally:
        if fd >= 0:
            os.close(fd)


def _validate_result_closure(root: Path, result: Result) -> None:
    """Require every referenced output, and native camera/encoder closure."""
    for key, value in result.artifacts.items():
        if not isinstance(value, str):
            raise TypeError("result artifact paths must be strings")
        candidate = root / value
        try:
            relative = candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            if key in _OPTIONAL_ARTIFACTS and not candidate.exists():
                continue
            raise ValueError("result artifact is missing or escapes attempt output: " + key)
        if _excluded(relative, temporary_kit=_temporary_kit(root)):
            raise ValueError("result artifact references ephemeral/unsealed output: " + key)
        if not candidate.is_file():
            raise ValueError("result artifact is not a regular file: " + key)

    spec = json.loads((root / "spec.json").read_text())
    if not isinstance(spec, dict) or spec.get("task_id") != result.task_id:
        raise ValueError("worker spec/result task identity mismatch")
    if result.task_id == "demo":
        return

    cameras_path = result.artifacts.get("native_camera_receipt")
    shutdown_path = result.artifacts.get("native_shutdown_receipt")
    camera_video_keys = {key for key in result.artifacts if key.startswith("video_")}
    if not camera_video_keys:
        raise ValueError("native result must reference every camera video")
    if not cameras_path or not shutdown_path:
        raise ValueError("native result must reference camera and shutdown receipts")
    camera_receipt = json.loads((root / cameras_path).read_text())
    shutdown_receipt = json.loads((root / shutdown_path).read_text())
    camera_rows = camera_receipt.get("cameras") if isinstance(camera_receipt, dict) else None
    if (not isinstance(camera_receipt, dict)
            or camera_receipt.get("schema") != "mind2act_eval.native_camera_receipt.v1"
            or camera_receipt.get("status") != "verified"
            or camera_receipt.get("initial_frame_encoded") is not True
            or not isinstance(camera_rows, dict) or not camera_rows
            or any(not isinstance(row, dict) or row.get("valid") is not True
                   for row in camera_rows.values())):
        raise ValueError("native camera receipt does not prove valid initial frames")
    expected_video_keys = {"video_" + name for name in camera_rows}
    if camera_video_keys != expected_video_keys:
        raise ValueError("native result must reference every camera video")
    for key in camera_video_keys:
        video = root / result.artifacts[key]
        if video.suffix.lower() != ".mp4" or video.stat().st_size <= 0:
            raise ValueError("native camera video is missing or empty: " + key)
    phases = shutdown_receipt.get("phases") if isinstance(shutdown_receipt, dict) else None
    phase_status = {(row.get("name"), row.get("status")) for row in phases or () if isinstance(row, dict)}
    if (not isinstance(shutdown_receipt, dict)
            or shutdown_receipt.get("schema") != "mind2act_eval.shutdown_receipt.v1"
            or shutdown_receipt.get("status") != "os_exit_prepared"
            or shutdown_receipt.get("teardown") != "os_exit"
            or ("recorder", "complete") not in phase_status
            or ("app.close", "skipped") not in phase_status
            or any(status not in ("complete", "skipped") for _, status in phase_status)):
        raise ValueError("native media producer shutdown is incomplete")


def _completion_contract(root, result, expected_consumer=None):
    """Public output seals require a scored model run or trusted replay."""
    if not result.infra_valid:
        raise ValueError("only infrastructure-valid worker output can be sealed")
    if expected_consumer is not None and result.consumer_kind != ConsumerKind(expected_consumer):
        raise ValueError("worker consumer differs from the trusted parent job")
    if result.consumer_kind == ConsumerKind.SCENE_PROBE:
        raise ValueError("diagnostic consumers are not part of the public runtime")
    if result.score is None:
        raise ValueError("only a valid scored result can be sealed")
    return ("spec.json", "result.json", "evidence.json", "score.json"), "run-episode-returned.json", "run_episode_returned"


def flush_failed_worker_outputs(directory: Path, *, exit_code: int) -> None:
    """Durably retain an invalid native attempt without publishing a success seal.

    The CLI also requires an in-memory witness from this runner's completed
    cleanup and source-receipt writes. These disk checks cannot grant that right.
    """
    root = Path(directory).resolve(strict=True)
    result = Result(**json.loads((root / "result.json").read_text()))
    spec = InstanceSpec.from_dict(json.loads((root / "spec.json").read_text()))
    if (type(exit_code) is not int or not 0 < exit_code < 256
            or result.infra_valid or result.score is not None
            or result.task_id == "demo" or result.consumer_kind == ConsumerKind.SCENE_PROBE
            or (result.task_id, result.instance_id) != (spec.task_id, spec.instance_id)
            or (root / SEAL_NAME).exists()):
        raise ValueError("failure exit requires an unsealed invalid native result and nonzero status")
    for name, stage in (("run-episode-returned.json", "run_episode_returned"),
                        ("worker-returned.json", "worker_returned"), ("cli-returned.json", "cli_returned")):
        row = _lifecycle_marker(root, name, stage)
        correct_status = (row.get('infra_valid') is False if stage == 'run_episode_returned'
                          else row.get('exit_code') == exit_code)
        if row['pid'] != os.getpid() or not correct_status:
            raise ValueError("failure lifecycle does not match this worker")
    shutdown = json.loads((root / "shutdown-receipt.json").read_text())
    phases = shutdown.get('phases', [])
    if (shutdown.get('schema') != 'mind2act_eval.shutdown_receipt.v1'
            or shutdown.get('status') != 'os_exit_prepared' or shutdown.get('teardown') != 'os_exit'
            or not isinstance(phases, list) or not phases or any(not isinstance(row, dict) for row in phases)
            or not any(row == {'name': 'recorder', 'status': 'complete'} for row in phases)
            or not any(row.get('name') == 'app.close' and row.get('status') == 'skipped' for row in phases)
            or any(row.get('status') != ('skipped' if row.get('name') == 'app.close' else 'complete') for row in phases)):
        raise ValueError("failure exit preparation is incomplete")

    # Match the existing output boundary, but enumerate explicitly: an unreadable
    # diagnostic directory must fail the flush rather than be silently omitted.
    def sync_tree(directory):
        with os.scandir(directory) as entries:
            children = list(entries)
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(root)
            if _excluded(relative, temporary_kit=temporary_kit):
                continue
            if entry.is_symlink():
                raise ValueError("worker output tree contains a symlink: " + relative.as_posix())
            if entry.is_dir(follow_symlinks=False):
                sync_tree(path)
            elif entry.is_file(follow_symlinks=False) and not path.name.endswith('.tmp'):
                _fsync_file(path)
            else:
                raise ValueError("worker failure output is not a finished regular file: " + relative.as_posix())
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    temporary_kit = _temporary_kit(root)
    sync_tree(root)


def seal_worker_outputs(directory: Path, *, expected_consumer=None) -> dict:
    """Fsync and hash a successful attempt, then atomically publish its seal receipt."""
    root = Path(directory).resolve(strict=True)
    result_path = root / "result.json"
    result = Result(**json.loads(result_path.read_text()))
    required, run_name, run_stage = _completion_contract(root, result, expected_consumer)
    for name in required:
        if not (root / name).is_file():
            raise ValueError("required result artifact missing before seal: " + name)
    run_row = _lifecycle_marker(root, run_name, run_stage)
    worker_row = _lifecycle_marker(root, "worker-returned.json", "worker_returned")
    cli_row = _lifecycle_marker(root, "cli-returned.json", "cli_returned")
    pids = {run_row["pid"], worker_row["pid"], cli_row["pid"]}
    if len(pids) != 1 or run_row.get("infra_valid") is not True or worker_row.get("exit_code") != 0 or cli_row.get("exit_code") != 0:
        raise ValueError("worker lifecycle markers do not attest a successful episode return")
    _validate_result_closure(root, result)

    relative_files = _inventory(root)
    if not set(required).issubset({p.as_posix() for p in relative_files}):
        raise ValueError("worker output inventory omitted a required artifact")
    for relative in relative_files:
        _fsync_file(root / relative)
    _fsync_directories(root)
    files = [{"path": relative.as_posix(), "bytes": (root / relative).stat().st_size,
              "sha256": _sha256(root / relative)} for relative in relative_files]
    seal = {
        "schema": "mind2act_eval.worker_seal.v1",
        "status": "sealed",
        "teardown": "os_exit",
        "worker_pid": next(iter(pids)),
        "files": files,
        "result_sha256": _sha256(result_path),
    }
    if _temporary_kit(root):
        seal['temporary_kit'] = True
    target = root / SEAL_NAME
    temporary = root / (SEAL_NAME + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return seal


def validate_worker_seal(directory: Path, *, durable=False, expected_consumer=None, strict=False) -> tuple[Result, dict]:
    """Read completed results; verify the byte seal only in explicit strict mode."""
    root = Path(directory).resolve(strict=True)
    if not strict:
        result = Result(**json.loads((root / "result.json").read_text()))
        if expected_consumer is not None and result.consumer_kind != expected_consumer:
            raise ValueError("worker result consumer mismatch")
        _validate_result_closure(root, result)
        return result, {"status": "unchecked", "integrity_verified": False}
    seal_path = root / SEAL_NAME
    seal = _read_seal(seal_path)
    if (not isinstance(seal, dict) or seal.get("schema") != "mind2act_eval.worker_seal.v1"
            or seal.get("status") != "sealed" or seal.get("teardown") != "os_exit"
            or type(seal.get("worker_pid")) is not int or not isinstance(seal.get("files"), list)):
        raise ValueError("invalid worker completion seal")
    temporary_kit = seal.get('temporary_kit', False)
    if type(temporary_kit) is not bool or temporary_kit != _temporary_kit(root):
        raise ValueError('worker seal temporary layout disagrees with its ownership marker')
    manifest = {}
    for row in seal["files"]:
        if (not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row["path"], str) or type(row["bytes"]) is not int
                or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64):
            raise ValueError("invalid worker seal file entry")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or _excluded(relative, temporary_kit=temporary_kit) or row["path"] in manifest:
            raise ValueError("invalid worker seal path")
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("sealed worker artifact is missing or changed: " + row["path"])
        if _sha256(path) != row["sha256"]:
            raise ValueError("sealed worker artifact digest mismatch: " + row["path"])
        manifest[row["path"]] = row["sha256"]
    if set(manifest) != {p.as_posix() for p in _inventory(root)}:
        raise ValueError("worker output files differ from the sealed inventory")
    result = Result(**json.loads((root / "result.json").read_text()))
    if temporary_kit and 'scratch-owner.json' not in manifest:
        raise ValueError('worker seal omitted scratch ownership marker')
    required, run_name, run_stage = _completion_contract(root, result, expected_consumer)
    for name in (*required, run_name, "worker-returned.json", "cli-returned.json"):
        if name not in manifest:
            raise ValueError("worker seal omitted required file: " + name)
    _validate_result_closure(root, result)
    if seal.get("result_sha256") != manifest.get("result.json"):
        raise ValueError("worker seal result digest disagrees with its manifest")
    run_row = _lifecycle_marker(root, run_name, run_stage)
    worker_row = _lifecycle_marker(root, "worker-returned.json", "worker_returned")
    cli_row = _lifecycle_marker(root, "cli-returned.json", "cli_returned")
    if ({run_row["pid"], worker_row["pid"], cli_row["pid"]} != {seal["worker_pid"]}
            or run_row.get("infra_valid") is not True
            or worker_row.get("exit_code") != 0 or cli_row.get("exit_code") != 0):
        raise ValueError("worker seal lifecycle markers disagree")
    if durable:
        # The writer publishes the name with replace() before its final parent
        # directory fsync. Re-sync the immutable tree after the worker stops,
        # then re-read every digest so visibility alone never commits a result.
        for relative in manifest:
            _fsync_file(root / relative)
        _fsync_file(seal_path)
        _fsync_directories(root)
        return validate_worker_seal(root, durable=False, expected_consumer=expected_consumer, strict=True)
    return result, seal
