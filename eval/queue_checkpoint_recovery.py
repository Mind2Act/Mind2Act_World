"""Explicit stopped-owner recovery of an observed canonical checkpoint, never fallback."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import uuid

from mind2act_eval.distributed_campaign import node_lock_key
from mind2act_eval.shared_queue import QueueStore, _SHA256, _json_bytes, _now, _process_birth, _write_atomic


def _read(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("recovery input must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_commit_fence(store, lock_fd, expected_state_sha256):
    held, named = os.fstat(lock_fd), os.stat(store.lock_path, follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino) or store._read_state()[1] != expected_state_sha256:
        raise RuntimeError("recovery CAS or lock identity changed before commit")


@contextmanager
def _maintenance(store):
    """Use the same maintenance order as legacy adoption; never signal owners."""
    import fcntl
    descriptors = []
    try:
        paths = [store.queue.lock_root / "batch.lock"] + [
            store.queue.lock_root / f"batch-node-{node_lock_key(h)}.lock" for h in sorted(store.queue.nodes)]
        paths.append(store.lock_path)
        for path in paths:
            fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            descriptors.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue_fd = descriptors[-1]
        for job in store.queue.jobs:
            fd = store._try_job_lock(job["job_id"])
            if fd is None:
                raise RuntimeError("checkpoint restore refuses a live attempt lease")
            descriptors.append(fd)
        yield queue_fd
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _validate(queue, plan, source):
    required = {"schema", "queue_id", "manifest_sha256", "checkpoint_path", "checkpoint_sha256",
                "expected_state_sha256", "capture_kind", "expected_running_tokens", "owners_exited", "evidence"}
    if (not isinstance(plan, dict) or set(plan) != required
            or plan["schema"] != "mind2act_eval.queue_recovery_plan.v1"
            or plan["queue_id"] != queue.queue_id or plan["manifest_sha256"] != queue.manifest_sha256
            or plan["capture_kind"] != "canonical_observation_under_lock"
            or not isinstance(plan["evidence"], str) or not plan["evidence"].strip()
            or any(not isinstance(plan[k], str) or not _SHA256.fullmatch(plan[k])
                   for k in ("checkpoint_sha256", "expected_state_sha256"))):
        raise ValueError("recovery requires an exact authorized canonical-observation plan")
    if (not isinstance(source, dict) or source.get("schema") != "mind2act_eval.queue_state.v1"
            or source.get("queue_id") != queue.queue_id or source.get("manifest_sha256") != queue.manifest_sha256
            or set(source.get("jobs", {})) != {j["job_id"] for j in queue.jobs}):
        raise ValueError("source must be a matching canonical state, not a prepared checkpoint")
    tokens, owners = {}, set()
    for job_id, row in source["jobs"].items():
        if row["status"] != "running":
            continue
        attempts = row["attempts"]
        if not 1 <= len(attempts) <= 2 or [a["attempt_number"] for a in attempts] != list(range(1, len(attempts) + 1)):
            raise ValueError("checkpoint attempt history is incomplete")
        attempt = attempts[-1]
        token = attempt["token"]
        if (not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token)
                or attempt["generation"] != row["generation"]
                or attempt["owner_hostname"] not in queue.nodes
                or type(attempt["owner_pid"]) is not int or attempt["owner_pid"] <= 0
                or not isinstance(attempt["owner_starttime"], str) or not attempt["owner_starttime"]):
            raise ValueError("original claim identity is missing or inconsistent")
        definition = row["definition"]
        inputs = attempt["inputs"]
        if (definition["job_id"] != job_id
                or inputs["input_sha256"] != hashlib.sha256(_json_bytes(definition)).hexdigest()
                or any(inputs.get(left) != definition["options"].get(right) for left, right in (
                    ("spec_sha256", "spec_sha256"), ("h5_sha256", "h5_sha256"),
                    ("metadata_sha256", "replay_metadata_sha256")))
                or (definition.get("inputs", {}).get("code_sha") is not None
                    and inputs.get("code_sha") != definition["inputs"]["code_sha"])):
            raise ValueError("checkpoint claim input pins differ from its definition")
        expected = (queue.output_root / f"queue-{queue.queue_id}" / job_id
                    / hashlib.sha256(str(row["generation"]).encode()).hexdigest()[:16]
                    / f"attempt-{attempt['attempt_number']:02d}")
        if Path(attempt["directory"]).resolve() != expected.resolve():
            raise ValueError("checkpoint attempt directory differs from its original identity")
        tokens[job_id] = token
        owners.add((attempt["owner_hostname"], attempt["owner_pid"], attempt["owner_starttime"]))
    if not tokens or tokens != plan["expected_running_tokens"] or len(set(tokens.values())) != len(tokens):
        raise ValueError("expected original running tokens do not match the complete checkpoint")
    observed = set()
    for owner in plan["owners_exited"]:
        if (set(owner) != {"hostname", "pid", "process_birth", "exited", "evidence"}
                or owner["exited"] is not True or type(owner["pid"]) is not int or owner["pid"] <= 0
                or not isinstance(owner["evidence"], str) or not owner["evidence"].strip()):
            raise ValueError("each original owner needs explicit exact exit evidence")
        identity = (owner["hostname"], owner["pid"], owner["process_birth"])
        if identity in observed:
            raise ValueError("duplicate owner exit evidence")
        observed.add(identity)
        if owner["hostname"] == queue.node.hostname:
            birth = _process_birth(owner["pid"])
            if birth == owner["process_birth"]:
                raise RuntimeError("original local claim owner is still alive")
            if birth is None:
                try:
                    os.kill(owner["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("local owner exit cannot be independently established")
    if observed != owners:
        raise ValueError("exit evidence must cover exactly the original claim owners")
    return tokens


def _check_attempt_inventory(queue, source):
    known = set()
    for row in source["jobs"].values():
        for entry in [row, *row.get("history", [])]:
            for attempt in entry.get("attempts", []):
                known.add(Path(attempt["directory"]).resolve())
    root = queue.output_root / f"queue-{queue.queue_id}"
    for path in root.glob("*/*/attempt-*"):
        if path.is_dir() and path.resolve() not in known:
            raise RuntimeError("attempt artifacts absent from the checkpoint require separate reconciliation")


def prepare_checkpoint_plan(queue, checkpoint_path, *, expected_checkpoint_sha256):
    """Read only the pinned source to produce an intentionally non-executable draft.

    This neither observes current state nor certifies owner exit. The operator
    fills those facts after maintenance; restore independently validates them.
    """
    path = Path(checkpoint_path)
    if not path.is_absolute():
        raise ValueError("checkpoint path must be absolute")
    payload = _read(path)
    if hashlib.sha256(payload).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    source = json.loads(payload)
    if (source.get("schema") != "mind2act_eval.queue_state.v1" or source.get("queue_id") != queue.queue_id
            or source.get("manifest_sha256") != queue.manifest_sha256):
        raise ValueError("source is not this queue's observed canonical state")
    claims = {job_id: row["attempts"][-1] for job_id, row in source["jobs"].items() if row["status"] == "running"}
    owners = sorted({(a["owner_hostname"], a["owner_pid"], a["owner_starttime"]) for a in claims.values()})
    return {"schema": "mind2act_eval.queue_recovery_plan.v1", "queue_id": queue.queue_id,
            "manifest_sha256": queue.manifest_sha256, "checkpoint_path": str(path),
            "checkpoint_sha256": expected_checkpoint_sha256, "expected_state_sha256": None,
            "capture_kind": "canonical_observation_under_lock",
            "expected_running_tokens": {job_id: a["token"] for job_id, a in claims.items()},
            "owners_exited": [{"hostname": h, "pid": pid, "process_birth": birth,
                               "exited": False, "evidence": ""} for h, pid, birth in owners],
            "evidence": ""}


def restore_checkpoint(queue, plan_path, *, expected_plan_sha256, check_only=False):
    """Maintenance-only exact restore, then existing seal-based owner-loss recovery.

    The operator pins proof that the raw source was observed under the canonical
    lock. A prepared journal is not accepted as that source. Remote owner exits
    are operator evidence, additionally fenced by every native ownership lock.
    """
    plan_bytes = _read(plan_path)
    if hashlib.sha256(plan_bytes).hexdigest() != expected_plan_sha256:
        raise ValueError("recovery plan SHA256 mismatch")
    plan = json.loads(plan_bytes)
    source_path = Path(plan["checkpoint_path"])
    if not source_path.is_absolute():
        raise ValueError("checkpoint path must be absolute")
    source_bytes = _read(source_path)
    if hashlib.sha256(source_bytes).hexdigest() != plan["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    source = json.loads(source_bytes)
    tokens = _validate(queue, plan, source)
    store = QueueStore(queue)
    with _maintenance(store) as lock_fd:
        before_bytes = _read(store.state_path)
        current, digest = store._read_state()
        if digest != plan["expected_state_sha256"] or hashlib.sha256(before_bytes).hexdigest() != digest:
            raise RuntimeError("current state differs from reviewed recovery CAS")
        initial = store._initial_state()
        if (set(current) != set(initial) or current["created_at"] == source["created_at"]
                or _json_bytes(current["jobs"]) != _json_bytes(initial["jobs"])):
            raise RuntimeError("restore refuses to replace non-initial or same-lifetime current facts")
        _check_attempt_inventory(queue, source)
        # Existing markers must belong to the source lifetime, not the reset.
        try:
            marker_bytes = _read(store.marker_path)
        except FileNotFoundError:
            marker_bytes = None
        if marker_bytes is not None:
            store._validate_marker(source)
        if check_only:
            _check_commit_fence(store, lock_fd, digest)
            return {"status": "validated", "plan_sha256": expected_plan_sha256,
                    "before_sha256": digest, "source_sha256": plan["checkpoint_sha256"],
                    "original_tokens": tokens}
        recovery_id = uuid.uuid4().hex  # New recovery transaction, never an old claim identity.
        audit_root = queue.state_root / f"queue-{queue.queue_id}.recoveries"
        audit_root.mkdir(parents=True, exist_ok=True); _fsync_dir(queue.state_root)
        audit = audit_root / recovery_id; audit.mkdir(); _fsync_dir(audit_root)
        for name, payload in (("before-state.json", before_bytes), ("source-checkpoint.json", source_bytes),
                              ("plan.json", plan_bytes)):
            store._exclusive_file(audit / name, payload)
        receipt = {"schema": "mind2act_eval.queue_recovery_receipt.v1", "recovery_id": recovery_id,
                   "plan_sha256": expected_plan_sha256, "before_sha256": digest,
                   "source_sha256": plan["checkpoint_sha256"], "original_tokens": tokens,
                   "audit_dir": str(audit), "status": "prepared",
                   "writer": {"hostname": socket.gethostname(), "pid": os.getpid(),
                              "process_birth": _process_birth(os.getpid()), "uid": os.getuid(),
                              "code_sha": os.environ.get("MIND2ACT_EVAL_CODE_SHA"),
                              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        store._exclusive_file(audit / "prepared.json", _json_bytes(receipt))
        if marker_bytes is None:
            store._exclusive_file(store.marker_path, _json_bytes({
                "schema": "mind2act_eval.queue_initialized.v1", "queue_id": queue.queue_id,
                "manifest_sha256": queue.manifest_sha256, "created_at": source["created_at"],
                "evidence": plan["evidence"], "recovery_id": recovery_id, "pid": os.getpid()}))
        _check_commit_fence(store, lock_fd, digest)
        store._validate_marker(source)
        temporary = store.state_path.with_name(f".restore-{recovery_id}.tmp")
        store._exclusive_file(temporary, source_bytes)
        os.replace(temporary, store.state_path); _fsync_dir(queue.state_root)
        if store._read_state()[1] != plan["checkpoint_sha256"]:
            raise RuntimeError("restored state readback differs from the observed checkpoint")
        store._validate_marker(source)
        receipt.update(status="checkpoint_restored", restored_at=_now())
        store._exclusive_file(audit / "restored.json", _json_bytes(receipt))
        # Same pre-existing owner-loss routine; all original job leases remain
        # fenced here. Valid results become await_review, never invented QA PASS.
        outcomes = {}
        for job_id, token in tokens.items():
            row = source["jobs"][job_id]
            row["history"].append({"kind": "confirmed_recovery", "evidence": plan["evidence"],
                                   "recovery_id": recovery_id, "at": _now()})
            seal_recovered = store._recover_or_fail(row, row["definition"])
            outcomes[job_id] = {"original_token": token, "seal_recovered": seal_recovered, "status": row["status"]}
        source["updated_at"] = _now()
        store._checkpoint(source, before_sha256=plan["checkpoint_sha256"], operation="reconcile_restored_owners")
        _check_commit_fence(store, lock_fd, plan["checkpoint_sha256"])
        store._validate_marker(source)
        _write_atomic(store.state_path, source)
        final_sha = hashlib.sha256(_json_bytes(source)).hexdigest()
        if store._read_state()[1] != final_sha:
            raise RuntimeError("reconciled state readback mismatch")
        receipt.update(status="reconciled", final_state_sha256=final_sha, outcomes=outcomes, finished_at=_now())
        store._exclusive_file(audit / "reconciled.json", _json_bytes(receipt))
        return receipt
