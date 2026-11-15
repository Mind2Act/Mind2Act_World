"""Immutable static campaign partitions for coordinated multi-node batches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import socket

from mind2act_eval.batch_resources import BatchResourceConfig
from mind2act_eval.contracts import InstanceSpec

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAMPAIGN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}\Z")


@dataclass(frozen=True)
class CampaignPartition:
    campaign_id: str
    output_root: Path
    lock_root: Path
    label: str
    hostname: str
    machine_id_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: bytes
    resource_config_path: Path
    resource_config_sha256: str
    resource_config: BatchResourceConfig
    campaign_manifest_path: Path
    campaign_manifest_sha256: str
    campaign_partitions: tuple[tuple[str, str, str, str, str, int, int], ...]


@dataclass(frozen=True)
class NodeIdentity:
    hostname: str
    machine_id_sha256: str


def _canonical_hostname(value):
    if (not isinstance(value, str) or not value or len(value) > 253
            or value != value.strip() or any(ord(char) < 33 or char in "/\\" for char in value)):
        raise RuntimeError("this node's hostname is malformed")
    return value.lower()


def node_identity(machine_id_path=None, *, hostname=None):
    """Read the kernel hostname and OS machine ID; accept no CLI/environment alias."""
    hostname = _canonical_hostname(socket.gethostname() if hostname is None else hostname)
    path = Path(machine_id_path or "/etc/machine-id")
    if not path.is_file() and path == Path("/etc/machine-id"):
        path = Path("/var/lib/dbus/machine-id")
    try:
        machine_id = path.read_text().strip()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("cannot establish this node's machine identity") from exc
    if not machine_id or any(char not in "0123456789abcdefABCDEF-" for char in machine_id):
        raise RuntimeError("this node's machine identity is malformed")
    return NodeIdentity(hostname, hashlib.sha256(machine_id.lower().encode("ascii")).hexdigest())


def node_lock_key(hostname):
    """Hash the canonical container hostname for this UID's node-scoped lock."""
    return hashlib.sha256(_canonical_hostname(hostname).encode("utf-8")).hexdigest()


def _contained_file(base, value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"campaign {name} must be a relative file path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"campaign {name} must be relative to the campaign manifest")
    path = (base / candidate).resolve(strict=True)
    if not path.is_relative_to(base):
        raise ValueError(f"campaign {name} escapes its manifest directory")
    if not path.is_file():
        raise ValueError(f"campaign {name} must be a file")
    return path


def load_partition(path, *, expected_sha256, local_identity=None):
    """Validate a shared immutable manifest and select only this node's shard.

    Manifest schema::

        {"schema":"mind2act_eval.distributed_campaign.v1",
         "campaign_id":"...", "output_root":"OUTPUT/results",
         "lock_root":"OUTPUT/.mind2act-eval-<uid>", "partitions":[
           {"hostname":"n1-container-host", "machine_id_sha256":"<sha256>", "label":"N1",
            "manifest":"N1-jobs.json", "manifest_sha256":"...",
            "resource_config":"N1-resources.json",
            "resource_config_sha256":"..."}]}
    """
    path = Path(path).resolve(strict=True)
    raw_bytes = path.read_bytes()
    expected = expected_sha256
    actual = hashlib.sha256(raw_bytes).hexdigest()
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected) or actual != expected:
        raise ValueError("campaign manifest digest mismatch")
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("campaign manifest is invalid JSON") from exc
    if (not isinstance(raw, dict) or set(raw) != {"schema", "campaign_id", "output_root", "lock_root", "partitions"}
            or raw["schema"] != "mind2act_eval.distributed_campaign.v1"):
        raise ValueError("invalid distributed campaign manifest schema/fields")
    campaign_id = raw["campaign_id"]
    if not isinstance(campaign_id, str) or not _CAMPAIGN_ID.fullmatch(campaign_id):
        raise ValueError("campaign_id must be a safe path component")
    if (not isinstance(raw["output_root"], str) or not Path(raw["output_root"]).is_absolute()
            or not isinstance(raw["lock_root"], str) or not Path(raw["lock_root"]).is_absolute()):
        raise ValueError("campaign output_root and lock_root must be absolute paths")
    output_root = Path(raw["output_root"])
    lock_root = Path(raw["lock_root"])
    if ".." in output_root.parts or ".." in lock_root.parts:
        raise ValueError("campaign output_root and lock_root cannot traverse parent paths")
    partitions = raw["partitions"]
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("campaign partitions must be a nonempty list")
    allowed = {"hostname", "machine_id_sha256", "label", "manifest", "manifest_sha256",
               "resource_config", "resource_config_sha256"}
    hostnames, labels, episode_keys = set(), set(), set()
    partition_receipts = []
    by_node = {}
    base = path.parent.resolve(strict=True)
    local = node_identity() if local_identity is None else local_identity
    if not isinstance(local, NodeIdentity):
        raise ValueError("local node identity is invalid")
    for item in partitions:
        if not isinstance(item, dict) or set(item) != allowed:
            raise ValueError("invalid campaign partition fields")
        hostname = item["hostname"]
        machine_sha = item["machine_id_sha256"]
        label = item["label"]
        if not isinstance(hostname, str) or _canonical_hostname(hostname) != hostname:
            raise ValueError("partition hostname must be canonical lowercase hostname")
        if not isinstance(machine_sha, str) or not _SHA256.fullmatch(machine_sha):
            raise ValueError("partition machine_id_sha256 must be lowercase SHA256")
        if not isinstance(label, str) or not _LABEL.fullmatch(label):
            raise ValueError("partition label must be a safe path component")
        if hostname in hostnames or label in labels:
            raise ValueError("campaign hostnames and labels must be unique")
        hostnames.add(hostname)
        labels.add(label)
        manifest = _contained_file(base, item["manifest"], "job manifest")
        resource = _contained_file(base, item["resource_config"], "resource config")
        manifest_bytes = manifest.read_bytes()
        manifest_sha = item["manifest_sha256"]
        if (not isinstance(manifest_sha, str) or not _SHA256.fullmatch(manifest_sha)
                or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha):
            raise ValueError("campaign job manifest digest mismatch")
        resource_bytes = resource.read_bytes()
        resource_sha = item["resource_config_sha256"]
        if (not isinstance(resource_sha, str) or not _SHA256.fullmatch(resource_sha)
                or hashlib.sha256(resource_bytes).hexdigest() != resource_sha):
            raise ValueError("campaign resource config digest mismatch")
        try:
            resource_raw = json.loads(resource_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError("campaign resource config is invalid JSON") from exc
        if (not isinstance(resource_raw, dict)
                or set(resource_raw) != {"schema", "rss_limit_bytes", "max_parallel"}
                or resource_raw["schema"] != "mind2act_eval.batch_resources.v1"):
            raise ValueError("invalid campaign resource config schema/fields")
        resource_config = BatchResourceConfig(resource_raw["rss_limit_bytes"],
                                              resource_raw["max_parallel"], str(resource), resource_sha)
        try:
            entries = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError("campaign job manifest is invalid JSON") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("campaign job manifest must be a nonempty list")
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("spec"), str)
                    or not isinstance(entry.get("spec_sha256"), str)
                    or not isinstance(entry.get("episode_key"), dict)
                    or set(entry["episode_key"]) != {"task_id", "instance_id", "difficulty"}
                    or any(not isinstance(value, str) or not value
                           for value in entry["episode_key"].values())):
                raise ValueError("campaign jobs must name a spec, SHA256, and episode_key")
            key = tuple(entry["episode_key"][name] for name in ("task_id", "instance_id", "difficulty"))
            if key in episode_keys:
                raise ValueError("campaign partitions contain a duplicate episode")
            episode_keys.add(key)
            if hostname == local.hostname:
                if machine_sha != local.machine_id_sha256:
                    raise ValueError("campaign machine_id_sha256 does not match this hostname")
                spec_path = Path(entry["spec"]).resolve(strict=True)
                spec_bytes = spec_path.read_bytes()
                if hashlib.sha256(spec_bytes).hexdigest() != entry["spec_sha256"]:
                    raise ValueError("campaign job spec digest mismatch")
                try:
                    spec = InstanceSpec.from_dict(json.loads(spec_bytes))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"campaign job spec is invalid: {spec_path}") from exc
                actual_key = (spec.task_id, spec.instance_id, spec.difficulty)
                if actual_key != key:
                    raise ValueError("campaign episode_key does not match its local spec")
        by_node[hostname] = (label, machine_sha, manifest, manifest_sha, manifest_bytes,
                                resource, resource_sha, resource_config)
        partition_receipts.append((label, hostname, machine_sha, manifest_sha, resource_sha,
                                   resource_config.rss_limit_bytes, resource_config.max_parallel))
    selected = by_node.get(local.hostname)
    if selected is None:
        raise ValueError("this hostname is not assigned a partition in the campaign")
    label, machine_sha, manifest, manifest_sha, manifest_bytes, resource, resource_sha, resource_config = selected
    return CampaignPartition(campaign_id, output_root, lock_root, label, local.hostname, machine_sha,
                             manifest, manifest_sha, manifest_bytes,
                             resource, resource_sha, resource_config, path, actual,
                             tuple(sorted(partition_receipts)))
