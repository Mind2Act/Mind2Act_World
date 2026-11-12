"""Append-only, hash-checked DR3 physical trace and scene manifest."""
import hashlib
import json
from pathlib import Path


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


class AppendOnlyTrace:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("xb")
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def append(self, row):
        if row.get("physics_tick") != self.rows:
            raise ValueError("DR3 raw audit requires each consecutive physics sample")
        payload = _encoded(row)
        self.stream.write(payload)
        self.digest.update(payload)
        self.bytes += len(payload)
        self.rows += 1

    def receipt(self):
        self.stream.flush()
        return dict(schema="dr3.native-trace.v1", path=str(self.path),
                    sha256=self.digest.hexdigest(), bytes=self.bytes, rows=self.rows)

    def close(self):
        if not self.stream.closed:
            self.stream.close()


def write_manifest(path, manifest):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encoded(manifest)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return dict(schema="dr3.native-manifest.v1", path=str(path),
                sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def _checked_receipt(receipt, schema, keys):
    if set(receipt) != keys or receipt.get("schema") != schema:
        raise ValueError("DR3 raw audit receipt schema mismatch")
    path = Path(receipt["path"])
    if not path.is_absolute():
        raise ValueError("DR3 raw audit path must be absolute")
    if type(receipt["bytes"]) is not int or receipt["bytes"] <= 0:
        raise ValueError("DR3 raw audit byte count mismatch")
    if not isinstance(receipt["sha256"], str) or len(receipt["sha256"]) != 64:
        raise ValueError("DR3 raw audit digest mismatch")
    if path.stat().st_size != receipt["bytes"]:
        raise ValueError("DR3 raw audit file length mismatch")
    return path


def read_manifest(receipt):
    path = _checked_receipt(receipt, "dr3.native-manifest.v1",
                            {"schema", "path", "sha256", "bytes"})
    digest = hashlib.sha256()
    chunks = []
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            chunks.append(chunk)
            digest.update(chunk)
    if digest.hexdigest() != receipt["sha256"]:
        raise ValueError("DR3 raw manifest digest mismatch")
    value = json.loads(b"".join(chunks))
    if not isinstance(value, dict):
        raise ValueError("DR3 raw manifest must be an object")
    return value


def read_trace(receipt, terminal_tick):
    path = _checked_receipt(receipt, "dr3.native-trace.v1",
                            {"schema", "path", "sha256", "bytes", "rows"})
    if type(receipt["rows"]) is not int or receipt["rows"] != terminal_tick + 1:
        raise ValueError("DR3 raw trace sample count mismatch")
    digest, rows, byte_count = hashlib.sha256(), [], 0
    with path.open("rb") as stream:
        for expected_tick, raw in enumerate(stream):
            digest.update(raw)
            byte_count += len(raw)
            row = json.loads(raw)
            if not isinstance(row, dict) or row.get("physics_tick") != expected_tick:
                raise ValueError("DR3 raw trace physics tick coverage mismatch")
            rows.append(row)
    if (len(rows) != receipt["rows"] or byte_count != receipt["bytes"]
            or digest.hexdigest() != receipt["sha256"]):
        raise ValueError("DR3 raw trace digest/coverage mismatch")
    return rows
