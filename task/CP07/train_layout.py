"""Frozen authority for held-out layouts generated with the training protocol."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path

from mind2act_eval.cp07_layout_identity import MODE, SOURCE, PROTOCOL, digest


@lru_cache(maxsize=8192)
def _file_hash(path, size, mtime_ns, ctime_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticate(spec, assets):
    from mind2act_eval.sim.source_assets import read_source_bytes
    from mind2act_eval.cp07_layout_identity import validate_identity
    authority, row = validate_identity(spec)
    raw = read_source_bytes(assets, 'cp07_binding', row['binding_sha256'])
    for item in authority['closures'][spec.difficulty]:
        path = (assets.root / item['path']).resolve(strict=True)
        if not path.is_relative_to(assets.root.resolve()):
            raise ValueError('CP07 training-layout asset escapes delivery')
        stat=path.stat()
        if _file_hash(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns) != item['sha256']:
            raise ValueError('CP07 training-layout base geometry authority differs')
    read_source_bytes(assets, 'station', row['station_sha256'])
    return json.loads(raw)
