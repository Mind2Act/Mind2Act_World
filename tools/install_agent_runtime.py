#!/usr/bin/env python3
"""Install the pinned agent runtime from upstream sources and robot assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DATA = Path(__file__).with_name('agent_runtime')
LOCK_SHA256 = 'a70f8c497901a6aee695657a54988a547b5940ec30b6bc0631edc89b952e9e49'
CUROBO_SHA256 = 'd018aef2af8d8fffd2aa3f11526d08ff4912ca8ae81b96051361f77f620633b7'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_lock():
    raw = (DATA / 'source-lock.json').read_bytes()
    if digest(raw) != LOCK_SHA256:
        raise ValueError('runtime source lock differs from the approved pin')
    if digest((DATA / 'curobo-source-hashes.json').read_bytes()) != CUROBO_SHA256:
        raise ValueError('cuRobo source lock differs from the approved pin')
    return raw, json.loads(raw)


def verify(root):
    raw, lock = source_lock()
    if (root / 'source-lock.json').read_bytes() != raw:
        raise ValueError('installed source lock differs from this release')
    expected = dict(lock['files'])
    for row in json.loads((DATA / 'downloads.json').read_text()):
        expected.setdefault(row['target'], row['installed_sha256'])
    checked, skipped = 0, []
    for name, sha in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('runtime file escapes source root: ' + name)
        # These four historical drive variants are unused by the pinned runtime.
        if '/configuration/X5A_temp_' in name and not path.exists():
            skipped.append(name)
            continue
        if digest(path.read_bytes()) != sha:
            raise ValueError('missing or changed runtime file: ' + name)
        checked += 1
    return dict(checked=checked, commit=lock['commit'], source_lock_sha256=digest(raw),
                unused_drive_variants_not_installed=skipped)


def install(out, asset_root):
    raw, lock = source_lock()
    if out.exists() and any(out.iterdir()):
        raise ValueError('--out must be new or empty')
    out.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        if 'asset' in row:
            data = (asset_root / 'shared/robots/x5' / row['asset']).read_bytes()
        else:
            with urlopen(row['url'], timeout=60) as response:
                data = response.read()
        if digest(data) != row['sha256']:
            raise ValueError('download/asset checksum mismatch: ' + row['target'])
        if 'patched_file' in row:
            data = (DATA / row['patched_file']).read_bytes()
        expected = lock['files'].get(row['target'], row['installed_sha256'])
        if digest(data) != expected:
            raise ValueError('installed checksum mismatch: ' + row['target'])
        target = out / row['target']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, json.loads((DATA / 'downloads.json').read_text())))
    (out / 'source-lock.json').write_bytes(raw)
    (out / 'curobo-source-hashes.json').write_bytes((DATA / 'curobo-source-hashes.json').read_bytes())
    return verify(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    if not args.verify and args.asset_root is None:
        parser.error('--asset-root is required for installation')
    print(json.dumps(verify(args.out) if args.verify else install(args.out, args.asset_root), indent=2))


if __name__ == '__main__':
    main()
