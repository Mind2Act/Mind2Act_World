#!/usr/bin/env python3
"""Install a pinned coding-agent runtime and verify every installed file.

Source downloads retain upstream licences. Robot assets come from the paired
asset release, never from an unpinned fallback download.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DATA = Path(__file__).with_name('coding_agent_runtime')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(data, name):
    if not name.endswith('.py'):
        return data
    text = data.decode()
    text = text.replace('RoboDojo_Agent_L3_Inspect_EEF', 'AgentRuntime_EEF')
    text = text.replace('RoboDojo_Agent_L3_Inspect', 'AgentRuntime')
    text = text.replace('RoboDojoActionSpec', 'RobotActionSpec')
    if name.endswith('/planner_http_worker.py'):
        text = text.replace('import sys\n', 'import sys\nfrom pathlib import Path\n', 1)
        text = text.replace('from typing import Any\n', '''from typing import Any

# Modified by Mind2Act: make the pinned package visible in standalone workers.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
''', 1)
    return text.encode()


def verify(root):
    lock_bytes = (DATA / 'source-lock.json').read_bytes()
    if (root / 'source-lock.json').read_bytes() != lock_bytes:
        raise ValueError('installed source lock differs from this release')
    lock = json.loads(lock_bytes)
    bad = [name for name, sha in lock['files'].items()
           if not (root / name).is_file() or digest((root / name).read_bytes()) != sha]
    if bad:
        raise ValueError('missing or changed runtime files: ' + ', '.join(bad))
    return {'checked': len(lock['files']), 'commit': lock['commit'], 'source_lock_sha256': digest(lock_bytes)}


def install(out, asset_root):
    manifest = json.loads((DATA / 'downloads.json').read_text())
    if out.exists() and any(out.iterdir()):
        raise ValueError('--out must be new or empty')
    out.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        if 'asset' in row:
            data = (asset_root / 'shared/robots/x5' / row['asset']).read_bytes()
        else:
            try:
                with urlopen(row['url'], timeout=60) as response:
                    data = response.read()
            except Exception as error:
                raise RuntimeError('download failed: ' + row['url']) from error
        if digest(data) != row['sha256']:
            raise ValueError('download/asset checksum mismatch: ' + row['target'])
        data = transform(data, row['target']) if row.get('transform') else data
        target = out / row['target']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, manifest))
    for name in ('source-lock.json', 'curobo-source-hashes.json'):
        (out / name).write_bytes((DATA / name).read_bytes())
    return verify(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify and args.asset_root is None:
        parser.error('--asset-root is required for installation')
    print(json.dumps(verify(args.out) if args.verify else install(args.out, args.asset_root), indent=2))


if __name__ == '__main__':
    main()
