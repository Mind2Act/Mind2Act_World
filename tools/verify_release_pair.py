#!/usr/bin/env python3
"""Verify code/control-file pairing without starting Isaac or reading large assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_files(root, files):
    for name, expected in files.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Invalid member: ' + name)
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        require(actual == expected, 'SHA256 mismatch: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True, type=Path)
    parser.add_argument('--expect-release', help='optional exact release tag, e.g. v1.0.5')
    args = parser.parse_args()
    root = args.asset_root.resolve(strict=True)
    code = Path(__file__).resolve().parents[1]
    pair = json.loads((root / 'RELEASE-PAIR.json').read_text())
    require(re.fullmatch(r'v\d+\.\d+\.\d+', str(pair.get('release', ''))), 'Invalid release tag in RELEASE-PAIR.json')
    if args.expect_release:
        require(pair['release'] == args.expect_release, 'Expected release %s, found %s' % (args.expect_release, pair['release']))
    expected = pair['code_commit']
    require(len(expected) == 40 and all(c in '0123456789abcdef' for c in expected),
            'Invalid code commit')
    marker = code / 'SOURCE_COMMIT'
    if marker.exists():
        require(marker.read_text().strip() == expected, 'SOURCE_COMMIT does not match assets')
    if (code / '.git').exists():
        actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'],
                                         text=True).strip()
        require(actual == expected, 'Git HEAD does not match assets')
        subprocess.run(['git', '-C', str(code), 'diff', '--quiet', 'HEAD', '--'], check=True)
    else:
        require(marker.is_file(), 'Source archives require SOURCE_COMMIT')
    check_files(code, pair['code_files'])
    check_files(root, pair['asset_control_files'])
    for name in ('old15-release-v1', 'new15-release-v1'):
        descriptor = json.loads((root / name / 'RELEASE.json').read_text())
        require(descriptor['code_commit'] == expected, 'Release binding mismatch: ' + name)
    print(json.dumps(dict(release=pair['release'], code_commit=expected,
                          paired_files_verified=True, large_assets_rehashed=False,
                          GPU_started=False)))


if __name__ == '__main__':
    main()
