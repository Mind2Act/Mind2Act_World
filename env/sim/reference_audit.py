"""Read every authored USD/USDC/USDZ asset token before delivering a release.

Also scan literal forbidden prefixes in payload bytes and decoded JSON. This
does not modify data, resolve missing relative assets, start Kit, or qualify RGB.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import stat
import zipfile

FORBIDDEN = r'/(?:home/|shared/users)|@/(?:root|home)/'
USD = {'.usd', '.usda', '.usdc'}


def audit(root):
    from pxr import Ar, Sdf, UsdUtils

    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('release payload root must be an existing directory')
    pattern = re.compile(FORBIDDEN.encode())
    text_pattern = re.compile(FORBIDDEN)
    references, byte_matches, json_matches, issues = [], [], [], []
    layers = packages = 0
    aliases = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            # Path.rglob suppresses enumeration errors on some Python versions.
            # Materialize each directory so partial/failed reads cannot pass.
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            issues.append(dict(file=str(directory), kind='unreadable_directory', detail=str(exc)))
            continue
        for entry in children:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                    continue
                if stat.S_ISLNK(info.st_mode):
                    try:
                        info = entry.stat(follow_symlinks=True)
                    except FileNotFoundError:
                        info = None
                    if info is None or stat.S_ISDIR(info.st_mode):
                        issues.append(dict(file=str(path), kind='unscanned_symlink',
                                           detail='dangling or directory symlink is not a complete payload scan'))
                        continue
                if stat.S_ISREG(info.st_mode):
                    aliases.setdefault((info.st_dev, info.st_ino), []).append(str(path))
            except OSError as exc:
                issues.append(dict(file=str(path), kind='unreadable_entry', detail=str(exc)))

    def scan_bytes(stream, identifier):
        found, tail = set(), b''
        while block := stream.read(1024 * 1024):
            data = tail + block
            found.update(m.group().decode() for m in pattern.finditer(data))
            tail = data[-64:]
        if found:
            byte_matches.append(dict(file=identifier, prefixes=sorted(found)))

    def scan_json(stream, identifier):
        try:
            def unique_object(pairs):
                result = {}
                for key,value in pairs:
                    if key in result:
                        raise ValueError('duplicate JSON object key: ' + key)
                    result[key] = value
                return result
            value = json.load(stream, object_pairs_hook=unique_object)
            def walk(value, pointer, position='value'):
                if isinstance(value, str) and text_pattern.search(value):
                    json_matches.append(dict(file=identifier, pointer=pointer,
                                             position=position, value=value))
                elif isinstance(value, dict):
                    for k,v in value.items():
                        child = pointer + '/' + k.replace('~', '~0').replace('/', '~1')
                        walk(k, child, 'key')
                        walk(v, child)
                elif isinstance(value, list):
                    for i,v in enumerate(value):
                        walk(v, pointer + '/' + str(i))
            walk(value, '')
        except (ValueError, UnicodeError) as exc:
            issues.append(dict(file=identifier, kind='unreadable_json', detail=str(exc)))

    def layer(identifier):
        nonlocal layers
        try:
            source = Sdf.Layer.FindOrOpen(identifier)
            if source is None:
                raise ValueError('USD layer cannot be opened')
            clone = Sdf.Layer.CreateAnonymous('reference-audit.usda')
            clone.TransferContent(source)
            tokens = set()
            UsdUtils.ModifyAssetPaths(clone, lambda raw: (tokens.add(raw) or raw) if raw else raw)
            layers += 1
            for raw in sorted(tokens):
                if raw.startswith(('/', '\\', '~', '$')) or re.match(r'^[A-Za-z][\w+.-]*:', raw):
                    references.append(dict(layer=identifier, asset_path=raw,
                                           kind='non_relative_asset_reference'))
        except Exception as exc:
            issues.append(dict(file=identifier, kind='unreadable_usd', detail=str(exc)))

    def archive(file, identifier):
        nonlocal packages
        try:
            with zipfile.ZipFile(file) as package:
                packages += 1
                names = package.namelist()
                if len(names) != len(set(names)):
                    raise ValueError('duplicate USDZ member names are ambiguous')
                if not any(Path(name).suffix.lower() in USD for name in names):
                    raise ValueError('USDZ contains no authored USD layer')
                for name in names:
                    if name.endswith('/'):
                        continue
                    child = Ar.JoinPackageRelativePath(identifier, name)
                    with package.open(name) as stream:
                        scan_bytes(stream, child)
                    suffix = Path(name).suffix.lower()
                    if suffix in USD:
                        layer(child)
                    elif suffix == '.json':
                        with package.open(name) as stream:
                            scan_json(stream, child)
                    elif suffix == '.usdz':
                        archive(io.BytesIO(package.read(name)), child)
        except Exception as exc:
            issues.append(dict(file=identifier, kind='unreadable_usdz', detail=str(exc)))

    for names in aliases.values():
        path = Path(names[0])
        with path.open('rb') as stream:
            scan_bytes(stream, str(path))
        # Inode deduplication applies to bytes, not semantic content types:
        # an earlier .bin alias must not hide a later USDC/JSON/USDZ name.
        typed = {}
        for name in names:
            suffix = Path(name).suffix.lower()
            kind = 'usd' if suffix in USD else suffix
            typed.setdefault(kind, Path(name))
        if 'usd' in typed:
            layer(str(typed['usd']))
        if '.usdz' in typed:
            archive(typed['.usdz'], str(typed['.usdz']))
        if '.json' in typed:
            with typed['.json'].open('rb') as stream:
                scan_json(stream, str(typed['.json']))
    if not aliases or not layers:
        issues.append(dict(kind='no_usd_payload'))
    return dict(schema='mind2act_eval.release_reference_audit.v1', root=str(root),
                files=sum(map(len, aliases.values())), unique_files=len(aliases),
                usd_layers=layers, usdz_packages=packages,
                passed=not (references or byte_matches or json_matches or issues),
                non_relative_references=references, forbidden_byte_matches=byte_matches,
                forbidden_json_values=json_matches, issues=issues,
                source_modified=False, kit_started=False, gpu_started=False,
                qualification='authored reference portability and literal prefix scan; not asset existence, appearance or score acceptance')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.resolve().is_relative_to(args.root.resolve()):
        parser.error('keep diagnostic reports outside the delivered payload')
    result = audit(args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
