"""Frozen test-instance catalogue; no native SDK or replay dependencies."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

from .release_contract import ReleaseContractError, reject_deprecated_paths
from .verified_files import verify_sha256
from .agent_names import agent_field, with_agent_aliases

SCHEMA = 'mind2act_eval.instance_catalog.v1'
INSTANCE_SET = 'test50-r5'
TIERS = frozenset(('easy', 'medium', 'hard'))


def sha256(path):
    # Shared filesystems may keep stat timestamps unchanged after a write.
    # Always read the bytes; inode/size/timestamps are not a content identity.
    path = Path(path)
    before = path.stat()
    key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if key != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ReleaseContractError('asset changed while hashing: ' + str(path))
    result = digest.hexdigest()
    return result


def relative_path(root, value):
    if (not isinstance(value, str) or not value or '\\' in value
            or ':' in value or PurePosixPath(value).is_absolute()
            or '..' in PurePosixPath(value).parts):
        raise ReleaseContractError('catalog path must be relative inside its asset root')
    path, = reject_deprecated_paths((Path(root) / value,))
    canonical_root, = reject_deprecated_paths((root,))
    if not path.is_relative_to(canonical_root):
        raise ReleaseContractError('catalog path escapes its asset root')
    return path


def digest_field(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ReleaseContractError('catalog requires a full lowercase SHA256')
    return value


def file_descriptor(root, descriptor, *, verify=False):
    if not isinstance(descriptor, dict):
        raise ReleaseContractError('catalog file descriptor must be an object')
    path = relative_path(root, descriptor.get('path'))
    expected = digest_field(descriptor.get('sha256'))
    if verify:
        try:
            verify_sha256(path, expected)
        except (OSError, ValueError) as exc:
            raise ReleaseContractError('catalog file SHA256 differs: ' + descriptor['path']) from exc
    return path


def asset_descriptor(root, descriptor, *, verify=False):
    if not isinstance(descriptor, dict):
        raise ReleaseContractError('catalog asset descriptor must be an object')
    path = relative_path(root, descriptor.get('path'))
    if 'files' not in descriptor and 'manifest' not in descriptor:
        file_descriptor(root, descriptor, verify=verify)
        return
    if 'manifest' in descriptor:
        if 'files' in descriptor:
            raise ReleaseContractError('catalog directory must select one file-list representation')
        manifest = file_descriptor(root, descriptor['manifest'], verify=verify)
        if not verify:
            return
        data = json.loads(manifest.read_text())
        if data.get('schema') != 'mind2act_eval.asset_directory.v1':
            raise ReleaseContractError('invalid catalog asset directory manifest')
        files = data.get('files')
    else:
        files = descriptor['files']
    if not isinstance(files, list) or not files or 'sha256' in descriptor:
        raise ReleaseContractError('catalog directory asset requires its explicit file list')
    names = set()
    for member in files:
        member_path = file_descriptor(path, member, verify=verify)
        if member['path'] in names:
            raise ReleaseContractError('duplicate catalog directory member')
        names.add(member['path'])
        if verify and not member_path.is_relative_to(path):
            raise ReleaseContractError('catalog directory member escapes its asset')
    if verify:
        actual = set()
        for member in path.rglob('*'):
            relative_path(path, str(member.relative_to(path)))
            if member.is_symlink() and member.is_dir():
                raise ReleaseContractError('catalog directory cannot contain an unexpanded directory symlink')
            if member.is_file():
                actual.add(str(member.relative_to(path)))
        if not path.is_dir() or actual != names:
            raise ReleaseContractError('catalog directory asset file set differs: ' + descriptor['path'])


def load_catalog(root, descriptor, *, tasks):
    path = file_descriptor(root, descriptor, verify=True)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError) as exc:
        raise ReleaseContractError('cannot read instance catalog') from exc
    if (not isinstance(data, dict) or data.get('schema') != SCHEMA
            or data.get('instance_set') != INSTANCE_SET
            or data.get('cases_per_cell') != 50 or agent_field(data, 'agent_per_cell') != 5
            or not isinstance(data.get('tasks'), list) or len(data['tasks']) != len(tasks)
            or set(data['tasks']) != set(tasks)):
        raise ReleaseContractError('catalog must declare its exact task grid and test50-r5 contract')
    rows = data.get('instances')
    if not isinstance(rows, list) or len(rows) != len(tasks) * 3 * 50:
        raise ReleaseContractError('catalog requires 50 actual cases per task/difficulty')
    ids, cells, coding, result = set(), Counter(), Counter(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ReleaseContractError('catalog instance must be an object')
        row = with_agent_aliases(row)
        case = row.get('case_id')
        cell = (row.get('task_id'), row.get('difficulty'))
        if (not isinstance(case, str) or not case.strip() or case in ids
                or cell[0] not in tasks or cell[1] not in TIERS
                or row.get('instance_id', case) != case
                or type(row.get('agent_selected')) is not bool):
            raise ReleaseContractError('catalog has duplicate/invalid case identity or subset')
        ids.add(case)
        try:
            from .budgets import validate_e2e_fields
            validate_e2e_fields(row)
        except ValueError as exc:
            raise ReleaseContractError(str(exc)) from exc
        cells[cell] += 1
        coding[cell] += row['agent_selected']
        file_descriptor(root, row.get('spec'))
        digest_field(row['spec'].get('semantic_sha256'))
        assets = row.get('assets')
        if not isinstance(assets, dict) or any(not isinstance(k, str) or not k for k in assets):
            raise ReleaseContractError('catalog instance requires an explicit asset mapping')
        for asset in assets.values():
            asset_descriptor(root, asset)
        metadata = row.get('replay_metadata')
        if metadata is not None:
            file_descriptor(root, metadata)
        file_descriptor(root, row.get('asset_closure'))
        normalized = dict(row, instance_id=case, replay_metadata=metadata)
        result.append(MappingProxyType(normalized))
    expected = {(task, tier) for task in tasks for tier in TIERS}
    if set(cells) != expected or set(cells.values()) != {50} or set(coding.values()) != {5}:
        raise ReleaseContractError('catalog requires fixed50 and five agent cases in every cell')
    return tuple(result)


def verify_case_assets(root, row, spec):
    declared = row['assets']
    if dict(spec.assets) != {key: value['path'] for key, value in declared.items()}:
        raise ReleaseContractError('spec assets differ from its catalog entry')
    for asset in declared.values():
        asset_descriptor(root, asset, verify=True)
    closure_path = file_descriptor(root, row['asset_closure'], verify=True)
    closure = json.loads(closure_path.read_text())
    if (closure.get('schema') != 'mind2act_eval.asset_closure.v1'
            or closure.get('root_assets') != dict(spec.assets)
            or not isinstance(closure.get('files'), list)
            or closure.get('unresolved') != []):
        raise ReleaseContractError('catalog asset closure differs from its spec or has unresolved dependencies')
    paths = set()
    for member in closure['files']:
        path = file_descriptor(root, member, verify=True)
        if path in paths:
            raise ReleaseContractError('duplicate asset in catalog closure')
        paths.add(path)
    for relative in spec.assets.values():
        asset = relative_path(root, relative)
        required = {p.resolve() for p in asset.rglob('*') if p.is_file()} if asset.is_dir() else {asset}
        if not required <= paths:
            raise ReleaseContractError('catalog closure omits a declared root asset')
