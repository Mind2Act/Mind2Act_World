"""Small release-descriptor semantics and a shared deprecated-path guard.

This module does not launch workers, change files, hash the asset tree, or decide
whether a campaign is authorized. Callers retain those existing responsibilities.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Mapping
from .verified_files import file_identity


class ReleaseContractError(ValueError):
    """A release descriptor or an actual asset path violates the contract."""


_path_checks = ContextVar('admission_path_checks', default=None)


def _path_identity(path):
    try:
        info = path.lstat()
        # Sibling creation in a shared ancestor (e.g. /tmp) changes its clocks,
        # not this path's resolution. Markers are checked separately below.
        if stat.S_ISDIR(info.st_mode):
            return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))
        return (*file_identity(info), stat.S_IFMT(info.st_mode))
    except (FileNotFoundError, NotADirectoryError):
        return None


@contextmanager
def path_check_session():
    """Batch CPU path checks, proving the full observed ancestry stable on exit.

    Only this synchronous admission reuses resolutions. Nothing escapes to a
    later call; aliases, missing paths and DEPRECATED markers are rechecked
    before any validated result is returned to its consumer.
    """
    if _path_checks.get() is not None:
        yield
        return
    state = dict(resolved={}, identities={}, checked=set(), aliases=set())
    token = _path_checks.set(state)
    try:
        yield
        for path, identity in state['identities'].items():
            if _path_identity(path) != identity:
                raise ReleaseContractError('asset path identity changed during admission: ' + str(path))
        for path in state['aliases']:
            try:
                current = path.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise ReleaseContractError('cannot recheck asset alias: ' + str(path)) from exc
            if current != state['resolved'][path]:
                raise ReleaseContractError('asset alias identity changed during admission: ' + str(path))
    finally:
        _path_checks.reset(token)


def reject_deprecated_paths(paths):
    """Return canonical paths, rejecting a DEPRECATED marker on any ancestor.

    Both the original absolute ancestry and resolved ancestry are checked, so
    aliases into or out of a deprecated root cannot bypass its marker. A
    missing leaf is allowed here; its normal reader remains responsible for
    existence. Marker permission/resolution failures never count as absence.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = (paths,)
    state = _path_checks.get()
    result, checked = [], state['checked'] if state is not None else set()
    for original in paths:
        try:
            spelled_path = Path(original)
            if not spelled_path.is_absolute():
                spelled_path = Path.cwd() / spelled_path
            original_path = Path(os.path.abspath(original))
            if (state is not None and spelled_path in state['resolved']
                    and spelled_path not in state['aliases']):
                result.append(state['resolved'][spelled_path])
                continue
            if state is not None:
                for component in (spelled_path, *spelled_path.parents):
                    if component not in state['identities']:
                        state['identities'][component] = _path_identity(component)
                if any(state['identities'][c] is not None
                       and state['identities'][c][-1] == stat.S_IFLNK
                       for c in (spelled_path, *spelled_path.parents)):
                    state['aliases'].add(spelled_path)
            path = Path(original).resolve(strict=False)
            if (state is not None and spelled_path in state['resolved']
                    and state['resolved'][spelled_path] != path):
                raise ReleaseContractError('asset alias identity changed during admission: ' + str(spelled_path))
            for parent in (original_path, *original_path.parents, path, *path.parents):
                if state is not None and parent not in state['identities']:
                    state['identities'][parent] = _path_identity(parent)
                marker = parent / "DEPRECATED"
                if marker in checked:
                    continue
                try:
                    marker.lstat()
                except (FileNotFoundError, NotADirectoryError):
                    pass
                else:
                    raise ReleaseContractError(
                        f"deprecated path {original!s} resolves to {path}; DEPRECATED marker: {marker}")
                checked.add(marker)
                if state is not None:
                    state['identities'][marker] = None
            if state is not None:
                state['resolved'][spelled_path] = path
            result.append(path)
        except ReleaseContractError:
            raise
        except (OSError, RuntimeError, TypeError) as exc:
            raise ReleaseContractError(f"cannot resolve/check asset path {original!s}: {exc}") from exc
    return tuple(result)


@dataclass(frozen=True)
class ReleaseConfig:
    asset_root: Path
    descriptor_path: Path
    asset_name: str
    manifest_path: Path
    manifest_sha256: str
    instances: tuple[Mapping[str, Any], ...]
    generated_at: str
    code_commit: str | None
    code_identity_checked: bool
    raw: Mapping[str, Any]
    catalog: tuple[Mapping[str, Any], ...] = ()
    test_source_index: tuple[Mapping[str, Any], ...] = ()
    task_statuses: Mapping[str, str] | None = None

    def resolve_case(self, case_id):
        """Resolve an exact test or replay identity; never choose another seed."""
        rows = [row for row in (*self.catalog, *self.instances)
                if row.get('case_id', row['instance_id']) == case_id]
        if len(rows) != 1:
            source = next((r for r in self.test_source_index if r['case_id'] == case_id), None)
            if source is not None:
                raise ReleaseContractError('test case is not ready: ' + str(case_id))
            raise ReleaseContractError('unknown or ambiguous release case_id: ' + str(case_id))
        return rows[0]

    def select_cases(self, instance_set='test50-r5'):
        if (instance_set in {'test50-r5','coding-agent'} and self.test_source_index
                and len(self.catalog) != len(self.test_source_index)):
            raise ReleaseContractError('full test campaign is incomplete; select explicit ready cases')
        if instance_set == 'test50-r5':
            return self.catalog
        if instance_set == 'coding-agent':
            return tuple(row for row in self.catalog if row['coding_agent_selected'])
        if instance_set == 'replay':
            return tuple(row for row in (*self.instances, *self.catalog) if row.get('replay_metadata'))
        raise ReleaseContractError('unknown instance set: ' + str(instance_set))

    @property
    def ready_case_tasks(self):
        if self.task_statuses is not None:
            return tuple(sorted(t for t,s in self.task_statuses.items() if s in {'ready','partial'}))
        return tuple(sorted({r['task_id'] for r in self.catalog}))

    def select_ready_cases(self, task_id=None, instance_set='test50-r5'):
        known = {r['task_id'] for r in (*self.test_source_index,*self.instances)}
        if task_id is not None and task_id not in known:
            raise ReleaseContractError('unknown test task: ' + str(task_id))
        rows = tuple(r for r in self.catalog if task_id is None or r['task_id'] == task_id)
        if instance_set == 'test50-r5':
            return rows
        if instance_set == 'coding-agent':
            return tuple(r for r in rows if r['coding_agent_selected'])
        raise ReleaseContractError('unknown task instance set: ' + str(instance_set))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ReleaseContractError(f"release {name} must be nonempty text")
    return value


def _commit(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ReleaseContractError(f"release {name} must be a full lowercase commit")
    return value


def _instances(value):
    if not isinstance(value, list) or len(value) != 45:
        raise ReleaseContractError("release instances must contain the explicit 15 x 3 entries")
    keys, tasks, result = set(), {}, []
    for row in value:
        if not isinstance(row, dict):
            raise ReleaseContractError("release instance must be an object")
        task = _text(row.get("task_id"), "instance task_id")
        _text(row.get("instance_id"), "instance instance_id")
        tier = row.get("difficulty")
        if tier not in {"easy", "medium", "hard"} or (task, tier) in keys:
            raise ReleaseContractError("release instances have an invalid or duplicate task/tier")
        keys.add((task, tier))
        tasks.setdefault(task, set()).add(tier)
        result.append(MappingProxyType(dict(row)))
    if len(tasks) != 15 or any(tiers != {"easy", "medium", "hard"} for tiers in tasks.values()):
        raise ReleaseContractError("release instances must cover all three tiers of 15 tasks")
    return tuple(result)


@path_check_session()
def load_release(assetroot, *, verified_code_commit=None, code_root=None):
    """Read RELEASE.json and return the unified, semantically checked config.

    The producer owns complete asset hashing. This function checks descriptor
    fields and paths, and compares code_commit only with an available actual
    SOURCE_COMMIT or an identity already verified by the caller. It adds no
    campaign/visual/score gate and does not inherit a prior generation's result.
    """
    root, = reject_deprecated_paths((assetroot,))
    descriptor, = reject_deprecated_paths((root / "RELEASE.json",))
    if descriptor.parent != root:
        raise ReleaseContractError("RELEASE.json must stay inside its asset root")
    actual_root = Path(code_root) if code_root is not None else Path(__file__).resolve().parents[1]
    reject_deprecated_paths((actual_root,))
    try:
        data = json.loads(descriptor.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"cannot read release descriptor {descriptor}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReleaseContractError("RELEASE.json must contain an object")
    name = _text(data.get("asset_name"), "asset_name")
    if name != root.name:
        raise ReleaseContractError("release asset_name differs from its actual root directory")
    manifest = _text(data.get("manifest"), "manifest")
    if Path(manifest).is_absolute():
        raise ReleaseContractError("release manifest must be relative inside its asset root")
    manifest_path, = reject_deprecated_paths((root / manifest,))
    if not manifest_path.is_relative_to(root):
        raise ReleaseContractError("release manifest must stay inside its asset root")
    if not manifest_path.is_file():
        raise ReleaseContractError("release manifest is not an existing file")
    digest = data.get("manifest_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ReleaseContractError("release manifest_sha256 must be a full lowercase SHA256")
    generated = _text(data.get("generated_at"), "generated_at")
    try:
        datetime.fromisoformat(generated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseContractError("release generated_at must be an ISO timestamp") from exc
    instances = _instances(data.get("instances"))
    expected = _commit(data["code_commit"], "code_commit") if "code_commit" in data else None
    identities = []
    if expected is not None:
        if verified_code_commit is not None:
            identities.append(("verified code", _commit(verified_code_commit, "verified code identity")))
        sources = [actual_root / 'SOURCE_COMMIT']
        if actual_root.resolve() == Path(__file__).resolve().parents[1]:
            sources.append(Path(__file__).with_name('SOURCE_COMMIT'))
        for candidate in dict.fromkeys(sources):
            source, = reject_deprecated_paths((candidate,))
            try:
                with source.open() as stream:
                    actual = stream.read(256).strip()
            except FileNotFoundError:
                pass
            except (OSError, UnicodeError) as exc:
                raise ReleaseContractError(f"cannot read actual SOURCE_COMMIT {source}: {exc}") from exc
            else:
                identities.append(("SOURCE_COMMIT", _commit(actual, "SOURCE_COMMIT")))
        # A persisted marker cannot override a newer or edited checkout.
        if any(label == 'SOURCE_COMMIT' for label, _ in identities) and (actual_root / '.git').exists():
            try:
                head = subprocess.check_output(
                    ['git', '-C', str(actual_root), 'rev-parse', 'HEAD'], text=True).strip()
                subprocess.run(['git', '-C', str(actual_root), 'diff', '--quiet', 'HEAD', '--'],
                               check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ReleaseContractError('cannot verify a clean Git checkout for SOURCE_COMMIT') from exc
            identities.append(('Git HEAD', _commit(head, 'Git HEAD')))
        for label, actual in identities:
            if actual != expected:
                raise ReleaseContractError(f"release code_commit differs from {label}")
    catalog, source_index, task_statuses = (), (), None
    if 'catalog' in data and 'task_shards' in data:
        raise ReleaseContractError('release must select one full catalog or explicit task shards')
    if 'catalog' in data:
        from .instance_catalog import load_catalog
        catalog = load_catalog(root, data['catalog'], tasks={r['task_id'] for r in instances})
    if 'task_shards' in data:
        from .task_shards import load_task_shards
        catalog, source_index, task_statuses = load_task_shards(
            root, data['task_shards'], tasks={r['task_id'] for r in instances})
    replay_ids = {row['instance_id'] for row in instances}
    if any(row['case_id'] in replay_ids for row in (*catalog, *source_index)):
        raise ReleaseContractError('test catalog and replay instance identities must be distinct')
    return ReleaseConfig(root, descriptor, name, manifest_path, digest, instances,
                         generated, expected, bool(identities), MappingProxyType(data), catalog,
                         source_index, task_statuses)
