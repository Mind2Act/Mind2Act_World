"""A stale checkout stamp must not turn new code into an old release."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import setuptools

from mind2act_eval.release_contract import ReleaseContractError, load_release
from test_release_contract import release


def git(root, *args):
    return subprocess.check_output(
        ['git', '-C', str(root), '-c', 'user.name=Test', '-c',
         'user.email=test@localhost', *args], text=True).strip()


@pytest.fixture(params=['directory', 'file'])
def checkout(tmp_path, request):
    root = tmp_path / 'code'
    root.mkdir()
    flags = ['--separate-git-dir', str(tmp_path / 'git-data')] if request.param == 'file' else []
    git(root, 'init', '-q', *flags)
    (root / 'runtime.py').write_text('version = 1\n')
    git(root, 'add', 'runtime.py')
    git(root, 'commit', '-qm', 'initial')
    return root


@pytest.fixture
def build_identity(monkeypatch):
    # Import the real build entrypoint without executing setuptools commands.
    monkeypatch.setattr(setuptools, 'setup', lambda **kwargs: None)
    spec = importlib.util.spec_from_file_location('identity_build', Path(__file__).parents[1] / 'setup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rejects_stale_stamp_and_modified_checkout(checkout, build_identity):
    build_identity.ROOT = checkout
    old = git(checkout, 'rev-parse', 'HEAD')
    assert build_identity.source_commit().strip() == old  # Git checkout without a marker
    (checkout / 'SOURCE_COMMIT').write_text(old)
    (checkout / 'runtime.py').write_text('version = 2\n')
    git(checkout, 'commit', '-qam', 'next')
    with pytest.raises(ValueError, match='differs from Git HEAD'):
        build_identity.source_commit()
    new = git(checkout, 'rev-parse', 'HEAD')
    (checkout / 'SOURCE_COMMIT').write_text(new)
    assert build_identity.source_commit().strip() == new
    (checkout / 'runtime.py').write_text('version = 3\n')
    with pytest.raises(subprocess.CalledProcessError):
        build_identity.source_commit()


def test_runtime_rejects_stale_stamp_and_modified_checkout(checkout, release):
    assets, data = release
    old = git(checkout, 'rev-parse', 'HEAD')
    (checkout / 'SOURCE_COMMIT').write_text(old)
    data['code_commit'] = old
    (assets / 'RELEASE.json').write_text(json.dumps(data))
    assert load_release(assets, code_root=checkout).code_identity_checked
    (checkout / 'runtime.py').write_text('version = 2\n')
    git(checkout, 'commit', '-qam', 'next')
    with pytest.raises(ReleaseContractError, match='Git HEAD'):
        load_release(assets, code_root=checkout)
    new = git(checkout, 'rev-parse', 'HEAD')
    (checkout / 'SOURCE_COMMIT').write_text(new)
    data['code_commit'] = new
    (assets / 'RELEASE.json').write_text(json.dumps(data))
    assert load_release(assets, code_root=checkout).code_identity_checked
    (checkout / 'runtime.py').write_text('version = 3\n')
    with pytest.raises(ReleaseContractError, match='clean Git checkout'):
        load_release(assets, code_root=checkout)


def test_archive_can_build_from_stamp_without_git(tmp_path, build_identity):
    build_identity.ROOT = tmp_path
    with pytest.raises(ValueError, match='explicit full SOURCE_COMMIT'):
        build_identity.source_commit()
    (tmp_path / 'SOURCE_COMMIT').write_text('a' * 40)
    assert build_identity.source_commit() == 'a' * 40 + '\n'
