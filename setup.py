"""Preserve source identity in wheels and source distributions."""
from pathlib import Path
import re
import subprocess

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def source_commit():
    source = ROOT / 'SOURCE_COMMIT'
    marker = source.read_text().strip() if source.is_file() else None
    if (ROOT / '.git').exists():
        value = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if marker is not None and marker != value:
            raise ValueError('SOURCE_COMMIT differs from Git HEAD; refresh it after checkout')
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--'], check=True)
    else:
        value = marker
    if value is None or re.fullmatch('[0-9a-f]{40}', value) is None:
        raise ValueError('building requires an explicit full SOURCE_COMMIT or Git identity')
    return value + '\n'


class BuildPy(build_py):
    def run(self):
        super().run()
        (Path(self.build_lib) / 'mind2act_eval' / 'SOURCE_COMMIT').write_text(source_commit())


class SourceDist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        (Path(base_dir) / 'SOURCE_COMMIT').write_text(source_commit())


# Keep the public import namespace while exposing functional areas at repository root.
PACKAGE_DIRS = {
    'mind2act_eval': 'eval',
    'mind2act_eval.tasks': 'task',
    'mind2act_eval.sim': 'env/sim',
    'mind2act_eval.robot': 'env_cfg',
    'mind2act_eval.adapters': 'policies',
}
packages = []
for namespace, directory in PACKAGE_DIRS.items():
    packages.append(namespace)
    packages.extend(namespace + '.' + child for child in find_packages(directory))

setup(packages=packages, package_dir=PACKAGE_DIRS,
      package_data={'': ['*.json', 'config/*.json', 'mechanism/*.json', 'data/*.json'],
                    'mind2act_eval': ['SOURCE_COMMIT']},
      cmdclass={'build_py': BuildPy, 'sdist': SourceDist})
