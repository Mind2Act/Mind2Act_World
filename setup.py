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
            raise ValueError('SOURCE_COMMIT differs from Git HEAD')
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


PACKAGE_ROOTS = {
    'mind2act_eval': 'eval',
    'mind2act_eval.tasks': 'task',
    'mind2act_eval.robot': 'env/robot',
    'mind2act_eval.sim': 'env/sim',
    'mind2act_eval.variations': 'env/variations',
    'mind2act_eval.adapters': 'policies',
    'mind2act_eval.env_cfg': 'env_cfg',
}
packages = []
for package, directory in PACKAGE_ROOTS.items():
    packages.append(package)
    packages.extend(package + '.' + child for child in find_packages(directory))

setup(packages=packages, package_dir=PACKAGE_ROOTS,
      cmdclass={'build_py': BuildPy, 'sdist': SourceDist})
