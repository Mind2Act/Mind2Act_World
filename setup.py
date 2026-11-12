"""Preserve source identity in wheels and source distributions."""
from pathlib import Path
import re
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def source_commit():
    source = ROOT / 'SOURCE_COMMIT'
    if source.is_file():
        value = source.read_text().strip()
    else:
        value = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if re.fullmatch('[0-9a-f]{40}', value) is None:
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


setup(cmdclass={'build_py': BuildPy, 'sdist': SourceDist})
