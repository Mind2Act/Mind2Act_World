"""Locate this checkout or installed distribution without assuming src layout."""
from pathlib import Path


def code_root():
    package = Path(__file__).resolve().parent
    if package.name == 'eval' and (package.parent / 'pyproject.toml').is_file():
        return package.parent
    return package
