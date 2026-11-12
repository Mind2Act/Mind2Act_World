"""Locate installed package resources independently of checkout layout."""
from importlib.util import find_spec
from pathlib import Path

# Import namespaces stay stable across editable checkouts and built wheels.
PACKAGES = ('mind2act_eval', 'mind2act_eval.tasks', 'mind2act_eval.sim',
            'mind2act_eval.robot', 'mind2act_eval.adapters')


def source_files():
    """Return Python sources keyed by their canonical package-relative paths."""
    result = {}
    for package in PACKAGES:
        root = Path(find_spec(package).origin).parent
        prefix = package.removeprefix('mind2act_eval').lstrip('.').replace('.', '/')
        for path in root.rglob('*.py'):
            relative = Path(prefix) / path.relative_to(root)
            result[relative] = path
    return result
