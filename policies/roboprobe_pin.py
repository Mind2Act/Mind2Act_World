"""Compile the pinned upstream RoboProbe sources; never import an old runner."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

COMMIT = "9ffacc54f372beef6479d61711bf4945e31ae4ce"
LOCK_SHA256 = "a70f8c497901a6aee695657a54988a547b5940ec30b6bc0631edc89b952e9e49"


class _Source(importlib.abc.Loader):
    def __init__(self, path, digest):
        self.path, self.digest = path, digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise RuntimeError("RoboProbe source digest mismatch: " + str(self.path))
        module.__file__ = str(self.path)
        module.__mind2act_source_pin__ = COMMIT
        exec(compile(raw, str(self.path), "exec"), module.__dict__)


class _PinnedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, files):
        self.root, self.files = root, files

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "XPolicyLab" and not fullname.startswith("XPolicyLab."):
            return None
        relative = fullname.replace(".", "/")
        for name, package in ((relative + "/__init__.py", True), (relative + ".py", False)):
            if name in self.files:
                return importlib.util.spec_from_loader(fullname, _Source(self.root / name, self.files[name]), is_package=package)
        # Namespace packages such as utils contain only pinned child modules.
        if any(name.startswith(relative + "/") for name in self.files):
            spec = importlib.util.spec_from_loader(fullname, loader=_Namespace(self.root / relative), is_package=True)
            spec.submodule_search_locations = [str(self.root / relative)]
            return spec
        raise ImportError("Unpinned RoboProbe module: " + fullname)


class _Namespace(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path / "__init__.py")
        module.__mind2act_source_pin__ = COMMIT


def resolve_source_root(value=None):
    if value is not None:
        return Path(value).resolve(strict=True)
    current = os.environ.get('MIND2ACT_AGENT_SOURCE_ROOT')
    legacy = os.environ.get('MIND2ACT_CODING_AGENT_SOURCE_ROOT')
    if current and legacy and Path(current).resolve() != Path(legacy).resolve():
        raise ValueError('conflicting agent runtime source roots')
    if not (current or legacy):
        raise ValueError('source_root or MIND2ACT_AGENT_SOURCE_ROOT is required')
    return Path(current or legacy).resolve(strict=True)


def load(source_root=None):
    root = resolve_source_root(source_root)
    raw = (root / "source-lock.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCK_SHA256:
        raise RuntimeError("RoboProbe source lock differs from the approved pin")
    lock = json.loads(raw)
    if lock["commit"] != COMMIT:
        raise RuntimeError("RoboProbe commit mismatch")
    files = {name: digest for name, digest in lock["files"].items()
             if name.startswith("XPolicyLab/") and name.endswith(".py")}
    for name, digest in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("RoboProbe source digest mismatch: " + name)
    for name, module in list(sys.modules.items()):
        if name == "XPolicyLab" or name.startswith("XPolicyLab."):
            if getattr(module, "__mind2act_source_pin__", None) != COMMIT:
                raise RuntimeError("A different or unverified XPolicyLab package is already imported")
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError("RoboProbe package is already loaded from another root")
    if not any(isinstance(finder, _PinnedFinder) and finder.root == root for finder in sys.meta_path):
        sys.meta_path.insert(0, _PinnedFinder(root, files))
    prefix = "XPolicyLab.policy."
    joint = importlib.import_module(prefix + "RoboDojo_Agent_L3_Inspect.policy")
    types = importlib.import_module(prefix + "RoboDojo_Agent_L3_Inspect.types")
    eef = importlib.import_module(prefix + "RoboDojo_Agent_L3_Inspect_EEF.policy")
    return joint, types, eef
