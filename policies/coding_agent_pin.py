"""Compile the pinned upstream CodingAgent sources; never import an old runner."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

COMMIT = "9ffacc54f372beef6479d61711bf4945e31ae4ce"
LOCK_SHA256 = "5802d8054c1e5ff590ebe4994896a9e418bcb824b4a300223e662b98b91037a0"


class _Source(importlib.abc.Loader):
    def __init__(self, path, digest):
        self.path, self.digest = path, digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise RuntimeError("CodingAgent source digest mismatch: " + str(self.path))
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
        raise ImportError("Unpinned CodingAgent module: " + fullname)


class _Namespace(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path / "__init__.py")
        module.__mind2act_source_pin__ = COMMIT


def load(source_root):
    root = Path(source_root).resolve(strict=True)
    raw = (root / "source-lock.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCK_SHA256:
        raise RuntimeError("CodingAgent source lock differs from the approved pin")
    lock = json.loads(raw)
    if lock["commit"] != COMMIT:
        raise RuntimeError("CodingAgent commit mismatch")
    files = {name: digest for name, digest in lock["files"].items()
             if name.startswith("XPolicyLab/") and name.endswith(".py")}
    for name, digest in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("CodingAgent source digest mismatch: " + name)
    for name, module in list(sys.modules.items()):
        if name == "XPolicyLab" or name.startswith("XPolicyLab."):
            if getattr(module, "__mind2act_source_pin__", None) != COMMIT:
                raise RuntimeError("A different or unverified XPolicyLab package is already imported")
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError("CodingAgent package is already loaded from another root")
    if not any(isinstance(finder, _PinnedFinder) and finder.root == root for finder in sys.meta_path):
        sys.meta_path.insert(0, _PinnedFinder(root, files))
    prefix = "XPolicyLab.policy."
    joint = importlib.import_module(prefix + "AgentRuntime.policy")
    types = importlib.import_module(prefix + "AgentRuntime.types")
    eef = importlib.import_module(prefix + "AgentRuntime_EEF.policy")
    return joint, types, eef
