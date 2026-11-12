"""CLI opt-in forwarding; the allocation owner tests actual SDK/Slurm identity."""
import argparse
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from mind2act_eval.cli import _gpu_selection_kwargs, _job, _job_defaults
from mind2act_eval.contracts import InstanceSpec, to_json


def test_no_mapping_preserves_original_keyword_signature():
    assert _gpu_selection_kwargs(argparse.Namespace()) == {}
    assert _gpu_selection_kwargs(argparse.Namespace(gpu_allocation=None, gpu_allocation_sha256=None)) == {}


@pytest.mark.parametrize("path,digest", [("mapping.json", None), (None, "a"*64), ("", "a"*64), ("mapping.json", "")])
def test_mapping_options_are_required_as_a_nonempty_pair(path, digest):
    with pytest.raises(ValueError, match="supplied together"):
        _gpu_selection_kwargs(argparse.Namespace(gpu_allocation=path, gpu_allocation_sha256=digest))


def test_explicit_mapping_goes_through_reviewed_loader_and_keeps_physical_index(tmp_path, monkeypatch):
    module = ModuleType("mind2act_eval.gpu_allocation")
    calls, selection = [], object()
    def load(path, expected, *, physical_gpu):
        calls.append((path, expected, physical_gpu))
        return selection
    module.load_gpu_selection = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = argparse.Namespace(backend="isaac", gpu=0, gpu_allocation=str(tmp_path/"mapping.json"), gpu_allocation_sha256="a"*64)
    assert _gpu_selection_kwargs(args) == {"gpu_selection": selection}
    assert calls == [(tmp_path/"mapping.json", "a"*64, 0)]
    spec = InstanceSpec("demo", "source", "easy", "original-source", "v1", "none", "v1")
    path = tmp_path/"spec.json"
    path.write_text(json.dumps(to_json(spec)))
    values = _job_defaults()
    values.update(spec=str(path), asset_root=str(tmp_path), backend="isaac", adapter="reviewed:adapter",
                  gpu=0, wall_s=10, physics_ticks=4, gpu_allocation=args.gpu_allocation,
                  gpu_allocation_sha256=args.gpu_allocation_sha256)
    job = _job(argparse.Namespace(**values))
    index = job.argv.index("--gpu-allocation")
    assert job.argv[index+1] == str((tmp_path/"mapping.json").resolve())
    assert job.argv[job.argv.index("--gpu-allocation-sha256")+1] == "a"*64
    assert job.spec == spec


def test_cpu_backend_cannot_accept_gpu_mapping():
    with pytest.raises(ValueError, match="Isaac backend"):
        _gpu_selection_kwargs(argparse.Namespace(backend="demo", gpu_allocation="x", gpu_allocation_sha256="a"*64))
