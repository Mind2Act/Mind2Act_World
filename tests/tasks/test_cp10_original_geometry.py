"""Retained pool child bytes require code-owned authority at the public gate."""
import json
import os
from pathlib import Path
import shutil

import pytest

from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.CP10.scene import validate_binding
from mind2act_eval.tasks.CP10.original_inputs import audit_source, read_audit


@pytest.fixture(scope='module')
def copied_root(tmp_path_factory):
    value = os.environ.get('CP10_ORIGINAL_COMPILED_ROOT')
    if not value:
        pytest.skip('private original CP10 delivery not configured')
    root = tmp_path_factory.mktemp('cp10-pool-authority')/'delivery'
    shutil.copytree(Path(value), root, copy_function=os.link)
    return root


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_actual_pool_dependency_rejects_with_unchanged_parent_and_valid_original(copied_root, tier):
    from pxr import Usd, Sdf, Gf
    root = copied_root
    path = root/'test-cases/CP10'/tier/('CP10_'+tier+'_clean_000')/'spec.json'
    spec = InstanceSpec.from_dict(json.loads(path.read_text()))
    assets = LocalAssets(root, spec.assets, spec=spec)
    declaration = audit_source(spec, assets)
    proof = json.loads(assets.resolve('source_geometry_proof').read_text())
    base = root/proof['common_station']['path']
    parent = Sdf.Layer.OpenAsAnonymous(str(base))
    refs = sorted(p for p in parent.GetExternalReferences() if Path(p).suffix in ('.usd', '.usda', '.usdc'))
    child = (base.parent/refs[-1]).resolve()
    before = child.read_bytes()
    layer = Sdf.Layer.OpenAsAnonymous(str(child))
    points = []
    layer.Traverse('/', lambda p: points.append(layer.GetAttributeAtPath(p))
        if p.IsPropertyPath() and p.name == 'points' else None)
    assert points and points[0].default
    value = list(points[0].default)
    value[0] = Gf.Vec3f(value[0][0], value[0][1], value[0][2]+1)
    points[0].default = value
    child.unlink()
    try:
        layer.Export(str(child))
        with pytest.raises(ValueError, match='transitive asset authority'):
            validate_binding(spec, assets)
        with pytest.raises(ValueError, match='transitive asset authority'):
            read_audit(spec, declaration)
    finally:
        child.write_bytes(before)


def test_warm_public_gate_rechecks_current_symlink_edge(copied_root, tmp_path):
    from pxr import Usd, Sdf, Gf
    root = tmp_path/'alias-delivery'
    shutil.copytree(copied_root, root, copy_function=os.link)
    path = root/'test-cases/CP10/medium/CP10_medium_clean_000/spec.json'
    spec = InstanceSpec.from_dict(json.loads(path.read_text()))
    assets = LocalAssets(root, spec.assets, spec=spec)
    proof = json.loads(assets.resolve('source_geometry_proof').read_text())
    base = root/proof['common_station']['path']
    parent = Sdf.Layer.OpenAsAnonymous(str(base))
    refs = sorted(p for p in parent.GetExternalReferences() if Path(p).suffix in ('.usd', '.usda', '.usdc'))
    child = base.parent/refs[-1]
    donor = child.with_name('unchanged-donor.usda')
    child.rename(donor)
    child.symlink_to(donor.name)
    # The first accepted read must actually contain the alias, not a cache entry
    # made while the logical edge still pointed directly to its own file.
    validate_binding(spec, assets)
    layer = Sdf.Layer.OpenAsAnonymous(str(donor))
    points = []
    layer.Traverse('/', lambda p: points.append(layer.GetAttributeAtPath(p))
        if p.IsPropertyPath() and p.name == 'points' else None)
    values = list(points[0].default)
    values[0] = Gf.Vec3f(values[0][0], values[0][1], values[0][2]+1)
    points[0].default = values
    malicious = child.with_name('changed-mesh.usda')
    layer.Export(str(malicious))
    child.unlink()
    child.symlink_to(malicious.name)
    assert donor.read_bytes() != child.read_bytes()
    with pytest.raises(ValueError, match='(edge resolution|transitive asset authority)'):
        validate_binding(spec, assets)
