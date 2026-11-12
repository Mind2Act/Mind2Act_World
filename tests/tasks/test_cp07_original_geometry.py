"""Source authority is independent of caller-refreshed scene/binding hashes."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.CP07.binding import load


@pytest.fixture(scope='module')
def copied_root(tmp_path_factory):
    value = os.environ.get('CP07_ORIGINAL_COMPILED_ROOT')
    if not value:
        pytest.skip('private original CP07 delivery not configured')
    root = tmp_path_factory.mktemp('cp07-source-geometry')/'delivery'
    shutil.copytree(Path(value), root, copy_function=os.link)
    return root


@pytest.fixture(params=['easy', 'medium', 'hard'])
def original(request, copied_root):
    root = copied_root
    path = root/'test-cases'/'CP07'/request.param/('CP07_'+request.param+'_clean_000')/'spec.json'
    spec = InstanceSpec.from_dict(json.loads(path.read_text()))
    return spec, LocalAssets(root, spec.assets, spec=spec)


@pytest.mark.parametrize('mutation', ['units', 'gravity', 'shape', 'task_shape'])
def test_caller_rehash_cannot_authorize_altered_actual_station(original, tmp_path, mutation):
    from pxr import Usd, Sdf
    spec, assets = original
    # Keep the real layer's relative dependencies valid while editing a copy.
    text = assets.resolve('station').read_text()
    layer = Sdf.Layer.CreateAnonymous('.usda')
    assert layer.ImportFromString(text)
    layer.subLayerPaths = [str((assets.resolve('station').parent/p).resolve()) for p in layer.subLayerPaths]
    if mutation == 'units':
        layer.pseudoRoot.SetInfo('metersPerUnit', .01)
    elif mutation in ('gravity', 'shape'):
        prim = Sdf.CreatePrimInLayer(layer, '/World/PhysicsScene' if mutation == 'gravity' else '/World/Pieces/SupportThick1')
        name, kind, value = ('physics:gravityMagnitude', Sdf.ValueTypeNames.Float, 0.) if mutation == 'gravity' else ('xformOp:scale', Sdf.ValueTypeNames.Double3, (10., 10., 10.))
        Sdf.AttributeSpec(prim, name, kind).default = value
    station = assets.resolve('station').with_name(tmp_path.name+'.usda')
    proof = json.loads(assets.resolve('source_geometry_proof').read_text())
    roles = dict(spec.assets)
    if mutation == 'task_shape':
        task = Sdf.Layer.OpenAsAnonymous(str(assets.resolve('task_source')))
        prim = task.GetPrimAtPath('/World/Pieces/SupportThick1')
        prim.attributes['physics:mass'].default = 10.
        task_path = assets.resolve('task_source').with_name(tmp_path.name+'-task.usda')
        task.Export(str(task_path))
        layer.subLayerPaths = [str(task_path), layer.subLayerPaths[1]]
        proof['actual_composed_task'] = dict(path=str(task_path.relative_to(assets.root)),
            sha256=hashlib.sha256(task_path.read_bytes()).hexdigest())
        roles['task_source'] = str(task_path.relative_to(assets.root))
    layer.Export(str(station))
    binding = json.loads(assets.resolve('cp07_binding').read_text())
    binding['scene_sha256'] = hashlib.sha256(station.read_bytes()).hexdigest()
    raw = json.dumps(binding).encode()
    binding_path = station.with_suffix('.binding.json')
    binding_path.write_bytes(raw)
    config = to_json(spec.config)
    config['cp07_binding_sha256'] = hashlib.sha256(raw).hexdigest()
    proof['compiled_scene'] = dict(path=str(station.relative_to(assets.root)),
        sha256=binding['scene_sha256'])
    proof_path = station.with_suffix('.proof.json')
    proof_path.write_text(json.dumps(proof))
    config['source_geometry_proof_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    roles.update(station=str(station.relative_to(assets.root)),
        cp07_binding=str(binding_path.relative_to(assets.root)),
        source_geometry_proof=str(proof_path.relative_to(assets.root)))
    changed = replace(spec, config=config, assets=roles)
    mapped = LocalAssets(assets.root, roles, spec=changed)
    with pytest.raises(ValueError):
        load(changed, mapped)


def test_unmodified_original150_actual_geometry_passes(copied_root):
    paths = sorted(copied_root.glob('test-cases/CP07/*/*/spec.json'))
    assert len(paths) == 150
    for path in paths:
        spec = InstanceSpec.from_dict(json.loads(path.read_text()))
        binding = load(spec, LocalAssets(copied_root, spec.assets, spec=spec))
        assert len(binding['pieces']) == {'easy': 8, 'medium': 11, 'hard': 14}[spec.difficulty]


def test_retained_control_pool_child_is_not_authorized_by_unchanged_parent(original):
    from pxr import Usd, Sdf, Gf
    spec, assets = original
    proof = json.loads(assets.resolve('source_geometry_proof').read_text())
    base = assets.root/proof['common_station']['path']
    parent = Sdf.Layer.OpenAsAnonymous(str(base))
    refs = sorted(p for p in parent.GetExternalReferences() if Path(p).suffix in ('.usd', '.usda', '.usdc'))
    path = (base.parent/refs[-1]).resolve()
    before = path.read_bytes()
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    points = []
    layer.Traverse('/', lambda p: points.append(layer.GetAttributeAtPath(p))
        if p.IsPropertyPath() and p.name == 'points' else None)
    assert points and points[0].default
    value = list(points[0].default)
    value[0] = Gf.Vec3f(value[0][0], value[0][1], value[0][2]+1)
    points[0].default = value
    path.unlink()  # Break the test-only hard link before changing its bytes.
    try:
        layer.Export(str(path))
        with pytest.raises(ValueError, match='transitive asset authority'):
            load(spec, assets)
    finally:
        path.write_bytes(before)


def test_warm_public_gate_rechecks_current_symlink_edge(copied_root, tmp_path):
    from pxr import Usd, Sdf, Gf
    root = tmp_path/'alias-delivery'
    shutil.copytree(copied_root, root, copy_function=os.link)
    path = root/'test-cases/CP07/medium/CP07_medium_clean_000/spec.json'
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
    load(spec, assets)
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
        load(spec, assets)
