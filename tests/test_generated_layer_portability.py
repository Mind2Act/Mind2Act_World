"""Relocation must survive moving a complete asset bundle between machines."""
import json
import os
import shutil

import pytest
from pxr import Sdf, Usd

from mind2act_eval.sim.asset_relocation import prepare_station
from test_asset_relocation import fixture, sha, write


def test_default_generated_paths_survive_bundle_move(tmp_path):
    bundle = tmp_path / 'l20'
    bundle.mkdir()
    source, assets, manifest = fixture(bundle)
    image = write(assets.root / 'image.png', 'exact texture bytes')
    manifest['mappings'].append(dict(source_layer_sha256=sha(source),
        authored_path='/home/unknown/Image_0.png', target_asset_id='image',
        target_sha256=sha(image)))
    assets.entries['image'] = 'image.png'
    (assets.root / 'relocation.json').write_text(json.dumps(manifest))
    runtime = prepare_station(source, assets, 'relocation', bundle / 'generated')
    stage = Usd.Stage.Open(str(runtime))
    assert not os.path.isabs(stage.GetPrimAtPath('/World').GetAttribute('image').Get().path)
    assert runtime.stat().st_size < 1024
    target = tmp_path / 'machine'
    shutil.copytree(bundle, target)
    bundle.rename(tmp_path / 'offline-l20')
    moved = target / runtime.relative_to(bundle)
    stage = Usd.Stage.Open(str(moved))
    prim = stage.GetPrimAtPath('/World')
    assert prim.GetAttribute('size').Get() == 3
    resolved = prim.GetAttribute('image').Get().resolvedPath
    assert resolved == str(target / 'assets/image.png')
    assert (target / 'assets/image.png').read_bytes() == b'exact texture bytes'


@pytest.mark.parametrize('token', ['/home/operator/a.png', '/shared/a.usd',
    '/root/a.mdl', 'file:/home/a.png', 'C:/cache/a.png',
    'https://nucleus.example/a.png', 'omniverse://host/a.mdl'])
def test_cpu_scan_rejects_nonportable_asset_tokens(tmp_path, token):
    from mind2act_eval.sim.asset_relocation import audit_generated_layers
    layer = write(tmp_path / 'generated.usda',
        '#usda 1.0\ndef Xform "World" {\n custom asset image = @' + token + '@\n}\n')
    report = audit_generated_layers([layer])
    assert not report['passed']
    assert report['issues'][0]['asset_path'] == token


def test_cpu_scan_covers_variants_arrays_time_samples_and_composition(tmp_path):
    from mind2act_eval.sim.asset_relocation import audit_generated_layers
    layer = Sdf.Layer.CreateNew(str(tmp_path / 'generated.usda'))
    prim = Sdf.CreatePrimInLayer(layer, '/World')
    prim.referenceList.prependedItems = [Sdf.Reference('/root/reference.usda')]
    prim.payloadList.prependedItems = [Sdf.Payload('/shared/payload.usda')]
    layer.subLayerPaths = ['/home/sublayer.usda']
    image = Sdf.AttributeSpec(prim, 'image', Sdf.ValueTypeNames.Asset)
    layer.SetTimeSample(image.path, 1, Sdf.AssetPath('/root/animated.png'))
    images = Sdf.AttributeSpec(prim, 'images', Sdf.ValueTypeNames.AssetArray)
    images.default = [Sdf.AssetPath('/root/array.png')]
    variants = Sdf.VariantSetSpec(prim, 'look')
    variant = Sdf.VariantSpec(variants, 'hidden')
    attr = Sdf.AttributeSpec(variant.primSpec, 'image', Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath('/root/variant.png')
    layer.Save()
    report = audit_generated_layers([tmp_path])
    assert {r['asset_path'] for r in report['issues']} == {
        '/root/reference.usda', '/shared/payload.usda', '/home/sublayer.usda',
        '/root/animated.png', '/root/array.png', '/root/variant.png'}


def test_cpu_scan_accepts_relative_and_symbolic_mdl_without_modifying_source(tmp_path):
    from mind2act_eval.sim.asset_relocation import audit_generated_layers
    layer = write(tmp_path / 'generated.usda', '''#usda 1.0
def Xform "World" (prepend references = @../assets/mesh.usdz@) {
 custom asset image = @../assets/image.png@
 custom asset shader = @gltf/pbr.mdl@
 custom string diagnostic = "/home/source-is-provenance"
}
''')
    before = layer.read_bytes()
    assert audit_generated_layers([tmp_path])['passed']
    assert layer.read_bytes() == before


def test_delivery_grep_includes_unused_layers_and_manifest_strings(tmp_path):
    from mind2act_eval.sim.asset_relocation import audit_delivery_tree, main
    write(tmp_path / 'current.usda', '#usda 1.0\ndef Xform "World" {}\n')
    assert audit_delivery_tree(tmp_path)['passed']
    write(tmp_path / 'unused/source.json', '{"source": "/home/ml-user/old/input"}')
    report = audit_delivery_tree(tmp_path)
    assert not report['passed']
    assert report['matching_files'] == [str(tmp_path / 'unused/source.json')]
    assert main(['--delivery-tree', str(tmp_path)]) == 1


def test_explicit_absolute_mode_cannot_generate_a_layer(tmp_path):
    source, assets, manifest = fixture(tmp_path)
    manifest['asset_path_mode'] = 'absolute'
    (assets.root / 'relocation.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='require relative'):
        prepare_station(source, assets, 'relocation', tmp_path / 'generated')
    assert not (tmp_path / 'generated').exists()
