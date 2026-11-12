"""Real CPU USD export and file-backed overlay portability."""
from pathlib import Path
import shutil

import pytest
from pxr import Ar, Sdf, Usd

from test_asset_relocation import write
from usdz_support import package_writer


def test_file_backed_writer_resolves_after_complete_bundle_move(tmp_path):
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    bundle = tmp_path / 'l20'
    texture = write(bundle / 'assets/image.png', 'exact pixels')
    output = bundle / 'episode/variation.usda'
    output.parent.mkdir(parents=True)
    layer = Sdf.Layer.CreateNew(str(output))
    prim = Sdf.CreatePrimInLayer(layer, '/World')
    attr = Sdf.AttributeSpec(prim, 'image', Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath(relative_asset_path(layer, texture))
    layer.Save()
    assert attr.default.path == '../assets/image.png'
    target = tmp_path / 'pro'
    shutil.copytree(bundle, target)
    bundle.rename(tmp_path / 'offline-l20')
    stage = Usd.Stage.Open(str(target / 'episode/variation.usda'))
    assert stage.GetPrimAtPath('/World').GetAttribute('image').Get().resolvedPath == str(target / 'assets/image.png')


def test_writer_rejects_unanchored_layer_and_missing_target(tmp_path):
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    target = write(tmp_path / 'image.png', 'pixels')
    with pytest.raises(ValueError, match='file-backed'):
        relative_asset_path(Sdf.Layer.CreateAnonymous(), target)
    layer = Sdf.Layer.CreateNew(str(tmp_path / 'layer.usda'))
    with pytest.raises(FileNotFoundError):
        relative_asset_path(layer, tmp_path / 'missing.png')


def test_local_file_uri_is_decoded_but_remote_uri_is_rejected(tmp_path):
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    target = write(tmp_path / 'image with spaces.png', 'pixels')
    layer = Sdf.Layer.CreateNew(str(tmp_path / 'layer.usda'))
    assert relative_asset_path(layer, target.as_uri()) == 'image with spaces.png'
    with pytest.raises(ValueError, match='local file'):
        relative_asset_path(layer, 'file://another-host/image.png')


def test_writer_preserves_package_member_and_checks_it_exists(tmp_path):
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    source = write(tmp_path / 'root.usda', '#usda 1.0\ndef Xform "World" {}\n')
    package = tmp_path / 'payload.usdz'
    with package_writer(package) as archive:
        archive.AddFile(str(source), 'root.usda')
    output = tmp_path / 'episode/variation.usda'
    output.parent.mkdir()
    layer = Sdf.Layer.CreateNew(str(output))
    token = relative_asset_path(layer, str(package) + '[root.usda]')
    assert token == '../payload.usdz[root.usda]'
    assert Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(layer, token))
    with pytest.raises(FileNotFoundError):
        relative_asset_path(layer, str(package) + '[missing.usda]')


def test_runtime_export_reanchors_absolute_and_source_relative_assets(tmp_path):
    from mind2act_eval.sim.asset_relocation import export_relative_layer, audit_generated_layers
    texture = write(tmp_path / 'assets/image.png', 'pixels')
    child = write(tmp_path / 'assets/child.usda', '#usda 1.0\ndef Xform "Child" {}\n')
    write(tmp_path / 'assets/local.mdl', 'mdl 1.3;')
    original = write(tmp_path / 'source.usda', '''#usda 1.0
def Xform "World" (prepend references = @assets/child.usda@</Child>) {
 double3 xformOp:translate = (1, 2, 3)
 custom asset image = @assets/image.png@
 custom asset shader = @gltf/pbr.mdl@
 custom asset local_shader = @assets/local.mdl@
}
''')
    layer = Sdf.Layer.FindOrOpen(str(original))
    layer.GetAttributeAtPath('/World.image').default = Sdf.AssetPath(str(texture))
    before_file, before_memory = original.read_bytes(), layer.ExportToString()
    output = export_relative_layer(layer, tmp_path / 'episode/scene.usda')
    assert original.read_bytes() == before_file and layer.ExportToString() == before_memory
    assert audit_generated_layers([output])['passed']
    runtime = Sdf.Layer.FindOrOpen(str(output))
    assert runtime.GetAttributeAtPath('/World.image').default.path == '../assets/image.png'
    assert runtime.GetPrimAtPath('/World').referenceList.prependedItems[0].assetPath == '../assets/child.usda'
    assert tuple(runtime.GetAttributeAtPath('/World.xformOp:translate').default) == (1, 2, 3)
    assert runtime.GetAttributeAtPath('/World.shader').default.path == 'gltf/pbr.mdl'
    assert runtime.GetAttributeAtPath('/World.local_shader').default.path == '../assets/local.mdl'
    with pytest.raises((FileExistsError, ValueError)):
        export_relative_layer(layer, original)
