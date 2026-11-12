import pytest
from pxr import Sdf, Usd, UsdGeom
from mind2act_eval.robot.export_history import without_export_history
from mind2act_eval.sim.asset_relocation import audit_delivery_tree


def test_removes_export_history_and_preserves_composed_geometry_and_physics(tmp_path):
    child = Usd.Stage.CreateNew(str(tmp_path / 'mesh.usda'))
    cube = UsdGeom.Cube.Define(child, '/root')
    cube.GetSizeAttr().Set(0.37)
    cube.GetPrim().CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float).Set(2.5)
    child.SetDefaultPrim(cube.GetPrim())
    child.GetRootLayer().Save()
    source = Sdf.Layer.CreateNew(str(tmp_path / 'source.usda'))
    source.documentation = 'Generated from Composed Stage of root layer /home/kaslensu/Desktop/middle_camera.usd'
    prim = Sdf.CreatePrimInLayer(source, '/root')
    prim.referenceList.prependedItems = [Sdf.Reference('mesh.usda')]
    source.defaultPrim = 'root'
    source.Save()
    before = (tmp_path / 'source.usda').read_bytes()
    target = tmp_path / 'derived.usda'
    receipt = without_export_history(tmp_path / 'source.usda', target)
    assert receipt['removed_machine_path_count'] == 1
    assert (tmp_path / 'source.usda').read_bytes() == before
    assert not Sdf.Layer.FindOrOpen(str(target)).documentation
    stage = Usd.Stage.Open(str(target))
    assert stage.GetPrimAtPath('/root').GetAttribute('size').Get() == 0.37
    assert stage.GetPrimAtPath('/root').GetAttribute('physics:mass').Get() == 2.5


def test_refuses_to_hide_a_live_machine_asset_reference(tmp_path):
    source = Sdf.Layer.CreateNew(str(tmp_path / 'source.usda'))
    source.documentation = 'Collected at /home/person/model.usd'
    prim = Sdf.CreatePrimInLayer(source, '/root')
    Sdf.AttributeSpec(prim, 'image', Sdf.ValueTypeNames.Asset).default = Sdf.AssetPath('/home/person/texture.png')
    source.Save()
    target = tmp_path / 'derived.usda'
    with pytest.raises(ValueError, match='outside layer documentation'):
        without_export_history(tmp_path / 'source.usda', target)
    assert not target.exists()


def test_delivery_scan_catches_plain_collector_documentation_but_accepts_prim_paths(tmp_path):
    layer = Sdf.Layer.CreateNew(str(tmp_path / 'robot.usda'))
    Sdf.CreatePrimInLayer(layer, '/root/mesh')
    layer.Save()
    assert audit_delivery_tree(tmp_path)['passed']
    layer.documentation = 'Generated from /home/kaslensu/workspace/Luminis/assets/robots/x5/ARX.usd'
    layer.Save()
    assert not audit_delivery_tree(tmp_path)['passed']


@pytest.mark.parametrize('text',['/home/operator/file.usd','/shared/users/operator/file.usd',
                                '/Users/operator/file.usd','@/root/file.usd@'])
def test_delivery_scan_uses_a_posix_compatible_guard_for_every_private_root(tmp_path,text):
    (tmp_path/'metadata.txt').write_text(text)
    report=audit_delivery_tree(tmp_path)
    assert not report['passed'] and report['returncode']==0
    assert report['matching_files']==[str(tmp_path/'metadata.txt')]


@pytest.mark.parametrize('root',['/home','/root','/shared','/Users'])
@pytest.mark.parametrize('wrapper',['{}','\n{}\n','path="{}"',"path='{}'",'\t{} ',
                                    '[{}]','@{}@','file://{}','file:{}',
                                    'file://localhost{}','@file://{}@'])
def test_private_path_boundaries_in_plain_text_quotes_and_file_uris(tmp_path,root,wrapper):
    path=tmp_path/'metadata.txt';path.write_text(wrapper.format(root+'/private/file.usd'))
    report=audit_delivery_tree(tmp_path)
    assert not report['passed'] and report['returncode']==0,report
    assert report['matching_files']==[str(path)] and report['error'] is None


@pytest.mark.parametrize('text',['/root','/root/','"/root"',"'/root'",'/home',
                                '/shared','/Users','file:///root','file://localhost/root'])
def test_private_root_at_end_of_line_or_quoted_value(tmp_path,text):
    (tmp_path/'metadata.txt').write_text(text)
    report=audit_delivery_tree(tmp_path)
    assert not report['passed'] and report['returncode']==0,report


@pytest.mark.parametrize('text',['root/private/file.usd','relative/root/private/file.usd',
    './root/private/file.usd','../root/private/file.usd','/rooted/private/file.usd',
    '/root-private/file.usd','/root_private/file.usd','/homework/private/file.usd',
    '/sharedness/file.usd','models/home/file.usd','/World/root/mesh','</root/mesh>',
    'https://example.org/root/private/file.usd','https://example.org/home/profile',
    'file:///opt/root/private/file.usd','prefix/home/file.usd','folder/Users/file.usd'])
def test_relative_paths_similar_roots_prim_targets_and_public_urls_are_not_private_roots(tmp_path,text):
    (tmp_path/'metadata.txt').write_text(text)
    report=audit_delivery_tree(tmp_path)
    assert report['passed'] and report['returncode']==1,report
    assert not report['matching_files'] and report['error'] is None


@pytest.mark.parametrize('pattern',['[','(?:never_matches)'])
def test_actual_grep_error_or_warning_cannot_pass(tmp_path,monkeypatch,pattern):
    import platform
    if pattern == '(?:never_matches)' and platform.system() != 'Linux':
        pytest.skip('GNU grep warning behavior; BSD grep accepts this extension')
    from mind2act_eval.sim import asset_relocation
    (tmp_path/'metadata.txt').write_text('ordinary metadata')
    monkeypatch.setattr(asset_relocation,'DELIVERY_FORBIDDEN_PATTERN',pattern)
    report=audit_delivery_tree(tmp_path)
    assert report['error'] and not report['passed'],report


def test_inconsistent_grep_output_cannot_pass(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(
        a[0],1,stdout='unexpected matching file\n',stderr=''))
    assert not audit_delivery_tree(tmp_path)['passed']
