import hashlib
import json
from pathlib import Path

from pxr import Sdf, Usd
from usdz_support import package_writer

from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim.asset_relocation import prepare_station
from mind2act_eval.sim.material_closure import audit_scene, audit_inventory, configure_mdl_search, verify_native_mdl_search


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hidden_texture_and_missing_sdk_mdl_fail_cpu_audit(tmp_path):
    write(tmp_path/'scene.usda', '''#usda 1.0
def Scope "World" {
 def Scope "Hidden" (active = false) {
  custom asset decal = @labels/absent.png@
 }
 custom asset mdl = @OmniPBR.mdl@
}
''')
    result = audit_scene(LocalAssets(tmp_path, {'scene': 'scene.usda'}), ['scene'])
    assert not result['passed']
    assert {r['authored_path'] for r in result['issues']} == {'labels/absent.png', 'OmniPBR.mdl'}


def test_exact_source_overlay_resolves_without_rewriting_original(tmp_path):
    source = write(tmp_path/'scene.usda', '#usda 1.0\ndef Scope "World" {\n custom asset decal = @/root/source/label.png@\n}\n')
    texture = write(tmp_path/'overlay/label.png', 'original bytes')
    manifest = {'schema': 'mind2act_eval.station_relocation.v1', 'source_sha256': sha(source),
                'asset_path_mode': 'relative', 'mappings': [dict(source_layer_sha256=sha(source),
                authored_path='/root/source/label.png', target_asset_id='label', target_sha256=sha(texture))]}
    write(tmp_path/'overlay.json', json.dumps(manifest))
    assets = LocalAssets(tmp_path, dict(scene='scene.usda', label='overlay/label.png', relocation='overlay.json'))
    assert not audit_scene(assets, ['scene'])['passed']
    assert audit_scene(assets, ['scene'], relocation_asset_id='relocation')['passed']
    before = source.read_bytes()
    derived = prepare_station(source, assets, 'relocation', tmp_path/'run')
    stage = Usd.Stage.Open(str(derived))
    image = stage.GetPrimAtPath('/World').GetAttribute('decal').Get()
    raw = image.path
    assert not Path(raw).is_absolute()
    assert Path(image.resolvedPath) == texture
    assert source.read_bytes() == before
    # Relocation output and asset payload can move together to another host root.
    import shutil
    moved = tmp_path.parent/(tmp_path.name+'-moved')
    shutil.copytree(tmp_path, moved)
    stage = Usd.Stage.Open(str(moved/derived.relative_to(tmp_path)))
    image = stage.GetPrimAtPath('/World').GetAttribute('decal').Get()
    assert Path(image.resolvedPath).read_bytes() == texture.read_bytes()


def test_usdz_internal_and_missing_external_material_are_both_scanned(tmp_path):
    member = write(tmp_path/'root.usda', '#usda 1.0\ndef Scope "World" {\n custom asset decal = @label.png@\n custom asset missing = @missing.png@\n}\n')
    image = write(tmp_path/'label.png', 'test resource')
    with package_writer(tmp_path/'scene.usdz') as writer:
        writer.AddFile(str(member), 'root.usda')
        writer.AddFile(str(image), 'label.png')
    result = audit_scene(LocalAssets(tmp_path, {'scene': 'scene.usdz'}), ['scene'])
    assert not result['passed']
    assert [r['authored_path'] for r in result['issues']] == ['missing.png']
    assert next(r for r in result['references'] if r['authored_path']=='label.png')['resolved']


def test_mdl_transitive_import_and_texture_fail_closed(tmp_path):
    write(tmp_path/'scene.usda', '#usda 1.0\ndef Scope "World" {\n custom asset mdl = @mdl/Surface.mdl@\n}\n')
    write(tmp_path/'mdl/Surface.mdl', 'mdl 1.7; import ::math::*; import ::custom::surface::*;')
    write(tmp_path/'mdl/custom/surface.mdl', 'mdl 1.7; export material surface() = material(); texture_2d("label.png");')
    assets = LocalAssets(tmp_path, dict(scene='scene.usda', modules='mdl'))
    result = audit_scene(assets, ['scene'], mdl_search_asset_ids=['modules'])
    assert not result['passed']
    assert result['issues'][0]['kind']=='unresolved_mdl_resource'
    write(tmp_path/'mdl/custom/label.png', 'original resource')
    assert audit_scene(assets, ['scene'], mdl_search_asset_ids=['modules'])['passed']
    (tmp_path/'mdl/custom/surface.mdl').unlink()
    assert not audit_scene(assets, ['scene'], mdl_search_asset_ids=['modules'])['passed']


def test_inventory_reports_missing_task_and_does_not_skip_bad_scene(tmp_path):
    write(tmp_path/'scene.usda', '#usda 1.0\ndef Scope "World" {}\n')
    report = audit_inventory(dict(required_tasks=['CP04','CP05'], scenes=[dict(task_id='CP04',
        difficulty='easy', assets={'scene':'scene.usda'}, scene_asset_ids=['scene'])]), tmp_path)
    assert not report['passed'] and report['uncovered_tasks']==['CP05']
    assert report['scenes'][0]['passed']


def test_native_search_paths_come_from_same_asset_resolver(tmp_path):
    import pytest
    (tmp_path/'mdl').mkdir()
    environment = {'MDL_USER_PATH':'/old/machine/cache'}
    receipt = configure_mdl_search(LocalAssets(tmp_path, {'modules':'mdl'}), ['modules'], environment)
    assert environment == {'MDL_SYSTEM_PATH':str((tmp_path/'mdl').resolve()),'MDL_USER_PATH':str((tmp_path/'mdl').resolve())}
    assert receipt['asset_ids']==['modules']
    with pytest.raises(ValueError):
        configure_mdl_search(LocalAssets(tmp_path, {'escape':'../'}), ['escape'], environment)
    assert verify_native_mdl_search(receipt,lambda:dict(system=[],user=[],active=receipt['resolved_paths']))['native_compiler_paths_verified']
    with pytest.raises(RuntimeError,match='native MDL compiler is missing'):
        verify_native_mdl_search(receipt,lambda:dict(system=receipt['resolved_paths'],user=[],active=[]))


def test_native_sdk_registers_paths_when_kit_ignores_standard_environment(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import sys
    from types import ModuleType, SimpleNamespace
    (tmp_path/'mdl').mkdir()
    receipt = configure_mdl_search(LocalAssets(tmp_path, {'modules':'mdl'}), ['modules'], {})
    registered = []
    cfg = SimpleNamespace(
        get_mdl_paths_length=lambda:len(registered),
        get_mdl_path=lambda i:nullcontext(SimpleNamespace(get_c_str=lambda:registered[i])),
        get_mdl_system_paths_length=lambda:1,
        get_mdl_system_path=lambda i:receipt['resolved_paths'][i],
        get_mdl_user_paths_length=lambda:0,
        get_mdl_user_path=lambda i:None,
        add_mdl_path=lambda p:(registered.append(p) or 0))
    omni, mdl, neuraylib = (ModuleType(n) for n in ('omni','omni.mdl','omni.mdl.neuraylib'))
    omni.mdl = mdl
    mdl.neuraylib = neuraylib
    neuraylib.get_neuraylib = lambda:SimpleNamespace(getNeurayAPI=lambda:123)
    mdl.pymdlsdk = SimpleNamespace(IMdl_configuration=object,
        attach_ineuray=lambda p:SimpleNamespace(get_api_component=lambda kind:nullcontext(cfg)))
    for module in (omni, mdl, neuraylib):
        monkeypatch.setitem(sys.modules,module.__name__,module)
    result = verify_native_mdl_search(receipt)
    assert registered == receipt['resolved_paths']
    assert result['native_compiler_paths']['active'] == receipt['resolved_paths']
    assert result['native_compiler_paths_verified']


def test_native_sdk_file_uri_search_paths_are_the_same_resolver_directory(tmp_path):
    (tmp_path/'mdl with space').mkdir()
    receipt = configure_mdl_search(LocalAssets(tmp_path, {'modules':'mdl with space'}), ['modules'], {})
    path = (tmp_path/'mdl with space').resolve()
    for value in [path.as_uri(), 'file:'+str(path)]:
        result = verify_native_mdl_search(receipt,lambda:dict(system=[],user=[],active=[value]))
        assert result['native_compiler_paths_verified']
    import pytest
    with pytest.raises(RuntimeError,match='missing resolver search paths'):
        verify_native_mdl_search(receipt,lambda:dict(active=['file://other-host'+str(path)]))


def test_missing_material_binding_and_shader_connection_fail(tmp_path):
    write(tmp_path/'scene.usda', '''#usda 1.0
def Xform "World" {
 rel material:binding = </World/MissingMaterial>
 def Material "Label" {
  token outputs:surface.connect = </World/AbsentShader.outputs:surface>
 }
}
''')
    report=audit_scene(LocalAssets(tmp_path, {'scene':'scene.usda'}), ['scene'])
    assert not report['passed']
    assert {r['kind'] for r in report['issues']} == {'unresolved_material_binding','unresolved_shader_connection'}


def test_usdz_mdl_imports_and_resources_use_package_resolver(tmp_path):
    member=write(tmp_path/'root.usda', '#usda 1.0\ndef Scope "World" {\n custom asset mdl = @materials/Surface.mdl@\n}\n')
    module=write(tmp_path/'Surface.mdl', 'mdl 1.7; import Child::*;')
    child=write(tmp_path/'Child.mdl', 'mdl 1.7; texture_2d("../textures/label.png");')
    image=write(tmp_path/'label.png', 'resource')
    with package_writer(tmp_path/'scene.usdz') as writer:
        for p,name in [(member,'root.usda'),(module,'materials/Surface.mdl'),(child,'materials/Child.mdl'),(image,'textures/label.png')]:
            writer.AddFile(str(p),name)
    result=audit_scene(LocalAssets(tmp_path, {'scene':'scene.usdz'}), ['scene'])
    assert result['passed'],result['issues']
    assert any(r.get('kind')=='mdl_resource' and r['resolved'].endswith('[textures/label.png]') for r in result['references'])


def test_declared_runtime_screen_asset_cannot_be_silently_missing(tmp_path):
    write(tmp_path/'scene.usda','#usda 1.0\ndef Scope "World" {}\n')
    result=audit_inventory(dict(scenes=[dict(task_id='CP04',difficulty='easy',
        assets=dict(scene='scene.usda',runtime_screen='missing-screen.png'),scene_asset_ids=['scene'])]),tmp_path)
    assert not result['passed']
    assert 'missing-screen.png' in result['scenes'][0]['issues'][0]['error']


def test_canonical_x5_camera_orphan_is_repaired_in_session_without_editing_asset(tmp_path, monkeypatch):
    import hashlib
    import mind2act_eval.robot as robot
    from pxr import Usd, UsdGeom
    source = tmp_path/'x5.usda'
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/X5A').GetPrim())
    mesh = UsdGeom.Mesh.Define(stage,'/X5A/camera/visuals/camera/World/mesh').GetPrim()
    mesh.CreateRelationship('material:binding').SetTargets(['/OmniKit_Viewport_LightRig/Looks/Ground_Material'])
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    unchanged = audit_scene(LocalAssets(tmp_path,{'x5':'x5.usda'}),['x5'])
    assert not unchanged['passed']  # An unpinned lookalike is not canonical X5.
    monkeypatch.setattr(robot,'PROFILE',dict(robot.PROFILE,asset_sha256=hashlib.sha256(original).hexdigest()))
    report = audit_scene(LocalAssets(tmp_path,{'x5':'x5.usda'}),['x5'])
    assert report['passed'], report['issues']
    assert len(report['render_repairs']) == 1
    assert report['render_repairs'][0]['after_targets'] == []
    assert source.read_bytes() == original


def test_random_native_search_requires_release_paths_first():
    from mind2act_eval.sim.material_closure import verify_native_mdl_search
    import pytest
    receipt = dict(resolved_paths=['/release/sdk/mdl'], prioritize_declared_paths=True)
    with pytest.raises(RuntimeError, match='prioritize pinned release paths'):
        verify_native_mdl_search(receipt, query=lambda: dict(system=[], user=[],
            active=['/installation/sdk/mdl', '/release/sdk/mdl']))
    result = verify_native_mdl_search(receipt, query=lambda: dict(system=[], user=[],
        active=['/release/sdk/mdl', '/installation/sdk/mdl']))
    assert result['native_compiler_paths_verified']
