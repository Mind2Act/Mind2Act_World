"""Real composed USD counterexamples; no Kit/renderer/physics or GPU process."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade

from mind2act_eval.sim.asset_resolution import AssetUnresolved, WorldAssetGate, check_stage, inspect_stage


def reopen(stage, path):
    stage.GetRootLayer().Export(str(path))
    result = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    assert result is not None  # USD Open succeeds despite missing composition.
    return result


@pytest.mark.parametrize('arc', ['reference', 'payload', 'sublayer'])
def test_open_success_does_not_admit_missing_visible_geometry(tmp_path, arc):
    stage=Usd.Stage.CreateInMemory()
    prim=UsdGeom.Xform.Define(stage,'/World/Object').GetPrim()
    if arc=='reference':prim.GetReferences().AddReference('missing-geometry.usda')
    elif arc=='payload':prim.GetPayloads().AddPayload('missing-geometry.usda')
    else:
        UsdGeom.Cube.Define(stage,'/World/Cube')
        stage.GetRootLayer().subLayerPaths.append('missing-geometry.usda')
    stage=reopen(stage,tmp_path/'scene.usda')
    with pytest.raises(AssetUnresolved) as failure:
        check_stage(stage,tmp_path)
    report=json.loads((tmp_path/'asset-resolution-report.json').read_text())
    assert report['status']=='invalid' and report['invalid_reason']=='asset_unresolved'
    assert 'missing-geometry.usda' in ' '.join(failure.value.missing_paths)
    assert report['errors'] and report['elapsed_s']<5


@pytest.mark.parametrize('arc',['reference','payload'])
def test_missing_geometry_cannot_hide_by_losing_its_inherited_type(arc):
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World')
    prim=stage.DefinePrim('/World/Object')
    if arc=='reference':prim.GetReferences().AddReference('missing-typeless.usda')
    else:prim.GetPayloads().AddPayload('missing-typeless.usda')
    assert not prim.GetTypeName()
    report=inspect_stage(stage,mdl_search_paths=())
    assert report['status']=='invalid' and report['missing_paths']==['missing-typeless.usda']
    library=UsdGeom.Scope.Define(stage,'/Library').GetPrim()
    stage.RemovePrim('/World')
    unused=stage.DefinePrim('/Library/Unused')
    unused.GetReferences().AddReference('unused-material-library.usda')
    assert inspect_stage(stage,mdl_search_paths=())['errors']==[]


def material(stage, *, mdl=None):
    mesh=UsdGeom.Cube.Define(stage,'/World/Cube').GetPrim()
    mat=UsdShade.Material.Define(stage,'/Looks/Bound')
    shader=UsdShade.Shader.Define(stage,'/Looks/Bound/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    context='mdl' if mdl else ''
    mat.CreateSurfaceOutput(context).ConnectToSource(shader.ConnectableAPI(),'surface')
    if mdl:
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath(mdl),'mdl')
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    return mesh,mat,shader


@pytest.mark.parametrize('root_schema',[UsdGeom.Xform,UsdGeom.Scope])
def test_scene_scope_group_cannot_hide_missing_inherited_geometry(root_schema):
    stage=Usd.Stage.CreateInMemory()
    root=root_schema.Define(stage,'/World').GetPrim();stage.SetDefaultPrim(root)
    UsdGeom.Scope.Define(stage,'/World/Props')
    stage.DefinePrim('/World/Props/Object').GetReferences().AddReference('missing-grouped.usda')
    assert inspect_stage(stage,mdl_search_paths=())['missing_paths']==['missing-grouped.usda']
    UsdGeom.Imageable(root).CreateVisibilityAttr('invisible')
    report=inspect_stage(stage,mdl_search_paths=())
    assert not report['errors'] and any(w['kind']=='unused_composition' for w in report['warnings'])


def test_bound_connected_texture_missing_rejects_but_unused_shader_does_not(tmp_path):
    stage=Usd.Stage.CreateInMemory()
    _,_,shader=material(stage)
    unused=UsdShade.Shader.Define(stage,'/Looks/Unconnected')
    unused.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('unused.png'))
    report=inspect_stage(stage,mdl_search_paths=())
    assert not report['errors'] and any(w['kind']=='unused_shader_asset' for w in report['warnings'])
    texture=UsdShade.Shader.Define(stage,'/Looks/Bound/Texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('missing-used.png'))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    stage=reopen(stage,tmp_path/'texture.usda')
    with pytest.raises(AssetUnresolved) as failure:check_stage(stage,tmp_path,mdl_search_paths=())
    assert failure.value.missing_paths==['missing-used.png']


def test_unused_library_invisible_geometry_and_unselected_variant_are_warning_only(tmp_path):
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage,'/World/Good')
    library=UsdGeom.Scope.Define(stage,'/UnusedLibrary').GetPrim()
    library.GetReferences().AddReference('unused-library.usda')
    hidden=UsdGeom.Xform.Define(stage,'/World/Hidden')
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    hidden.GetPrim().GetReferences().AddReference('hidden-geometry.usda')
    variant=UsdGeom.Xform.Define(stage,'/World/Variant').GetPrim().GetVariantSets().AddVariantSet('choice')
    for name in ('good','bad'):variant.AddVariant(name)
    variant.SetVariantSelection('bad')
    with variant.GetVariantEditContext():
        UsdGeom.Xform.Define(stage,'/World/Variant/Bad').GetPrim().GetReferences().AddReference('unchosen.usda')
    variant.SetVariantSelection('good')
    report=inspect_stage(reopen(stage,tmp_path/'unused.usda'),mdl_search_paths=())
    assert report['errors']==[]
    assert any(w['kind']=='unused_composition' for w in report['warnings'])
    assert not any('unchosen.usda' in str(w) for w in report['warnings'])


def test_viewport_dangling_binding_and_builtin_material_are_not_invalid():
    stage=Usd.Stage.CreateInMemory();material(stage)
    light=UsdGeom.Xform.Define(stage,'/OmniKit_Viewport_LightRig').GetPrim()
    light.CreateRelationship('material:binding').SetTargets(['/OmniKit_Viewport_LightRig/Looks/Ground_Material'])
    report=inspect_stage(stage,mdl_search_paths=())
    assert report['status']=='passed' and not report['errors']
    assert any(w['kind']=='dangling_material_binding' for w in report['warnings'])


def test_invisible_collision_uses_geometry_but_not_its_visual_texture():
    from pxr import UsdPhysics
    stage=Usd.Stage.CreateInMemory();mesh,_,shader=material(stage)
    UsdGeom.Imageable(mesh).CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(mesh)
    shader.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('unused-hidden-texture.png'))
    report=inspect_stage(stage,mdl_search_paths=())
    assert not report['errors'] and any(w['path']=='unused-hidden-texture.png' for w in report['warnings'])
    mesh.GetReferences().AddReference('missing-physical-geometry.usda')
    assert inspect_stage(stage,mdl_search_paths=())['status']=='invalid'


def test_visible_light_schema_texture_is_used_but_custom_asset_metadata_is_not():
    from pxr import UsdLux
    stage=Usd.Stage.CreateInMemory()
    cube=UsdGeom.Cube.Define(stage,'/World/Cube').GetPrim()
    cube.CreateAttribute('debug:unusedFile',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('unused.bin'))
    assert not inspect_stage(stage,mdl_search_paths=())['errors']
    light=UsdLux.DomeLight.Define(stage,'/World/Light')
    light.CreateTextureFileAttr(Sdf.AssetPath('missing-environment.hdr'))
    assert inspect_stage(stage,mdl_search_paths=())['missing_paths']==['missing-environment.hdr']


@pytest.mark.parametrize('kind',['camera','light','instancer','volume'])
def test_only_drawn_surfaces_consume_a_bound_material(kind):
    from pxr import UsdLux, UsdVol
    stage=Usd.Stage.CreateInMemory();mesh,mat,shader=material(stage)
    stage.RemovePrim(str(mesh.GetPath()))
    schema={'camera':UsdGeom.Camera,'light':UsdLux.DomeLight,
            'instancer':UsdGeom.PointInstancer,'volume':UsdVol.Volume}[kind]
    prim=schema.Define(stage,'/World/Consumer').GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    texture=UsdShade.Shader.Define(stage,'/Looks/Bound/Texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('missing-bound.png'))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    report=inspect_stage(stage,mdl_search_paths=())
    if kind in ('camera','light'):
        assert not report['errors'] and any(w['path']=='missing-bound.png' for w in report['warnings'])
    else:
        assert report['missing_paths']==['missing-bound.png']


@pytest.mark.parametrize('token',['OmniPBR.mdl','gltf/pbr.mdl'])
def test_symbolic_mdl_uses_active_search_paths_including_file_uri(tmp_path,token):
    mdl=tmp_path/'sdk'/token;mdl.parent.mkdir(parents=True);mdl.write_text('mdl 1.7;')
    stage=Usd.Stage.CreateInMemory();material(stage,mdl=token)
    report=inspect_stage(stage,mdl_search_paths=((tmp_path/'sdk').as_uri(),))
    assert not report['errors'] and any(u['resolved']==str(mdl) for u in report['used'])


def test_explicit_used_mdl_file_missing_is_invalid(tmp_path):
    stage=Usd.Stage.CreateInMemory();material(stage,mdl=(tmp_path/'actually-missing.mdl').as_uri())
    report=inspect_stage(stage,mdl_search_paths=())
    assert report['status']=='invalid' and 'actually-missing.mdl' in report['missing_paths'][0]


@pytest.mark.parametrize('token',['path','file_uri','symlink_uri'])
def test_existing_used_texture_in_deprecated_root_refuses_without_asset_unresolved(tmp_path,token):
    from mind2act_eval.release_contract import ReleaseContractError
    old=tmp_path/'old';old.mkdir();texture_path=old/'texture.png';texture_path.write_bytes(b'image')
    (old/'DEPRECATED').write_text('retired')
    if token=='symlink_uri':
        alias=tmp_path/'alias.png';alias.symlink_to(texture_path);value=alias.as_uri()
    else:value=texture_path.as_uri() if token=='file_uri' else str(texture_path)
    stage=Usd.Stage.CreateInMemory();_,_,shader=material(stage)
    texture=UsdShade.Shader.Define(stage,'/Looks/Bound/Texture');texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(value))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    with pytest.raises(ReleaseContractError,match='DEPRECATED') as failure:
        check_stage(stage,tmp_path/'output',mdl_search_paths=())
    assert not isinstance(failure.value,AssetUnresolved)
    assert json.loads((tmp_path/'output/asset-resolution-report.json').read_text())['missing_paths']==[]


def test_unused_old_texture_and_anonymous_layer_are_not_deprecated_files(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();(old/'DEPRECATED').write_text('retired')
    texture=old/'unused.png';texture.write_bytes(b'image');monkeypatch.chdir(old)
    stage=Usd.Stage.CreateInMemory();UsdGeom.Cube.Define(stage,'/World/Cube')
    unused=UsdShade.Shader.Define(stage,'/Looks/Unused')
    unused.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture.as_uri()))
    report=check_stage(stage,tmp_path/'output',mdl_search_paths=())
    assert report['status']=='passed' and any(w['path']==texture.as_uri() for w in report['warnings'])


def test_used_package_alias_checks_deprecated_outer_archive(tmp_path):
    from pxr import UsdUtils
    from mind2act_eval.release_contract import ReleaseContractError
    old=tmp_path/'old';old.mkdir()
    source=Usd.Stage.CreateNew(str(old/'cube.usda'));cube=UsdGeom.Cube.Define(source,'/Cube')
    source.SetDefaultPrim(cube.GetPrim());source.GetRootLayer().Save()
    package=old/'cube.usdz';assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(old/'cube.usda')),str(package))
    alias=tmp_path/'alias.usdz';alias.symlink_to(package);(old/'DEPRECATED').write_text('retired')
    stage=Usd.Stage.CreateInMemory();stage.DefinePrim('/World/Cube').GetReferences().AddReference(str(alias))
    with pytest.raises(ReleaseContractError,match='DEPRECATED'):
        check_stage(stage,tmp_path/'output',mdl_search_paths=())


def test_existing_reference_file_with_missing_geometry_target_is_invalid(tmp_path):
    source=Usd.Stage.CreateNew(str(tmp_path/'exists.usda'))
    UsdGeom.Cube.Define(source,'/Actual');source.GetRootLayer().Save()
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World/Object').GetPrim().GetReferences().AddReference(str(tmp_path/'exists.usda'),'/Absent')
    report=inspect_stage(stage,mdl_search_paths=())
    assert report['status']=='invalid' and any('/Absent' in p for p in report['missing_paths'])


def test_time_zero_shader_asset_value_is_checked(tmp_path):
    stage=Usd.Stage.CreateInMemory();_,_,shader=material(stage)
    texture=UsdShade.Shader.Define(stage,'/Looks/Bound/Texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('time-zero-missing.png'),0)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    assert inspect_stage(stage,mdl_search_paths=())['missing_paths']==['time-zero-missing.png']


def test_world_gate_does_not_keep_native_world_alive(tmp_path):
    import weakref
    stage=Usd.Stage.CreateInMemory()
    class World:
        def reset(self):pass
        def step(self):pass
    world=World();reference=weakref.ref(world)
    gate=WorldAssetGate(world,stage,tmp_path)
    del world
    try:assert reference() is None
    finally:gate.close()


def test_package_member_uses_usd_resolver_not_filesystem_path(tmp_path):
    from pxr import UsdUtils
    source=Usd.Stage.CreateNew(str(tmp_path/'inner.usda'))
    UsdGeom.Cube.Define(source,'/Cube');source.SetDefaultPrim(source.GetPrimAtPath('/Cube'))
    source.GetRootLayer().Save()
    archive=tmp_path/'packed.usdz'
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(tmp_path/'inner.usda')),str(archive))
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World/Packed').GetPrim().GetReferences().AddReference(str(archive))
    report=inspect_stage(stage,mdl_search_paths=())
    assert not report['errors']
    assert Ar.GetResolver().Resolve(str(archive)+'[inner.usda]')


def test_actual_first_reset_and_direct_step_never_reach_physics_when_invalid(tmp_path):
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World/Object').GetPrim().GetReferences().AddReference('missing.usda')
    count=[]
    world=SimpleNamespace(reset=lambda:count.append('reset'),step=lambda **kw:count.append('tick'))
    gate=WorldAssetGate(world,stage,tmp_path)
    try:
        for operation in (world.reset,world.step):
            with pytest.raises(AssetUnresolved):operation()
        assert count==[]
    finally:gate.close()


def test_new_connected_asset_invalidates_cached_gate_without_charging_normal_ticks(tmp_path):
    stage=Usd.Stage.CreateInMemory();_,_,shader=material(stage)
    texture=UsdShade.Shader.Define(stage,'/Looks/Bound/Texture')
    texture.CreateIdAttr('UsdUVTexture')
    existing=tmp_path/'existing.png';existing.write_bytes(b'existence-only fixture')
    file=texture.CreateInput('file',Sdf.ValueTypeNames.Asset)
    file.Set(Sdf.AssetPath(str(existing)))
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    count=[];world=SimpleNamespace(reset=lambda:None,step=lambda:count.append(1))
    gate=WorldAssetGate(world,stage,tmp_path)
    try:
        world.step();assert count==[1]
        file.Set(Sdf.AssetPath('new-missing.png'))
        with pytest.raises(AssetUnresolved):world.step()
        assert count==[1]
    finally:gate.close()


def test_existing_shader_connection_change_invalidates_cache_before_next_tick(tmp_path):
    stage=Usd.Stage.CreateInMemory();_,_,shader=material(stage)
    color=shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f)
    texture=UsdShade.Shader.Define(stage,'/Looks/AlreadyAuthoredTexture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('already-authored-missing.png'))
    texture.CreateOutput('rgb',Sdf.ValueTypeNames.Color3f)
    ticks=[];world=SimpleNamespace(reset=lambda:None,step=lambda:ticks.append(1))
    gate=WorldAssetGate(world,stage,tmp_path)
    try:
        world.step();assert ticks==[1] and gate.dirty is False
        color.ConnectToSource(texture.ConnectableAPI(),'rgb')
        assert gate.dirty is True
        with pytest.raises(AssetUnresolved):world.step()
        assert ticks==[1]
    finally:gate.close()


def test_enabling_hidden_collision_rechecks_its_previously_unused_geometry(tmp_path):
    from pxr import UsdPhysics
    stage=Usd.Stage.CreateInMemory()
    mesh=UsdGeom.Cube.Define(stage,'/World/Hidden')
    mesh.CreateVisibilityAttr('invisible')
    collision=UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(False)
    mesh.GetPrim().GetReferences().AddReference('previously-unused-physics.usda')
    ticks=[];world=SimpleNamespace(reset=lambda:None,step=lambda:ticks.append(1))
    gate=WorldAssetGate(world,stage,tmp_path)
    try:
        world.step();assert ticks==[1] and not gate.dirty
        collision.GetCollisionEnabledAttr().Set(True)
        assert gate.dirty
        with pytest.raises(AssetUnresolved):world.step()
        assert ticks==[1]
    finally:gate.close()


def test_runner_publishes_null_invalid_reason_before_adapter_or_score(tmp_path):
    from test_runner import fixture
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.contracts import ConsumerKind
    from mind2act_eval.runner import EpisodeLimits,run_episode
    from mind2act_eval.tasks import TaskDefinition
    spec,_,ctx,events=fixture(tmp_path)
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World/Missing').GetPrim().GetReferences().AddReference('missing-geometry.usda')
    def build(spec,ctx):check_stage(stage,ctx.output_dir,mdl_search_paths=())
    class Adapter(DemoAdapter):
        def reset(self,seed):pytest.fail('adapter started after asset rejection')
    task=TaskDefinition('demo',build,lambda evidence:pytest.fail('invalid assets scored'))
    result=run_episode(spec,task,ctx,Adapter(),consumer_kind=ConsumerKind.DIAGNOSTIC,limits=EpisodeLimits(4,10))
    assert not result.infra_valid and result.ticks==0 and result.score is None and not events
    assert result.task_metadata['invalid_reason']=='asset_unresolved'
    assert result.task_metadata['missing_paths']==('missing-geometry.usda',)
    assert not (tmp_path/'score.json').exists()


def test_all_45_dry_manifest_keeps_44_original_pass_and_unknown_without_kit(tmp_path,capsys):
    from mind2act_eval.asset_resolution_dry_run import OLD_TASKS,main,read_cases
    from mind2act_eval.contracts import InstanceSpec,to_json
    rows=[]
    for task in OLD_TASKS:
        for tier in ('easy','medium','hard'):
            spec=InstanceSpec(task,task+'-'+tier,tier,'original-source','v1','none','v1',seed=17)
            path=tmp_path/(task+'-'+tier+'.json');path.write_text(json.dumps(to_json(spec)))
            rows.append(dict(options=dict(spec=str(path),asset_root=str(tmp_path)),
                original_status='unknown' if (task,tier)==('PR3','easy') else 'PASS'))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(rows))
    cases=read_cases(manifest)
    assert len(cases)==45 and sum(r['original_status']=='PASS' for r,_ in cases)==44
    assert next(r['original_status'] for r,s in cases if (s.task_id,s.difficulty)==('PR3','easy'))=='unknown'
    assert main(['--manifest',str(manifest),'--output',str(tmp_path/'never-created'),'--gpu','0','--validate-only'])==0
    assert json.loads(capsys.readouterr().out)['kit_started'] is False
    assert not (tmp_path/'never-created').exists()
    manifest.write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError,match='all 45'):read_cases(manifest)
    manifest.write_text(json.dumps(rows[:-1]+[rows[0]]))
    with pytest.raises(ValueError,match='duplicate'):read_cases(manifest)
