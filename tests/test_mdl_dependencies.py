"""A present MDL file can still fail to compile when an import is absent."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pxr import Usd, UsdGeom, UsdShade

from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim.asset_resolution import inspect_stage
from mind2act_eval.sim.material_closure import audit_scene
from test_asset_resolution import material


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_present_used_wood_mdl_does_not_admit_missing_automotive_import(tmp_path):
    wood = write(tmp_path/'Material/Wood/Grain.mdl',
                 'mdl 1.8; import ..::Templates::OmniUber_Automotive::*;')
    stage = Usd.Stage.CreateInMemory()
    material(stage, mdl=str(wood))
    result = inspect_stage(stage, mdl_search_paths=())
    assert result['status'] == 'invalid'
    assert any(row['kind'] == 'unresolved_mdl_import' and
               'OmniUber_Automotive' in row['path'] for row in result['errors'])


def test_cpu_audit_resolves_parent_package_import_from_original_location(tmp_path):
    write(tmp_path/'scene.usda', '#usda 1.0\ndef Scope "World" {\n custom asset mdl = @Material/Wood/Grain.mdl@\n}\n')
    write(tmp_path/'Material/Wood/Grain.mdl',
          'mdl 1.8; import ..::Templates::OmniUber_Automotive::*;')
    write(tmp_path/'Material/Templates/OmniUber_Automotive.mdl',
          'mdl 1.8; import ::math::*; import ::scene::*;')
    assets = LocalAssets(tmp_path, {'scene': 'scene.usda'})
    result = audit_scene(assets, ['scene'])
    assert result['passed'], result['issues']


def test_native_gate_requires_compiler_success_and_compiles_shared_module_once(tmp_path, monkeypatch):
    from mind2act_eval.sim import asset_resolution
    wood = write(tmp_path/'Grain.mdl', 'mdl 1.8; invalid_material_body;')
    stage = Usd.Stage.CreateInMemory()
    _, mat, _ = material(stage, mdl=str(wood))
    other = UsdGeom.Cube.Define(stage, '/World/Other').GetPrim()
    UsdShade.MaterialBindingAPI.Apply(other).Bind(mat)
    calls = []
    monkeypatch.setattr(asset_resolution, 'native_mdl_paths', lambda: ())
    def compiler(path):
        calls.append(path)
        return dict(passed=False, compiler='fixture', detail='compiler type error')
    monkeypatch.setattr(asset_resolution, '_native_mdl_compile', compiler)
    result = inspect_stage(stage)
    assert result['status'] == 'invalid'
    assert len(calls) == 1
    assert [row['kind'] for row in result['errors']] == ['mdl_compilation_failed']
    check = next(iter(result['mdl_checks'].values()))
    assert check['passed'] and check['compilation']['passed'] is False


def test_unused_broken_mdl_never_invalidates_visible_preview_material(tmp_path):
    wood = write(tmp_path/'Grain.mdl', 'mdl 1.8; import ::missing::*;')
    stage = Usd.Stage.CreateInMemory()
    material(stage)
    unused = UsdShade.Shader.Define(stage, '/Unused/Shader')
    unused.SetSourceAsset(str(wood), 'mdl')
    result = inspect_stage(stage, mdl_search_paths=())
    assert result['status'] == 'passed' and not result['mdl_checks']


def test_nested_import_resource_and_annotation_are_checked_at_their_module(tmp_path):
    from mind2act_eval.sim.mdl_dependencies import audit_mdl
    wood = write(tmp_path/'Wood/Grain.mdl', 'mdl 1.8; using ..::Templates::Uber import surface;')
    write(tmp_path/'Templates/Uber.mdl',
          'mdl 1.8; import ::math::*; string description="import ::not_an_import::*;"; texture_2d("grain.png");')
    result = audit_mdl(wood)
    assert [x['kind'] for x in result['issues']] == ['unresolved_mdl_resource']
    write(tmp_path/'Templates/grain.png', 'original bytes')
    assert audit_mdl(wood)['passed']


def test_visibility_updates_reuse_compilation_but_changed_asset_rechecks_before_physics(tmp_path, monkeypatch):
    from mind2act_eval.sim import asset_resolution
    wood = write(tmp_path/'Grain.mdl', 'mdl 1.8;')
    broken = write(tmp_path/'Broken.mdl', 'mdl 1.8; import ::absent::*;')
    stage = Usd.Stage.CreateInMemory()
    mesh, _, shader = material(stage, mdl=str(wood))
    image = UsdGeom.Imageable(mesh)
    image.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    compiled, steps = [], []
    monkeypatch.setattr(asset_resolution, 'native_mdl_paths', lambda: ())
    def compile_module(path):
        compiled.append(path)
        return dict(passed=True, compiler='fixture')
    monkeypatch.setattr(asset_resolution, '_native_mdl_compile', compile_module)
    world = SimpleNamespace(reset=lambda: None, step=lambda: steps.append('physics'))
    gate = asset_resolution.WorldAssetGate(world, stage, tmp_path)
    try:
        world.step()
        image.MakeInvisible(); world.step()
        image.MakeVisible(); world.step()
        assert compiled == [str(wood)]
        shader.SetSourceAsset(str(broken), 'mdl')
        with pytest.raises(asset_resolution.AssetUnresolved, match='absent'):
            world.step()
        assert len(steps) == 3
    finally:
        gate.close()
