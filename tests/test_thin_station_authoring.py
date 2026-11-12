"""Task authoring must work on the real SHA-checked thin station projection."""
import hashlib
import json
from pathlib import Path

import pytest

from pxr import Sdf, Usd, UsdGeom, UsdShade

from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim.asset_relocation import prepare_station
from mind2act_eval.sim.stage_authoring import copy_local_prim, remove_composed_prim
from mind2act_eval.tasks.CP02.mechanism.native_assets import ROOT, restore_authored


def test_cp02_restore_copies_inherited_geometry_materials_and_removes_templates(tmp_path, monkeypatch):
    monkeypatch.setenv('M2A_CACHE_ROOT',str(tmp_path/'node-cache'))
    root=tmp_path/'assets';root.mkdir()
    source=root/'source.usda'
    stage=Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage,'/World')
    UsdGeom.Xform.Define(stage,'/World/CakeTemplate/CakeAssembly').AddTranslateOp().Set((0,0,0))
    cube=UsdGeom.Cube.Define(stage,'/World/CakeTemplate/CakeAssembly/Cake');cube.CreateSizeAttr(.15)
    material=UsdShade.Material.Define(stage,'/World/Materials/Cake')
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    UsdGeom.Xform.Define(stage,'/World/CakeTemplate/Source')
    UsdGeom.Xform.Define(stage,'/World/Props/Box').AddTranslateOp().Set((.2,.3,0))
    UsdGeom.Xform.Define(stage,'/World/Props/Lid').AddTranslateOp().Set((.2,.4,0))
    UsdGeom.Xform.Define(stage,'/World/Props/Lid/Handle').AddTranslateOp().Set((0,0,.1))
    stage.GetRootLayer().Save()
    before=source.read_bytes()
    relocation=dict(schema='mind2act_eval.station_relocation.v1',source_sha256=hashlib.sha256(before).hexdigest(),mappings=[])
    (root/'relocation.json').write_text(json.dumps(relocation))
    assets=LocalAssets(root,dict(station='source.usda',relocation='relocation.json'))
    output=tmp_path/'attempt'
    thin=prepare_station(source,assets,'relocation',output)
    receipt=json.loads((output/'station-relocation.json').read_text())
    immutable=Path(receipt['immutable_runtime_path'])
    cache_before=immutable.read_bytes()
    stage=Usd.Stage.Open(str(thin))
    assert not stage.GetRootLayer().GetPrimAtPath('/World/CakeTemplate/CakeAssembly')
    result=restore_authored(stage,dict(spec=dict(ingredient_shift_xy_m=[.1,.2],center_xy_m=[0,0],
        assembly_z_m=.1,difficulty='easy'),v4=dict(buttons={})))
    assert result['lid_pivot']==pytest.approx([.3,.6,0])
    copied=stage.GetPrimAtPath(ROOT+'/CakeAssembly/Cake')
    assert UsdGeom.Cube(copied).GetSizeAttr().Get()==.15
    assert str(UsdShade.MaterialBindingAPI(copied).ComputeBoundMaterial()[0].GetPath())==ROOT+'/Looks/Authored_0'
    for path in ('/World/CakeTemplate','/World/Props'):
        prim=stage.GetPrimAtPath(path)
        assert not prim or not prim.IsActive()
    assert source.read_bytes()==before and immutable.read_bytes()==cache_before


def test_copy_includes_current_overrides_new_prototypes_and_relative_references(tmp_path):
    asset=tmp_path/'asset.usda'
    authored=Usd.Stage.CreateNew(str(asset));UsdGeom.Cube.Define(authored,'/Geom').CreateSizeAttr(2)
    authored.SetDefaultPrim(authored.GetPrimAtPath('/Geom'));authored.GetRootLayer().Save()
    base=tmp_path/'base.usda'
    authored=Usd.Stage.CreateNew(str(base))
    UsdGeom.Xform.Define(authored,'/World/Template')
    UsdGeom.Cube.Define(authored,'/World/Template/Cube').CreateSizeAttr(.2)
    authored.GetPrimAtPath('/World/Template').CreateAttribute('mdl',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('gltf/pbr.mdl'))
    (tmp_path/'texture.png').write_bytes(b'asset-path-only fixture')
    authored.GetPrimAtPath('/World/Template').CreateAttribute('image',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('texture.png'))
    authored.DefinePrim('/World/Template/External').GetReferences().AddReference('asset.usda')
    authored.GetRootLayer().Save();before=base.read_bytes()
    thin=Sdf.Layer.CreateNew(str(tmp_path/'thin.usda'));thin.subLayerPaths=['base.usda'];thin.Save()
    stage=Usd.Stage.Open(thin)
    cube=UsdGeom.Cube(stage.GetPrimAtPath('/World/Template/Cube'))
    cube.GetSizeAttr().Set(.4);UsdGeom.Imageable(cube).MakeInvisible()
    copy_local_prim(stage,'/World/Template','/World/Clone')
    assert UsdGeom.Cube(stage.GetPrimAtPath('/World/Clone/Cube')).GetSizeAttr().Get()==.4
    assert UsdGeom.Imageable(stage.GetPrimAtPath('/World/Clone/Cube')).ComputeVisibility()=='invisible'
    assert UsdGeom.Cube(stage.GetPrimAtPath('/World/Clone/External')).GetSizeAttr().Get()==2
    assert stage.GetPrimAtPath('/World/Clone').GetAttribute('mdl').Get().path=='gltf/pbr.mdl'
    image=stage.GetPrimAtPath('/World/Clone').GetAttribute('image').Get()
    assert Path(image.resolvedPath)==tmp_path/'texture.png'
    # CP03 can copy a just-materialized prototype after changing its runtime pose.
    UsdGeom.Cube(stage.GetPrimAtPath('/World/Clone/Cube')).GetSizeAttr().Set(.6)
    copy_local_prim(stage,'/World/Clone','/World/Clone2')
    assert UsdGeom.Cube(stage.GetPrimAtPath('/World/Clone2/Cube')).GetSizeAttr().Get()==.6
    remove_composed_prim(stage,'/World/Template')
    assert not stage.GetPrimAtPath('/World/Template').IsActive()
    remove_composed_prim(stage,'/World/Clone')
    assert not stage.GetPrimAtPath('/World/Clone')
    assert base.read_bytes()==before
