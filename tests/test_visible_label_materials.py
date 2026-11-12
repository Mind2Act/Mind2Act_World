"""Real composed label shaders; catches files that resolve but cannot render."""
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade
import pytest

from mind2act_eval.sim.material_closure import audit_material_graph


def scene(tmp_path, filename, *, uniform=False, hidden=False, name='VolumeTitle'):
    path=tmp_path/filename
    image=Image.new('RGB',(32,16),(0,210,220))
    if not uniform:
        image.paste((250,240,220),(4,4,24,12))
    image.save(path,format='PNG')
    stage=Usd.Stage.CreateInMemory()
    prim=UsdGeom.Cube.Define(stage,'/World/'+name).GetPrim()
    if hidden:UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    mat=UsdShade.Material.Define(stage,'/Looks/Label')
    surf=UsdShade.Shader.Define(stage,'/Looks/Label/Surface')
    surf.CreateIdAttr('UsdPreviewSurface')
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(),'surface')
    tex=UsdShade.Shader.Define(stage,'/Looks/Label/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
    surf.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    return stage,path


def test_resolved_png_without_extension_cannot_pass_label_closure(tmp_path):
    stage,_=scene(tmp_path,'5566479d7905')
    issues=audit_material_graph(stage)
    assert any(r['kind']=='unsupported_visible_texture_format' for r in issues)


def test_uniform_cyan_label_image_is_not_a_renderable_text_label(tmp_path):
    stage,_=scene(tmp_path,'label.png',uniform=True)
    assert any(r['kind']=='placeholder_label_texture' for r in audit_material_graph(stage))


def test_corrupt_image_with_valid_name_fails(tmp_path):
    stage,path=scene(tmp_path,'label.png')
    path.write_bytes(b'not a PNG image')
    assert any(r['kind']=='undecodable_visible_texture' for r in audit_material_graph(stage))


def test_original_nonuniform_png_passes(tmp_path):
    stage,_=scene(tmp_path,'label.png')
    assert audit_material_graph(stage)==[]


def test_unused_and_invisible_uniform_textures_do_not_fail(tmp_path):
    stage,_=scene(tmp_path,'unused.png',uniform=True,hidden=True)
    assert audit_material_graph(stage)==[]


def test_constant_surface_color_is_allowed_for_geometry_text(tmp_path):
    stage,_=scene(tmp_path,'label.png')
    surf=UsdShade.Shader(stage.GetPrimAtPath('/Looks/Label/Surface'))
    surf.GetInput('diffuseColor').DisconnectSource()
    surf.GetInput('diffuseColor').Set((.9,.9,.9))
    assert audit_material_graph(stage)==[]


def test_empty_texture_input_on_visible_label_fails(tmp_path):
    stage,_=scene(tmp_path,'label.png')
    texture=stage.GetPrimAtPath('/Looks/Label/Texture')
    texture.RemoveProperty('inputs:file')
    assert any(r['kind']=='missing_label_texture' for r in audit_material_graph(stage))


def test_constant_normal_map_does_not_make_a_legible_label_invalid(tmp_path):
    stage,_=scene(tmp_path,'label.png')
    path=tmp_path/'normal.png';Image.new('RGB',(4,4),(128,128,255)).save(path)
    surf=UsdShade.Shader(stage.GetPrimAtPath('/Looks/Label/Surface'))
    normal=UsdShade.Shader.Define(stage,'/Looks/Label/Normal')
    normal.CreateIdAttr('UsdUVTexture')
    normal.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
    surf.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(),'rgb')
    assert audit_material_graph(stage)==[]
