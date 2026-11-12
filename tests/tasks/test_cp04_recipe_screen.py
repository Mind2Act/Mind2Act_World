"""Source-equivalent single-mesh recipe screen; real USD, no Kit/physics."""
from types import SimpleNamespace as NS

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from mind2act_eval.tasks.CP04.scene import CookingScene
from .test_cp04_build_binding import instance
from .test_cp04_evaluation_clock import evaluation_spec


def screen_pair():
    stage=Usd.Stage.CreateInMemory()
    recipe=UsdShade.Material.Define(stage,'/Looks/Recipe')
    ready=UsdShade.Material.Define(stage,'/Looks/Ready')
    visible=UsdGeom.Mesh.Define(stage,'/World/RecipeScreen')
    hidden=UsdGeom.Mesh.Define(stage,'/World/RecipeScreenHidden')
    for mesh,material in ((visible,recipe),(hidden,ready)):
        mesh.CreatePointsAttr([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdGeom.Imageable(hidden).MakeInvisible()
    return stage,visible.GetPrim(),hidden.GetPrim(),recipe,ready


@pytest.mark.parametrize('tier',['easy','medium','hard'])
@pytest.mark.parametrize('clock',['evaluation','source'])
def test_recipe_deadline_rebinds_original_screen_like_capture_source(tier,clock):
    stage,visible,hidden,recipe,ready=screen_pair()
    spec=evaluation_spec(tier) if clock=='evaluation' else instance(tier)
    n=dict(counter={'step':0},visible=visible,hidden=hidden,
           appliance=NS(post_step=lambda dt,update_display:{}),
           observer=NS(thermal_sample=lambda *a:None,hidden=None),sample=lambda:None)
    scene=CookingScene(spec,None,n);substeps=scene.clock.substeps
    points=UsdGeom.Mesh(visible).GetPointsAttr().Get()
    scene.after_step(419*substeps)
    assert UsdShade.MaterialBindingAPI(visible).ComputeBoundMaterial()[0].GetPath()==recipe.GetPath()
    scene.after_step(420*substeps)
    # Source RecipeDisplay.hide binds a material on self.cook.recipe. It does
    # not replace a draw primitive with an identical coplanar duplicate.
    assert UsdGeom.Imageable(visible).ComputeVisibility()!='invisible'
    assert UsdGeom.Imageable(hidden).ComputeVisibility()=='invisible'
    assert UsdShade.MaterialBindingAPI(visible).ComputeBoundMaterial()[0].GetPath()==ready.GetPath()
    assert UsdGeom.Mesh(visible).GetPointsAttr().Get()==points
    assert n['observer'].hidden==420 and n['counter']['step']==420


def test_missing_ready_binding_does_not_hide_the_working_screen():
    from mind2act_eval.tasks.CP04.mechanism.recipe_screen import show_ready
    stage,visible,hidden,recipe,_=screen_pair()
    UsdShade.MaterialBindingAPI(hidden).UnbindAllBindings()
    with pytest.raises(RuntimeError,match='READY screen material missing'):
        show_ready(visible,hidden)
    assert UsdGeom.Imageable(visible).ComputeVisibility()!='invisible'
    assert UsdShade.MaterialBindingAPI(visible).ComputeBoundMaterial()[0].GetPath()==recipe.GetPath()


def test_ready_swap_changes_no_geometry_shader_or_physics_opinions():
    from mind2act_eval.tasks.CP04.mechanism.recipe_screen import show_ready
    stage,visible,hidden,_,ready=screen_pair()
    visible.CreateAttribute('physics:collisionEnabled',Sdf.ValueTypeNames.Bool).Set(False)
    source=stage.GetRootLayer().ExportToString()
    stage.SetEditTarget(stage.GetSessionLayer())
    show_ready(visible,hidden)
    assert stage.GetRootLayer().ExportToString()==source
    paths=[];stage.GetSessionLayer().Traverse(Sdf.Path.absoluteRootPath,lambda p:paths.append(p))
    allowed={'visibility','material:binding'}
    for path in paths:
        obj=stage.GetSessionLayer().GetObjectAtPath(path)
        if isinstance(obj,(Sdf.AttributeSpec,Sdf.RelationshipSpec)):
            assert obj.name in allowed
    assert UsdShade.MaterialBindingAPI(visible).ComputeBoundMaterial()[0].GetPath()==ready.GetPath()
    assert visible.GetAttribute('physics:collisionEnabled').Get() is False
