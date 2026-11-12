"""Actual USD collision vertices, including empty API-only prims."""
import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from mind2act_eval.tasks.CP06.geometry import collider_vertices


def test_empty_collision_api_has_no_vertices_and_local_transform_applies_once():
    stage=Usd.Stage.CreateInMemory()
    body=UsdGeom.Xform.Define(stage,'/World/Pan');body.AddTranslateOp().Set((1.,2.,3.))
    empty=UsdGeom.Xform.Define(stage,'/World/Pan/ApiOnly');UsdPhysics.CollisionAPI.Apply(empty.GetPrim())
    floor=UsdGeom.Cube.Define(stage,'/World/Pan/Floor');floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set((0.,0.,.02));floor.AddScaleOp().Set((.2,.2,.01))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    vertices=collider_vertices(stage,'/World/Pan')
    assert len(vertices)==8
    assert vertices.min(axis=0)==pytest.approx([-.1,-.1,.015])
    assert vertices.max(axis=0)==pytest.approx([.1,.1,.025])


def test_api_only_groups_and_disabled_shapes_do_not_expand_actual_collision_center():
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/Body')
    wrapper=UsdGeom.Xform.Define(stage,'/World/Body/Visual');UsdPhysics.CollisionAPI.Apply(wrapper.GetPrim())
    visual=UsdGeom.Cube.Define(stage,'/World/Body/Visual/Model');visual.CreateSizeAttr(10.)
    disabled=UsdGeom.Cube.Define(stage,'/World/Body/Disabled');disabled.CreateSizeAttr(20.)
    UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
    actual=UsdGeom.Cube.Define(stage,'/World/Body/Actual');actual.CreateSizeAttr(.02)
    UsdPhysics.CollisionAPI.Apply(actual.GetPrim())
    vertices=collider_vertices(stage,'/World/Body')
    assert len(vertices)==8
    assert vertices.min(axis=0)==pytest.approx([-.01]*3)
    assert vertices.max(axis=0)==pytest.approx([.01]*3)
