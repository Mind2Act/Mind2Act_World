"""Real USD schema regressions for the native trace's hidden collision parents."""
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics


def test_hidden_guide_mesh_under_collision_xform_is_read_in_rigid_coordinates():
    from mind2act_eval.tasks.DR3.mesh import collider_vertices
    stage=Usd.Stage.CreateInMemory()
    body=UsdGeom.Xform.Define(stage,'/Link')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    collider=UsdGeom.Xform.Define(stage,'/Link/Collision')
    collider.AddTranslateOp().Set((.1,0,0))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    mesh=UsdGeom.Mesh.Define(stage,'/Link/Collision/mesh')
    vertices=np.asarray([[x,y,z] for x in (-.01,.01) for y in (-.02,.02) for z in (-.03,.03)])
    mesh.CreatePointsAttr(vertices.tolist())
    mesh.CreatePurposeAttr('guide');mesh.CreateVisibilityAttr('invisible')
    # This matches the actual native missing node_STL_BINARY_ hierarchy.
    assert UsdGeom.BBoxCache(0,['default','render']).ComputeUntransformedBound(collider.GetPrim()).ComputeAlignedRange().IsEmpty()
    result=collider_vertices(stage,collider.GetPrim(),body.GetPrim())
    assert np.allclose(result,vertices+[.1,0,0])
    assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_missing_collision_geometry_is_rejected_not_certified():
    from mind2act_eval.tasks.DR3.mesh import collider_vertices
    stage=Usd.Stage.CreateInMemory();root=stage.DefinePrim('/Empty','Xform')
    UsdPhysics.CollisionAPI.Apply(root)
    with pytest.raises(ValueError,match='mesh geometry'):
        collider_vertices(stage,root,root)


def test_floor_union_covers_tessellated_surface_without_filling_holes():
    from mind2act_eval.tasks.DR3.geometry import floor_region_covered
    points=np.linspace(-1,1,5)
    triangles=[]
    for i in range(4):
        for j in range(4):
            a,b,c,d=([points[i],points[j],0],[points[i+1],points[j],0],
                     [points[i+1],points[j+1],0],[points[i],points[j+1],0])
            triangles.extend([[a,b,c],[a,c,d]])
    assert floor_region_covered(triangles,[0,0],[.8,.8],0)
    # A floor's convex hull still surrounds this real hole; it must not pass.
    with_hole=[t for t in triangles if not np.all(np.abs(np.mean(t,axis=0)[:2])<.5)]
    assert not floor_region_covered(with_hole,[0,0],[.8,.8],0)


def test_collision_approximation_is_read_from_hidden_mesh_parent_without_writes():
    from mind2act_eval.tasks.DR3.mesh import collider_convex_parts
    stage=Usd.Stage.CreateInMemory();body=UsdGeom.Xform.Define(stage,'/Link')
    root=UsdGeom.Xform.Define(stage,'/Link/Collider')
    UsdPhysics.MeshCollisionAPI.Apply(root.GetPrim()).CreateApproximationAttr('convexHull')
    mesh=UsdGeom.Mesh.Define(stage,'/Link/Collider/mesh')
    mesh.CreatePointsAttr([[x,y,z] for x in (-.01,.01) for y in (-.02,.02) for z in (-.03,.03)])
    before=stage.GetRootLayer().ExportToString()
    parts=collider_convex_parts(stage,root.GetPrim(),body.GetPrim())
    assert len(parts)==1 and parts[0]['approximation']=='convexHull'
    assert parts[0]['approximation_source_path']=='/Link/Collider'
    assert stage.GetRootLayer().ExportToString()==before
    UsdPhysics.MeshCollisionAPI(root.GetPrim()).GetApproximationAttr().Set('none')
    assert collider_convex_parts(stage,root.GetPrim(),body.GetPrim()) is None


def test_captured_tray_floor_keeps_actual_extent_and_triangles():
    from mind2act_eval.tasks.DR3.capture import tray_floor_geometry
    origin = np.array([.4, -.2, .8])
    floor = np.array([[[-.08,-.1,0.],[.08,-.1,0.],[.08,.1,0.]],
                      [[-.08,-.1,0.],[.08,.1,0.],[-.08,.1,0.]]])
    wall = np.array([[[.08,-.1,0.],[.08,-.1,.05],[.08,.1,.05]]])
    triangles = np.concatenate([floor,wall])+origin
    captured = tray_floor_geometry(triangles, origin)
    assert np.allclose(captured, floor[:, :, :2])
    assert np.asarray(captured)[:, :, 0].max() == pytest.approx(.08)

