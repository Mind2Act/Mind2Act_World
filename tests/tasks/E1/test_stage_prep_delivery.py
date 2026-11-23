"""D1 delivery regressions for E1 static conveyor physics material and Belt Xform cleanup."""
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from mind2act_eval.tasks.E1.mechanism.stage_prep import make_station_static


def _e1_conveyor_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Conveyor/Asset/ConveyorTrack")

    # Visual-only material without UsdPhysics.MaterialAPI (mirrors upstream Aluminum_Anodized_Black).
    render_only = UsdShade.Material.Define(stage, "/World/Conveyor/Asset/Looks/Aluminum_Anodized_Black")

    def _unit_mesh(path):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        return mesh

    frame = _unit_mesh("/World/Conveyor/Asset/ConveyorTrack/SM_ConveyorBelt_A09_02")
    UsdPhysics.CollisionAPI.Apply(frame.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(frame.GetPrim()).CreateApproximationAttr("sdf")

    decal = _unit_mesh("/World/Conveyor/Asset/ConveyorTrack/SM_ConveyorBelt_A09_Decal_02")
    UsdShade.MaterialBindingAPI.Apply(decal.GetPrim()).Bind(render_only)

    # Shapeless Belt Xform carrying stray CollisionAPI, render-only physics binding, and -0.1 m/s surfaceVelocity.
    belt_xf = UsdGeom.Xform.Define(stage, "/World/Conveyor/Asset/ConveyorTrack/Belt").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(belt_xf).CreateRigidBodyEnabledAttr(False)
    UsdPhysics.CollisionAPI.Apply(belt_xf).CreateCollisionEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(belt_xf).Bind(render_only, materialPurpose="physics")
    belt_xf.CreateAttribute("physxSurfaceVelocity:surfaceVelocityEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    belt_xf.CreateAttribute("physxSurfaceVelocity:surfaceVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(-0.1, 0.0, 0.0))

    _unit_mesh("/World/Conveyor/Asset/ConveyorTrack/Belt/SM_ConveyorBelt_A09_Belt_02")

    cube = UsdGeom.Cube.Define(stage, "/World/Conveyor/Asset/Cube")
    cube.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return stage


def test_make_station_static_binds_conveyor_colliders_and_clears_belt_xform():
    stage = _e1_conveyor_stage()
    make_station_static(stage)

    station_mat = stage.GetPrimAtPath("/World/StationPhysicsMaterial")
    assert station_mat.IsValid()
    assert station_mat.HasAPI(UsdPhysics.MaterialAPI)
    pm = UsdPhysics.MaterialAPI(station_mat)
    assert pm.GetStaticFrictionAttr().Get() == pytest.approx(0.9)
    assert pm.GetDynamicFrictionAttr().Get() == pytest.approx(0.8)
    assert pm.GetRestitutionAttr().Get() == pytest.approx(0.0)

    expected_colliders = (
        "/World/Conveyor/Asset/ConveyorTrack/SM_ConveyorBelt_A09_02",
        "/World/Conveyor/Asset/ConveyorTrack/SM_ConveyorBelt_A09_Decal_02",
        "/World/Conveyor/Asset/ConveyorTrack/Belt/SM_ConveyorBelt_A09_Belt_02",
        "/World/Conveyor/Asset/Cube",
    )
    for path in expected_colliders:
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        assert str(bound.GetPath()) == "/World/StationPhysicsMaterial"

    belt_xf = stage.GetPrimAtPath("/World/Conveyor/Asset/ConveyorTrack/Belt")
    assert not belt_xf.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not belt_xf.HasAPI(UsdPhysics.CollisionAPI)
    bound_belt, _ = UsdShade.MaterialBindingAPI(belt_xf).ComputeBoundMaterial(materialPurpose="physics")
    assert not bound_belt.GetPrim().IsValid()
    assert belt_xf.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled").Get() is False
    assert tuple(belt_xf.GetAttribute("physxSurfaceVelocity:surfaceVelocity").Get()) == pytest.approx((0.0, 0.0, 0.0))

    # OpenUSD physics parser must resolve a valid PhysicsMaterial on every parsed conveyor shape.
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, [Sdf.Path("/World/Conveyor")])
    resolved_shapes = {}
    for kind, (paths, descs) in parsed.items():
        if "Shape" in str(kind):
            for p, desc in zip(paths, descs):
                resolved_shapes[str(p)] = [str(m) for m in desc.materials]
    assert set(resolved_shapes) == set(expected_colliders)
    for path, mats in resolved_shapes.items():
        assert mats == ["/World/StationPhysicsMaterial"], f"{path} materials={mats}"
