"""D1 delivery regressions for E1 static conveyor physics material and Belt Xform cleanup."""
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from mind2act_eval.tasks.E1.mechanism.stage_prep import _cargo_mass_kg, make_station_static


@pytest.mark.parametrize(
    ("identity", "dimensions_m", "expected_kg"),
    [
        ("Rigid/chocolate_bar/00000", [0.012529, 0.065, 0.007384], 0.00677),
        ("Rigid/chocolate_bar/00000", [0.0212034748, 0.11, 0.0124967555], 0.03283),
        ("Rigid/chocolate_bar/00001", [0.0196038065, 0.065, 0.0082187335], 0.00991),
        ("Rigid/chocolate_bar/00001", [0.033176, 0.11, 0.013909], 0.04803),
        ("Rigid/chocolate_bar/00002", [0.020266, 0.065, 0.007118], 0.00911),
        ("Rigid/chocolate_bar/00002", [0.034296, 0.11, 0.012047], 0.04413),
        ("Rigid/chocolate_bar/00001", [0.012529, 0.065, 0.007384], 0.08),
        ("Clutter/power_bank/00000", [0.027289, 0.065, 0.006951], 0.018),
        ("Rigid/power_bank/00002", [0.050, 0.109513, 0.020895], 0.155),
        ("Rigid/mouse/00009", [0.012529, 0.065, 0.007384], 0.08),
    ],
)
def test_cargo_mass_uses_identity_and_ordered_dimensions(identity, dimensions_m, expected_kg):
    item = {"visible_identity": identity, "dimensions_m": dimensions_m}
    assert _cargo_mass_kg(item) == pytest.approx(expected_kg)


def test_legacy_cargo_without_identity_and_dimensions_keeps_original_mass():
    assert _cargo_mass_kg({"prim_path": "/World/Items/Batch0/Item0"}) == pytest.approx(0.08)


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
