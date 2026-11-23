from types import SimpleNamespace

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from mind2act_eval.variations.stage import _bind_distractor_physics_material


def test_explicit_distractor_friction_binds_enabled_shapes():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World/Clutter/Box').GetPrim()
    shape = UsdGeom.Cube.Define(stage, '/World/Clutter/Box/Shape').GetPrim()
    disabled = UsdGeom.Cube.Define(stage, '/World/Clutter/Box/Disabled').GetPrim()
    UsdPhysics.CollisionAPI.Apply(shape)
    UsdPhysics.CollisionAPI.Apply(disabled).CreateCollisionEnabledAttr(False)

    item = SimpleNamespace(name='Box', physics={'friction': .45})
    receipt = _bind_distractor_physics_material(stage, '/World/Clutter', item, root)
    assert receipt['status'] == 'legacy_scalar_metadata_only'
    assert not UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial(materialPurpose='physics')[0]

    item.physics = {'static_friction': .6, 'dynamic_friction': .45, 'restitution': .05}
    receipt = _bind_distractor_physics_material(stage, '/World/Clutter', item, root)
    assert receipt['status'] == 'bound_explicit'
    assert receipt['colliders'] == [str(shape.GetPath())]
    bound, _ = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial(materialPurpose='physics')
    material = UsdPhysics.MaterialAPI(bound.GetPrim())
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(.6)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(.45)
    assert material.GetRestitutionAttr().Get() == pytest.approx(.05)
    assert not UsdShade.MaterialBindingAPI(disabled).ComputeBoundMaterial(materialPurpose='physics')[0]
