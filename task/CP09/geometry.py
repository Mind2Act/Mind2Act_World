"""Acquisition finger collision overlay declaration and persistent read-only audit."""
from mind2act_eval.robot.cp09_pad_geometry import descriptor as acquisition_descriptor


def descriptor(difficulty=None):
    """All difficulties retain the same acquisition geometry asset."""
    return acquisition_descriptor()


def verify(stage):
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade
    import numpy as np
    for row in descriptor()['pads']:
        prim=stage.GetPrimAtPath(row['path'])
        if not prim or not prim.IsActive() or not prim.IsDefined() or not prim.IsA(UsdGeom.Cube):raise ValueError('CP09 acquisition finger pad missing')
        if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':raise ValueError('CP09 finger pad hidden during task')
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:raise ValueError('CP09 finger pad collision disabled')
        cube=UsdGeom.Cube(prim)
        ops={str(op.GetOpType()):op.Get() for op in cube.GetOrderedXformOps()}
        if (cube.GetSizeAttr().Get()!=1 or not np.allclose(ops.get('UsdGeom.XformOp.TypeTranslate',[]),row['translate'],rtol=0,atol=1e-8)
            or not np.allclose(ops.get('UsdGeom.XformOp.TypeScale',[]),row['scale'],rtol=0,atol=1e-8)):
            raise ValueError('CP09 acquisition pad transform differs')
        api=PhysxSchema.PhysxCollisionAPI(prim)
        if abs(api.GetContactOffsetAttr().Get()-.0001)>1e-9 or api.GetRestOffsetAttr().Get()!=0:
            raise ValueError('CP09 pad contact/rest offset differs')
        material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')[0]
        physics=UsdPhysics.MaterialAPI(material.GetPrim())
        if (physics.GetStaticFrictionAttr().Get()!=2 or abs(physics.GetDynamicFrictionAttr().Get()-1.8)>1e-6
            or physics.GetRestitutionAttr().Get()!=0):raise ValueError('CP09 acquisition pad material differs')
    return True
