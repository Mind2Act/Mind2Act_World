"""Validate baked rear geometry; loading never extends an asset again."""
from pxr import Gf,UsdPhysics
REAR_EXTENSION_M=.035
ROOT='/World/CP02V1'

def verify_rear_extension(stage):
 box=stage.GetPrimAtPath(ROOT+'/Box')
 if not box or box.GetCustomDataByKey('cp02RearExtensionM') != REAR_EXTENSION_M:
  raise ValueError('CP02 requires the published 35 mm rear extension')
 for path,depth in [('/Box/Wall0',.190),('/Box/Wall1',.190),('/Lid/Panel',.183),('/Lid/Skirt0',.183),('/Lid/Skirt1',.183)]:
  prim=stage.GetPrimAtPath(ROOT+path)
  if not prim or abs(prim.GetAttribute('xformOp:scale').Get()[1]-depth)>1e-7:
   raise ValueError('CP02 rear geometry differs: '+path)
 return dict(extension_m=REAR_EXTENSION_M,baked_into_asset=True,applied_during_loading=False,native_qualified=False)

def update_lid_mass_properties(stage, delta=REAR_EXTENSION_M):
    """Preserve authored mass; update plate-dominant distribution for new depth."""
    lid = stage.GetPrimAtPath(ROOT + '/Lid')
    mass = UsdPhysics.MassAPI(lid)
    old_center = Gf.Vec3f(0, -0.074, 0.003)
    old_inertia = Gf.Vec3f(0.00011, 0.00063, 0.00073)
    stretch = (0.148 + delta) / 0.148
    second_y = (old_inertia[0] + old_inertia[2] - old_inertia[1]) / 2
    increment = (stretch * stretch - 1) * second_y
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(old_center[0], old_center[1] - delta / 2, old_center[2]))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(old_inertia[0] + increment, old_inertia[1], old_inertia[2] + increment))
