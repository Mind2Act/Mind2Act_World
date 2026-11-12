"""Passive spring nib: a small body sliding on the marker's X axis, driven only by a fixed spring and contacts.

Ported from the baseline (source_id pr3-original-source-mainline-pro-sourcepath:
cases_unified/PR3/source/mind2act_pr3_passive_nib.py). No action, target-path or policy access.
"""
from . import physx

NIB_SPEC = {
    "extension_m": 0.003,
    "compression_m": 0.003,
    "radius_m": 0.0025,
    "mass_kg": 0.001,
    "spring_N_per_m": 40.0,
    "damping_Ns_per_m": 0.2,
    "max_spring_force_N": 0.3,
}
ROOT = "/World/MarkerNib"
SPRING = "/World/MarkerNibSpring"


def install_passive_nib(stage, marker_center, contact_offset):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    extension = NIB_SPEC["extension_m"]
    body = UsdGeom.Xform.Define(stage, ROOT)
    body.AddTranslateOp().Set(Gf.Vec3d(float(marker_center[0]) - .084 - extension,
                                       float(marker_center[1]), float(marker_center[2])))
    physx.dynamic_body(body.GetPrim(), NIB_SPEC["mass_kg"], 64, 16)
    # Reuse the felt material; replace the old static visual and collider.
    old_visual = stage.GetPrimAtPath("/World/MarkerDynamic/Visuals/RoundFeltNib")
    felt, _ = UsdShade.MaterialBindingAPI(old_visual).ComputeBoundMaterial()
    if not felt:
        raise ValueError('prepared scene: RoundFeltNib has no bound material')
    UsdGeom.Imageable(old_visual).MakeInvisible()
    stage.RemovePrim("/World/MarkerDynamic/Colliders/Tip")
    sphere = UsdGeom.Sphere.Define(stage, ROOT + "/Tip")
    sphere.CreateRadiusAttr(NIB_SPEC["radius_m"])
    physx.collision(sphere.GetPrim(), "convexHull", contact_offset)
    shaft = UsdGeom.Cylinder.Define(stage, ROOT + "/FeltShaft")
    shaft.CreateAxisAttr("X")
    shaft.CreateRadiusAttr(NIB_SPEC["radius_m"])
    shaft.CreateHeightAttr(.008)
    shaft.AddTranslateOp().Set(Gf.Vec3d(.004, 0., 0.))
    for visual in (sphere, shaft):
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(felt)
    joint = UsdPhysics.PrismaticJoint.Define(stage, SPRING)
    joint.CreateBody0Rel().SetTargets(["/World/MarkerDynamic"])
    joint.CreateBody1Rel().SetTargets([ROOT])
    joint.CreateLocalPos0Attr(Gf.Vec3f(-.084, 0., 0.))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-extension)
    joint.CreateUpperLimitAttr(NIB_SPEC["compression_m"])
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(-extension)
    drive.CreateStiffnessAttr(NIB_SPEC["spring_N_per_m"])
    drive.CreateDampingAttr(NIB_SPEC["damping_Ns_per_m"])
    drive.CreateMaxForceAttr(NIB_SPEC["max_spring_force_N"])
    return ROOT, str(sphere.GetPath())
