"""DR2 spring-button panel mechanics (source `cases_unified/FR2/world.py`, b-FR2-work-20261025-v3).

Only the panel is authored here: three articulated spring buttons anchored at their authored pose,
colliders for static props, ring materials for the cue light. Robot mount, camera joints, drives and
rest pose are owned by robot/ (`restore_asset`, `attach_cameras`, `initialize`).
"""
from dataclasses import dataclass


@dataclass
class Panel:
    buttons: dict
    ring_paths: list


def author_panel(stage, lay):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
    for i in range(lay['buttons']):
        UsdGeom.Imageable(stage.GetPrimAtPath('/World/CallPanel/Label' + str(i))).MakeInvisible()
        root = '/World/CallPanel/Button' + str(i) + '/Asset'
        asset = stage.GetPrimAtPath(root)
        UsdPhysics.ArticulationRootAPI.Apply(asset)
        for child in Usd.PrimRange(asset):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(child)
                body.CreateRigidBodyEnabledAttr(True)
                body.CreateKinematicEnabledAttr(False)
                PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.)
            if child.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(child).CreateJointEnabledAttr(True)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(True)
                api = PhysxSchema.PhysxCollisionAPI.Apply(child)
                api.CreateContactOffsetAttr(lay['button_contact_offset_m'])
                api.CreateRestOffsetAttr(0.)
        # Anchor in the world at the authored pose, preserving the tilted enclosure.
        fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root + '/E_body_16/FixedJoint'))
        mat = UsdGeom.Xformable(stage.GetPrimAtPath(root + '/E_body_16')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        fixed.CreateBody0Rel().SetTargets([])
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
        fixed.CreateLocalRot1Attr(Gf.Quatf(1))
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*mat.ExtractTranslation()))
        rotation = mat.ExtractRotationQuat()
        fixed.CreateLocalRot0Attr(Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary())))
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(root + '/E_button_01_13/PrismaticJoint_orangeswitch_up'), 'linear')
        drive.CreateTargetPositionAttr(lay['button_drive_target_m'])
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith('/World/') or any(t in path for t in ['/Robot', '/Button', '/Looks', '/Ring', '/Label', '/Socket']):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    return ['/World/CallPanel/Ring' + str(i) for i in range(lay['buttons'])]


def set_rings(stage, ring_paths, lit_button, lay):
    from pxr import UsdShade
    for i, path in enumerate(ring_paths):
        material = lay['lit_material'] if i == lit_button else lay['unlit_material']
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(UsdShade.Material(stage.GetPrimAtPath(material)))


def subscribe_contacts(table):
    import math
    from pxr import PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType

    def report(headers, data):
        for h in headers:
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
            points = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
            table.report(pair, h.type != ContactEventType.CONTACT_LOST,
                         max((math.hypot(*p.impulse) for p in points), default=0.),
                         min((float(p.separation) for p in points), default=1.))
    return get_physx_simulation_interface().subscribe_contact_report_events(report)
