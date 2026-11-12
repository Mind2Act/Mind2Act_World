"""Read-only checks of the selected prepared Memory namespace and joint relationships."""

def check_stage(stage, count):
    from pxr import UsdGeom, UsdPhysics
    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('M2 selected scene requires source metres / Z-up')
    for i in range(count):
        for kind, root, joint_root, schema, axis in (
            ('Card', 'Cards', 'Joints', UsdPhysics.RevoluteJoint, 'Y'),
            ('Key', 'Buttons', 'ButtonJoints', UsdPhysics.PrismaticJoint, 'Z')):
            body_path = f'/World/Memory/{root}/{kind}{i}'
            prim = stage.GetPrimAtPath(body_path)
            joint_path = f'/World/Memory/{joint_root}/{kind}{i}'
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('M2 source rigid body missing: '+body_path)
            if not joint_prim or not joint_prim.IsA(schema):
                raise ValueError('M2 source joint schema missing: '+joint_path)
            joint = schema(joint_prim)
            if (joint.GetAxisAttr().Get() != axis or joint.GetBody0Rel().GetTargets()
                    or [str(path) for path in joint.GetBody1Rel().GetTargets()] != [body_path]):
                raise ValueError('M2 source joint axis/body relationship mismatch: '+joint_path)
        if not stage.GetPrimAtPath(f'/World/Memory/Paired{i}'):
            raise ValueError('M2 source matched marker missing')
    return dict(schema='m2.prepared-structure.v1', cards=count, buttons=count,
                card_axis='Y', button_axis='Z', body0='world', units='metres', up='Z')
