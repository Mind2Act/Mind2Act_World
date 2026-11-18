"""PhysX authoring shared by the precision fixtures (baseline cases_unified/PR1/world.py generic precision branch)."""
from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

INT, FLOAT = Sdf.ValueTypeNames.Int, Sdf.ValueTypeNames.Float
# Names that stay visual-only (no collider): printed scale, markers, screw slots.
DECORATIVE = ('Scale', 'Label', 'IndexLine', 'IndexTip', 'IndexArrow', 'Artwork', 'DialPrint', 'Slot')


def physx(prim, schema, **attributes):
    """Apply a PhysX API schema by name so no omni-only PhysxSchema module is needed (CPU tests use usd-core)."""
    prim.AddAppliedSchema(schema)
    for name, (type_name, value) in attributes.items():
        prim.CreateAttribute(name, type_name).Set(value)


def make_dynamic(body, p):
    rigid = UsdPhysics.RigidBodyAPI.Apply(body)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(p['body_mass_kg'])
    physx(body, 'PhysxRigidBodyAPI',
          **{'physxRigidBody:solverPositionIterationCount': (INT, p['body_solver_position_iterations']),
             'physxRigidBody:solverVelocityIterationCount': (INT, p['body_solver_velocity_iterations'])})
    physx(body, 'PhysxContactReportAPI', **{'physxContactReport:threshold': (FLOAT, 0.0)})


def contact_material(stage, path, p):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(p['static_friction'])
    api.CreateDynamicFrictionAttr(p['dynamic_friction'])
    api.CreateRestitutionAttr(p['restitution'])
    return mat


def add_colliders(stage, root, body_path, mat, p):
    """Collider on every non-decorative gprim under root; convex hull inside the body, triangle mesh elsewhere."""
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not (path.startswith(root) and prim.IsA(UsdGeom.Gprim)):
            continue
        if (prim.IsA(UsdGeom.BasisCurves) or prim.GetName().startswith('ResistanceWinding')
                or any(k in prim.GetName() for k in DECORATIVE)):
            continue
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                'convexHull' if path.startswith(body_path + '/') else 'none')
        physx(prim, 'PhysxCollisionAPI', **{'physxCollision:contactOffset': (FLOAT, p['contact_offset_m']),
                                            'physxCollision:restOffset': (FLOAT, p['rest_offset_m'])})
        if path.startswith(body_path + '/'):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')
    colliders = [str(q.GetPath()) for q in stage.Traverse()
                 if str(q.GetPath()).startswith(body_path + '/') and q.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        raise RuntimeError('task body has no collision geometry')
    return colliders
