"""Apply PhysX API schemas by name so no omni-only PhysxSchema module is needed (CPU tests use usd-core)."""
from pxr import Sdf

INT, FLOAT = Sdf.ValueTypeNames.Int, Sdf.ValueTypeNames.Float


def apply(prim, schema, **attributes):
    prim.AddAppliedSchema(schema)
    for name, (type_name, value) in attributes.items():
        prim.CreateAttribute(name, type_name).Set(value)


def dynamic_body(prim, mass, position_iterations, velocity_iterations):
    from pxr import UsdPhysics
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    apply(prim, 'PhysxRigidBodyAPI',
          **{'physxRigidBody:solverPositionIterationCount': (INT, position_iterations),
             'physxRigidBody:solverVelocityIterationCount': (INT, velocity_iterations)})
    apply(prim, 'PhysxContactReportAPI', **{'physxContactReport:threshold': (FLOAT, 0.0)})


def collision(prim, approximation, contact_offset, **extra):
    from pxr import UsdGeom, UsdPhysics
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    apply(prim, 'PhysxCollisionAPI', **{'physxCollision:contactOffset': (FLOAT, contact_offset)},
          **{('physxCollision:' + k): (FLOAT, v) for k, v in extra.items()})
