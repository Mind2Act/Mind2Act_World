"""M1 stage authoring and environment pose writes (source `cases_unified/M/world.py`, `source_mechanics.py`).

Bodies start kinematic and are moved only by the compiled environment schedule; at the physical transition the
target and its cover become dynamic and are write-locked. Authored mesh collider decomposition is kept (covers are
never convex-filled). Robot mount/cameras/drives/rest pose belong to robot/.
"""
import numpy as np


def quat_rotate(q, v):
    w, x, y, z = (float(a) for a in q)
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def body_center(position, quaternion_wxyz, offset_local):
    return np.asarray(position, float) + quat_rotate(quaternion_wxyz, offset_local)


class EnvironmentWriter:
    """Environment-only pose writes to kinematic bodies; refuses writes to locked (dynamic) bodies."""

    ALLOWED = {'environment_close', 'environment_swap'}

    def __init__(self):
        self.locked, self.log = set(), []

    def lock(self, paths):
        self.locked.update(paths)

    def write(self, tick, phase, body, path, pose):
        if any(under(path, x) or under(x, path) for x in self.locked):
            raise RuntimeError('pose write prohibited after physical transition: ' + path)
        if phase not in self.ALLOWED:
            raise RuntimeError('undisclosed environment pose-write phase: ' + str(phase))
        from pxr import Gf, UsdGeom
        import omni.usd
        position, orientation = pose[:3], pose[3:]
        xf = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(path))
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*map(float, position)))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                q = Gf.Quatd(float(orientation[0]), *map(float, orientation[1:]))
                op.Set(q if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf(q))
        body.set_world_pose(position=np.asarray(position), orientation=np.asarray(orientation))
        self.log.append(dict(tick=tick, path=path, phase=phase))


def under(path, root):
    return path == root or path.startswith(root + '/')


def author_stage(stage, M, lay):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    # SDK-owned live RenderProducts share /Render with source opinions.
    stand = stage.GetPrimAtPath('/World/CameraStand')
    if stand:  # inherited workstation texture is unavailable: constant visual only, mesh/physics untouched
        visual = UsdShade.Material.Define(stage, '/World/MRebuiltStandVisual')
        shader = UsdShade.Shader.Define(stage, '/World/MRebuiltStandVisual/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(tuple(lay['stand_visual']['diffuse_rgb']))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(lay['stand_visual']['roughness'])
        visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(stand).Bind(visual, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    cm = lay['contact_material']
    mat = UsdShade.Material.Define(stage, '/World/M1PhysicsMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(cm['static_friction'])
    pm.CreateDynamicFrictionAttr(cm['dynamic_friction'])
    pm.CreateRestitutionAttr(cm['restitution'])
    for spec in M['bodies'].values():
        prim = stage.GetPrimAtPath(spec['prim_path'])
        if not prim:
            raise ValueError('missing M body prim: ' + spec['prim_path'])
        collision = spec.get('collision_prims', [spec.get('support_prim')])
        for child in Usd.PrimRange(prim):
            if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if str(child.GetPath()) in collision:  # preserve authored mesh decomposition
                UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                api = PhysxSchema.PhysxCollisionAPI.Apply(child)
                api.CreateContactOffsetAttr(lay['body_contact_offset_m'])
                api.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(child).Bind(mat, materialPurpose='physics')
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec['mass_kg']))
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(lay['body_solver_iterations'])
    roots = M.get('static_collision_roots', [])
    if not roots:
        raise ValueError('scene contract must explicitly provide collision-active static station roots')
    for path in roots:
        root = stage.GetPrimAtPath(path)
        if not root:
            raise ValueError('missing static root: ' + path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    ready = M.get('replay_ready_prim')
    if ready and stage.GetPrimAtPath(ready):
        UsdGeom.Imageable(stage.GetPrimAtPath(ready)).MakeInvisible()


def runtime_collision(stage, M):
    from pxr import Usd, UsdPhysics
    expected = {k: set(s.get('collision_prims', [s.get('support_prim')])) for k, s in M['bodies'].items()}
    actual = {k: {str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(s['prim_path']))
                  if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()}
              for k, s in M['bodies'].items()}
    if actual != expected:
        raise ValueError('M collider decomposition differs from the contract')


def set_dynamic(stage, path):
    from pxr import UsdPhysics
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr(False)


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
                         min((float(p.separation) for p in points), default=1.),
                         normals=[tuple(float(x) for x in p.normal) for p in points])
    return get_physx_simulation_interface().subscribe_contact_report_events(report)
