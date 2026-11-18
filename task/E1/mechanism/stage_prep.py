"""USD-side E1 mechanism: camera visibility intervals, station colliders, item dynamics, tray colliders.

Sources (E-142 baseline): engine physco_e2_long_transport.py (intervals, verbatim),
cases/E/runtime/development_world.py (collider/item setup), cases/E/runtime/training_scene.py
(`align_colliders` only). No Carrier alignment of the scene is done here: the prepared scene asset
already carries the aligned stage (done offline by the E1 scene-preparation tool).
"""
import copy
import numpy as np


def scene_visible_intervals(case, frustum, batch_prims):
    """Bound actual rendered mesh pieces, without the old 18 cm proxy/padding.

    Each mesh retains its oriented world bound. Binary search locates the
    entrance/exit to 0.01 mm; unioning mesh bounds remains conservative.
    """
    from pxr import Gf, Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    centers = [0.] * len(batch_prims)
    intervals = []
    for batch, local_center in zip(batch_prims, centers):
        bounds = []
        for prim in Usd.PrimRange(batch.GetPrim()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            bounds.append(cache.ComputeWorldBound(prim))
        assert bounds, 'Batch must contain visible source meshes'
        def visible(center):
            for bound in bounds:
                matrix = Gf.Matrix4d(bound.GetMatrix())
                matrix.SetTranslateOnly(matrix.ExtractTranslation() + Gf.Vec3d(center-local_center, 0, 0))
                if frustum.Intersects(Gf.BBox3d(bound.GetRange(), matrix)):
                    return True
            return False
        assert visible(local_center) and not visible(-4.) and not visible(4.)
        lo, hi = -4., local_center
        while hi-lo > .00001:
            mid = (lo+hi)/2
            if visible(mid): hi = mid
            else: lo = mid
        entrance = lo
        lo, hi = local_center, 4.
        while hi-lo > .00001:
            mid = (lo+hi)/2
            if visible(mid): lo = mid
            else: hi = mid
        intervals.append((entrance, hi))
    return intervals


def compute_visible_intervals(stage, manifest):
    from pxr import Gf, Usd, UsdGeom
    layout = manifest['case']['layout']
    xforms = [UsdGeom.Xform(stage.GetPrimAtPath('/World/Task/Batch' + str(i))) for i in range(len(manifest['rounds']))]
    originals = [b.GetOrderedXformOps()[0].Get() for b in xforms]
    for b in xforms:
        b.GetOrderedXformOps()[0].Set(Gf.Vec3d(0, layout['carrier_center'][1], layout['carrier_center'][2]))
        UsdGeom.Imageable(b).MakeVisible()
    bound_case = copy.deepcopy(manifest['case'])
    bound_case['layout']['source_center_xy'][0] = 0
    bound_case['layout']['carrier_center'][0] = 0
    frustum = UsdGeom.Camera(stage.GetPrimAtPath('/World/HeadCamera')).GetCamera(Usd.TimeCode.Default()).frustum
    intervals = scene_visible_intervals(bound_case, frustum, xforms)
    for b, t in zip(xforms, originals):
        b.GetOrderedXformOps()[0].Set(t)
    return intervals


def _translate_op(stage, path):
    from pxr import UsdGeom
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()
    return next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)


def align_colliders(stage, receipt):
    """First-batch transport colliders as in training: tray only, lowered."""
    from pxr import Gf
    receipt = dict(receipt)
    root = stage.GetPrimAtPath("/World/Carriers/Batch0/Collision")
    if not root:
        raise ValueError("E first-batch transport colliders are missing")
    shift = receipt["observation_tray_shift_m"]
    removed, lowered = [], []
    for prim in list(root.GetChildren()):
        name = prim.GetName()
        if name == "carrier" or name.startswith(("submit", "obstacle_item_")):
            removed.append(str(prim.GetPath()))
            stage.RemovePrim(prim.GetPath())
            continue
        if not name.startswith("source"):
            raise ValueError("Unexpected E first-batch collider: " + name)
        op = _translate_op(stage, str(prim.GetPath()))
        value = op.Get()
        op.Set(Gf.Vec3d(value[0], value[1], value[2] - shift))
        lowered.append(str(prim.GetPath()))
    if not lowered or not any(path.endswith("/carrier") for path in removed):
        raise ValueError("E first-batch transport colliders differ from the expected layout")
    receipt.update(colliders_aligned=True, removed_colliders=removed, lowered_colliders=lowered)
    return receipt


def make_station_static(stage):
    """Station meshes become static colliders; they catch collisions the planner does not model."""
    from pxr import Usd, UsdGeom, UsdPhysics
    for prefix in ['/World/Conveyor', '/World/CameraStand', '/World/PedestalLeft', '/World/PedestalRight', '/World/Floor']:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')


def make_items_dynamic(stage, items):
    """Item meshes: convex-hull colliders, dynamic rigid bodies, 0.08 kg, shared physics material."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    mat = UsdShade.Material.Define(stage, '/World/PhysicsMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(.9)
    pm.CreateDynamicFrictionAttr(.8)
    pm.CreateRestitutionAttr(0.)
    for item in items:
        root = stage.GetPrimAtPath(item['prim_path'])
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdGeom.Mesh):
                is_collision = 'collision' in str(prim.GetPath()).lower()
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(not is_collision)
                if not is_collision:
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    contact.CreateContactOffsetAttr(.001)
                    contact.CreateRestOffsetAttr(0.)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')
        rb = UsdPhysics.RigidBodyAPI.Apply(root)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.08)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateSolverPositionIterationCountAttr(32)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateSolverVelocityIterationCountAttr(8)
