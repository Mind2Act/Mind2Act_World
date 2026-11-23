"""USD-side E1 mechanism: camera visibility intervals, station colliders, item dynamics, tray colliders.

Sources (E-142 baseline): engine physco_e2_long_transport.py (intervals, verbatim),
cases/E/runtime/development_world.py (collider/item setup), cases/E/runtime/training_scene.py
(`align_colliders` only). No Carrier alignment of the scene is done here: the prepared scene asset
already carries the aligned stage (done offline by the E1 scene-preparation tool).
"""
import copy
import numpy as np


# Simulation reference masses in grams, keyed by catalog identity and ordered
# dimensions in mm. Use typical finished goods of the represented size.
# Assume empty general boxes, can and cup, a filled juice carton, lightly
# stocked tea/coffee boxes, and an unweighted desktop tape dispenser.
# Chocolate uses hull volume and density (1300 kg/m³, DOI 10.1039/D1FO04049A).
_CARGO_MASS_G = {
    ('Clutter/box/00000', (50, 60.607, 49.774)): 12,
    ('Clutter/box/00001', (50, 52.245, 13.425)): 5,
    ('Clutter/box/00001', (56, 58.515, 15.036)): 7,
    ('Clutter/box/00002', (50, 62.813, 37.033)): 10,
    ('Clutter/calculator/00000', (50, 64.933, 9.428)): 22,
    ('Clutter/can/00000', (50, 50, 21.07)): 5,
    ('Clutter/cup/00000', (50, 62.602, 49.011)): 18,
    ('Clutter/eraser/00000', (31.777, 65, 22.087)): 42,
    ('Clutter/eraser/00000', (50, 102.275, 34.753)): 155,
    ('Clutter/power_bank/00000', (27.289, 65, 6.951)): 18,
    ('Clutter/power_bank/00000', (46.181, 110, 11.762)): 75,
    ('Clutter/stapler/00003', (29.454, 65, 17.918)): 26,
    ('Clutter/stapler/00003', (49.844, 110, 30.323)): 105,
    ('Levels38/magenta_box/00000', (40, 50.25, 29.626)): 6,
    ('Levels38/magenta_box/00000', (50, 62.813, 37.033)): 10,
    ('Rigid/action_camera/00000', (46.339, 65, 35.215)): 115,
    ('Rigid/action_camera/00000', (50, 70.136, 37.998)): 140,
    ('Rigid/action_camera/00001', (47.695, 65, 32.099)): 105,
    ('Rigid/action_camera/00001', (50, 68.142, 33.65)): 120,
    ('Rigid/action_camera/00002', (50, 54.682, 30.069)): 80,
    ('Rigid/alarm/00000', (50, 50, 31.25)): 40,
    ('Rigid/box/00000', (40.574, 65, 37.937)): 9,
    ('Rigid/box/00000', (50, 80.101, 46.75)): 15,
    ('Rigid/cassette/00000', (40.258, 65, 7.622)): 16,
    ('Rigid/cassette/00000', (50, 80.728, 9.467)): 25,
    ('Rigid/charger/00000', (43.282, 65, 27.828)): 55,
    ('Rigid/charger/00000', (50, 75.089, 32.148)): 80,
    ('Rigid/chocolate_bar/00000', (12.529, 65, 7.384)): 6.77,
    ('Rigid/chocolate_bar/00000', (21.203, 110, 12.497)): 32.83,
    ('Rigid/chocolate_bar/00001', (19.604, 65, 8.219)): 9.91,
    ('Rigid/chocolate_bar/00001', (33.176, 110, 13.909)): 48.03,
    ('Rigid/chocolate_bar/00002', (20.266, 65, 7.118)): 9.11,
    ('Rigid/chocolate_bar/00002', (34.296, 110, 12.047)): 44.13,
    ('Rigid/clock/00000', (50, 51.989, 25.355)): 35,
    ('Rigid/correction_tape/00000', (27.267, 65, 10.935)): 12,
    ('Rigid/correction_tape/00000', (46.145, 110, 18.506)): 35,
    ('Rigid/game_machine/00000', (50, 53.844, 19.091)): 35,
    ('Rigid/jewelry_box/00000', (36.433, 65, 19.237)): 22,
    ('Rigid/jewelry_box/00000', (50, 89.205, 26.4)): 45,
    ('Rigid/juice_carton/00000', (24.322, 65.0, 18.446)): 28,
    ('Rigid/juice_carton/00000', (41.16, 110, 31.216)): 135,
    ('Rigid/mouse/00004', (42.701, 65.0, 21.279)): 25,
    ('Rigid/mouse/00004', (50, 76.111, 24.916)): 34,
    ('Rigid/mouse/00006', (35.834, 65, 21.334)): 22,
    ('Rigid/mouse/00006', (50, 90.696, 29.768)): 45,
    ('Rigid/mouse/00007', (36.92, 65, 22.507)): 23,
    ('Rigid/mouse/00007', (50, 88.029, 30.481)): 42,
    ('Rigid/mouse/00008', (44.499, 65, 24.89)): 27,
    ('Rigid/mouse/00008', (50, 73.036, 27.967)): 36,
    ('Rigid/mouse/00009', (50, 64.764, 26.534)): 30,
    ('Rigid/music_player/00000', (46.73, 65, 17.01)): 42,
    ('Rigid/music_player/00000', (50, 69.548, 18.2)): 50,
    ('Rigid/phone/00000', (27.304, 65, 12.679)): 32,
    ('Rigid/phone/00000', (46.207, 110, 21.457)): 120,
    ('Rigid/power_bank/00002', (29.677, 65, 12.402)): 35,
    ('Rigid/power_bank/00002', (50, 109.513, 20.895)): 155,
    ('Rigid/tape_dispense/00000', (32.852, 65, 32.063)): 25,
    ('Rigid/tape_dispense/00000', (50, 98.929, 48.8)): 75,
    ('RoboTwin/107_soap', (20.108, 65, 13.448)): 15,
    ('RoboTwin/107_soap', (34.029, 110, 22.758)): 72,
    ('RoboTwin/112_tea-box', (48.328, 65, 48.248)): 30,
    ('RoboTwin/112_tea-box', (50, 67.249, 49.917)): 33,
    ('RoboTwin/113_coffee-box', (49.705, 65, 44.04)): 48,
    ('RoboTwin/113_coffee-box', (50, 65.386, 44.301)): 49,
}


def _cargo_mass_kg(item):
    dimensions_mm = tuple(round(d * 1000, 3) for d in item.get('dimensions_m') or ())
    return _CARGO_MASS_G.get((item.get('visible_identity'), dimensions_mm), 80) / 1000


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
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    station_mat = UsdShade.Material.Define(stage, '/World/StationPhysicsMaterial')
    station_pm = UsdPhysics.MaterialAPI.Apply(station_mat.GetPrim())
    station_pm.CreateStaticFrictionAttr(.9)
    station_pm.CreateDynamicFrictionAttr(.8)
    station_pm.CreateRestitutionAttr(0.)
    for prefix in ['/World/Conveyor', '/World/CameraStand', '/World/PedestalLeft', '/World/PedestalRight', '/World/Floor']:
        root = stage.GetPrimAtPath(prefix)
        if not root:
            continue
        is_conveyor = prefix == '/World/Conveyor'
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            if is_conveyor:
                if prim.IsA(UsdGeom.Gprim):
                    collision = UsdPhysics.CollisionAPI(prim)
                    if prim.HasAPI(UsdPhysics.CollisionAPI) and collision.GetCollisionEnabledAttr().Get() is not False:
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(station_mat, materialPurpose='physics')
                else:
                    # Shapeless conveyor Xforms (e.g. Belt container) must not carry dangling
                    # CollisionAPI, render-only physics bindings, or orphaned surfaceVelocity.
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    if prim.HasAPI(UsdShade.MaterialBindingAPI):
                        UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding(materialPurpose='physics')
                    sv_enabled = prim.GetAttribute('physxSurfaceVelocity:surfaceVelocityEnabled')
                    sv_vel = prim.GetAttribute('physxSurfaceVelocity:surfaceVelocity')
                    if sv_enabled or sv_vel or 'PhysxSurfaceVelocityAPI' in prim.GetAppliedSchemas():
                        prim.CreateAttribute('physxSurfaceVelocity:surfaceVelocityEnabled', Sdf.ValueTypeNames.Bool).Set(False)
                        prim.CreateAttribute('physxSurfaceVelocity:surfaceVelocity', Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0., 0., 0.))


def make_items_dynamic(stage, items):
    """Set up dynamic cargo bodies, convex-hull colliders and shared physics material."""
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
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr().Set(_cargo_mass_kg(item))
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateSolverPositionIterationCountAttr(32)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateSolverVelocityIterationCountAttr(8)
