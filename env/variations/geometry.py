"""Native shape clearance, separate from advisory source exclusion rectangles.

PhysX queries use the real target collision shapes. A mesh query uses its convex
hull, so only a source explicitly authored as convexHull can establish a fatal
intersection. Other mesh hits remain explicit candidates for visual review.
No stage opinions, solver settings or physics steps are changed here.
"""


def _under(path, root):
    return path == root or path.startswith(root + "/")


def _query_shape(path):
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf
    hits = []

    def report(hit):
        hits.append(str(hit.collision))
        return True

    encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(path))
    get_physx_scene_query_interface().overlap_shape(*encoded, report, False)
    return hits


def _exact_query(prim, root):
    from pxr import UsdGeom, UsdPhysics
    if not prim.IsA(UsdGeom.Mesh):
        return True
    owner = prim
    while owner and _under(str(owner.GetPath()), str(root.GetPath())):
        if owner.HasAPI(UsdPhysics.MeshCollisionAPI):
            return UsdPhysics.MeshCollisionAPI(owner).GetApproximationAttr().Get() == "convexHull"
        owner = owner.GetParent()
    return False


def _enabled_collider(prim, root):
    from pxr import UsdPhysics
    owner = prim
    while owner and _under(str(owner.GetPath()), str(root.GetPath())):
        if owner.HasAPI(UsdPhysics.CollisionAPI):
            return UsdPhysics.CollisionAPI(owner).GetCollisionEnabledAttr().Get() is not False
        owner = owner.GetParent()
    return False


def verify_geometry(stage, targets):
    """Check all clutter shapes against task/mechanism colliders, across tasks.

    Support surfaces are excluded by the builder's explicit capabilities. Task
    roots are not guessed from names: native hits outside support/clutter count,
    including mechanisms that do not live under /World/Task.
    """
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(targets.clutter_root)
    rows, candidates = [], []
    if root:
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim) or not _enabled_collider(prim, root):
                continue
            path = str(prim.GetPath())
            exact = _exact_query(prim, root)
            hits = sorted(set(_query_shape(path)))
            relevant = [p for p in hits if not any(_under(p, r) for r in
                        (targets.clutter_root, *targets.table_surfaces, *targets.floor_surfaces))]
            if relevant and exact:
                raise ValueError("distractor intersects task/mechanism collision geometry: "
                                 + path + " -> " + ", ".join(relevant))
            if relevant:
                candidates.append({"path": path, "collisions": relevant,
                                   "reason": "convex query can overestimate this mesh; inspect actual geometry"})
            rows.append({"path": path, "query_matches_source_collision": exact,
                         "collision_hits": hits})
    return {"status": "requires_visual_review" if candidates else "verified",
            "readback": "native_physx_overlap_shape", "shapes": rows,
            "unconfirmed_mesh_overlaps": candidates, "occlusion_verified": False,
            "physics_steps_added": 0}
