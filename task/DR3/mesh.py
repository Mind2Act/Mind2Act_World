# Source: fr3-replay-new15-20261031 / dual_geometry.py; see config/source.json.
"""Physical collision mesh versus the finite closed-door swept aperture.

A long rotated link's axis-aligned box can cover empty space. Narrow-phase
triangle clipping avoids calling that empty space an arm caught in the door.
"""
import numpy as np


def collider_vertices(stage, collider, rigid_body):
    """Read authored collision meshes, including hidden guide children of Xforms.

    Render visibility/purpose cannot decide collision coverage. The selected X5
    places CollisionAPI on node_STL_BINARY_ and its points on a Mesh child.
    """
    from pxr import Usd, UsdGeom
    cache=UsdGeom.XformCache()
    inverse=cache.GetLocalToWorldTransform(rigid_body).GetInverse()
    parts=[]
    for prim in Usd.PrimRange(collider):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('invalid collision mesh geometry')
        transform=np.asarray(cache.GetLocalToWorldTransform(prim)*inverse)
        parts.append((np.c_[points,np.ones(len(points))]@transform)[:,:3])
    if not parts:
        raise ValueError('missing authored collision mesh geometry')
    return np.concatenate(parts)


def collider_convex_parts(stage, collider, rigid_body):
    """Read explicitly authored per-mesh convexHull collision shapes.

    This reads MeshCollisionAPI, including the official URDF Xform parent;
    it never changes the USD or declares a body-wide hull to be an exact shape.
    Unknown approximation types remain unavailable rather than being guessed.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    parts=[]
    for mesh in Usd.PrimRange(collider):
        if not mesh.IsA(UsdGeom.Mesh):continue
        owner=mesh
        while owner and not owner.HasAPI(UsdPhysics.MeshCollisionAPI) and owner!=collider:
            owner=owner.GetParent()
        approximation=UsdPhysics.MeshCollisionAPI(owner).GetApproximationAttr().Get()
        if approximation!='convexHull':return None
        points=collider_vertices(stage,mesh,rigid_body)
        parts.append(dict(vertices=points[ConvexHull(points).vertices].tolist(),
                          approximation=approximation,source_path=str(mesh.GetPath()),
                          approximation_source_path=str(owner.GetPath())))
    return parts or None


def intersects_box(triangles, low, high):
    low, high = np.asarray(low), np.asarray(high)
    candidates = triangles[(triangles.max(axis=1) >= low).all(axis=1)
                           & (triangles.min(axis=1) <= high).all(axis=1)]
    for triangle in candidates:
        polygon = list(triangle)
        for axis in range(3):
            for bound, keep_above in ((low[axis], True), (high[axis], False)):
                if not polygon:
                    break
                clipped = []
                previous = polygon[-1]
                previous_in = previous[axis] >= bound if keep_above else previous[axis] <= bound
                for current in polygon:
                    current_in = current[axis] >= bound if keep_above else current[axis] <= bound
                    if current_in != previous_in:
                        fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                        clipped.append(previous + fraction * (current - previous))
                    if current_in:
                        clipped.append(current)
                    previous, previous_in = current, current_in
                polygon = clipped
        if polygon:
            return True
    return False


def collision_mesh(stage, link):
    """Freeze collision vertices in rigid-link coordinates, not world space."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(link).GetInverse()
    all_triangles = []
    for prim in Usd.PrimRange(link):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
        vertices = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if approximation == 'convexHull':
            faces = ConvexHull(vertices).simplices
        else:
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            faces, offset = [], 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                for i in range(1, count - 1):
                    faces.append((indices[offset], indices[offset+i], indices[offset+i+1]))
                offset += count
            faces = np.asarray(faces)
        if len(faces):
            all_triangles.append(vertices[faces])
    return np.concatenate(all_triangles) if all_triangles else np.empty((0, 3, 3))
