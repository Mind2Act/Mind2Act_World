"""Collision-shape measurements for CP06 grasp and physical deposit evidence."""
import numpy as np


def collider_vertices(stage, body_path):
    from pxr import Usd, UsdGeom, Gf
    cache = UsdGeom.XformCache()
    body = stage.GetPrimAtPath(body_path)
    inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    points = []
    for prim in Usd.PrimRange(body):
        from pxr import UsdPhysics
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if prim.IsA(UsdGeom.Mesh):
            local = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim)
            radius, half = float(shape.GetRadiusAttr().Get()), float(shape.GetHeightAttr().Get())/2
            axis = str(shape.GetAxisAttr().Get())
            local = []
            for z in (-half, half):
                for angle in np.linspace(0, 2*np.pi, 360, endpoint=False):
                    point = [radius*np.cos(angle), radius*np.sin(angle), z]
                    if axis == 'X':
                        point = [point[2], point[0], point[1]]
                    elif axis == 'Y':
                        point = [point[0], point[2], point[1]]
                    local.append(point)
        else:
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeUntransformedBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty():
                continue  # A CollisionAPI-only Xform is not collision geometry.
            lo, hi = bounds.GetMin(), bounds.GetMax()
            local = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        relative = cache.GetLocalToWorldTransform(prim)*inverse
        points.extend(relative.Transform(Gf.Vec3d(*map(float, point))) for point in local)
    if not points:
        raise ValueError('CP06 object has no measurable collision geometry: '+body_path)
    vertices=np.asarray(points)
    if not np.isfinite(vertices).all():raise ValueError('CP06 nonfinite collision geometry: '+body_path)
    return vertices
