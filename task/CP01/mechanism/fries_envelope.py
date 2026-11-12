"""CP01 task contact candidate; visual meshes and shared robot stay unchanged."""
import math

def replace_welded_fries_contact(stage, roots, native_fries_mass_properties):
    """Replace 14 welded wall/rod shapes by their convex exterior envelope.

    Preserve baseline native mass, COM and body-frame inertia. This closes
    artificial interior hook gaps without enlarging exterior AABB or creating
    an attachment/contact filter. Only fries are changed in this candidate.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
    props = native_fries_mass_properties
    mass = float(props['mass_kg'])
    com = np.asarray(props['com_local_xyz'], float)
    q = np.asarray(props['principal_axes_xyzw'], float)
    diagonal = np.asarray(props['diagonal_inertia_kg_m2'], float)
    if not np.isfinite(np.r_[mass,com,q,diagonal]).all() or mass <= 0 or np.any(diagonal <= 0):
        raise ValueError('Invalid measured baseline mass properties')
    if abs(float(q@q)-1.) > 1e-5: raise ValueError('Baseline principal axes not normalized')
    cache = UsdGeom.XformCache()
    changes = []
    for root in roots:
        actor = stage.GetPrimAtPath(root)
        if not actor.GetChild('Fry20'): continue
        colliders = [p for p in Usd.PrimRange(actor) if p.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        if len(colliders) != 14 or not all(p.IsA(UsdGeom.Mesh) for p in colliders):
            raise ValueError('Unexpected baseline fries topology at '+root)
        inverse = cache.GetLocalToWorldTransform(actor).GetInverse()
        points = []
        for prim in colliders:
            transform = cache.GetLocalToWorldTransform(prim) * inverse
            points.extend(list(transform.Transform(Gf.Vec3d(*v))) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        points = np.unique(np.asarray(points,float),axis=0)
        hull = ConvexHull(points)
        vertex_ids = sorted(set(int(i) for f in hull.simplices for i in f))
        if len(vertex_ids) > 255: raise ValueError('Envelope exceeds bounded hull vertex limit')
        vertices = points[vertex_ids]
        remap = {v:i for i,v in enumerate(vertex_ids)}
        center = vertices.mean(axis=0)
        faces = []
        for f in hull.simplices:
            a,b,c=points[f]
            indices=[remap[int(i)] for i in f]
            if np.dot(np.cross(b-a,c-a),(a+b+c)/3-center) < 0: indices.reverse()
            faces.append(indices)
        if np.max(np.abs(vertices.min(axis=0)-points.min(axis=0))) > 1e-7 or np.max(np.abs(vertices.max(axis=0)-points.max(axis=0))) > 1e-7:
            raise ValueError('Envelope exterior bounds differ')
        path = root+'/PackagedFoodContact'
        if stage.GetPrimAtPath(path): raise ValueError('Candidate contact exists already')
        mesh = UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdGeom.Imageable(mesh).CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(255)
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        api.CreateContactOffsetAttr(.002);api.CreateRestOffsetAttr(0.)
        material = UsdShade.Material(stage.GetPrimAtPath('/World/TrayGripMaterial'))
        if not material: raise ValueError('Task physics material missing')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
        for prim in colliders: prim.RemoveAPI(UsdPhysics.CollisionAPI)
        mass_api = UsdPhysics.MassAPI.Apply(actor)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        changes.append({'root':root,'repair':'welded_fries_continuous_exterior_contact',
                        'old_collider_paths':[str(p.GetPath()) for p in colliders],
                        'collision_path':path,'vertices':len(vertices),'faces':len(faces),
                        'local_min_m':vertices.min(axis=0).tolist(),'local_max_m':vertices.max(axis=0).tolist(),
                        'baseline_native_mass_COM_inertia_preserved':True,
                        'visual_meshes_transforms_and_materials_unchanged':True,
                        'no_attachment_or_contact_pair_filter':True})
    return changes
