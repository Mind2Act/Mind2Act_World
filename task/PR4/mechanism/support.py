"""Conservative geometric support witnesses from actual source collision faces.

This is not a FEM force/contact-report reconstruction. The authored rest offsets
bound the effective surfaces; contactOffset is deliberately never used. Missing
native collision provenance is an evidence failure, not a model score of zero.
"""
import numpy as np

SCHEMA = 'mind2act_eval.pr4_geometric_support.v1'
REST_SOURCE = 'authored_physxCollision:restOffset'
GEOMETRY_SOURCE = 'composed_native_collision_mesh'


def source_support_paths(origin):
    baked = [v['mesh'] for v in origin.get('rigid_fixture_bake', [])
             if v.get('mesh', '').startswith('/World/PR4/Support/')]
    paths = baked or [origin.get('panel_prim')]
    if not all(isinstance(p, str) and p.startswith('/World/') for p in paths):
        raise ValueError('PR4 missing source support geometry identity')
    return paths


def validate_support(value, origin=None):
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('PR4 missing native support geometry proof')
    if value.get('pair_filter_check') != 'no_source_pair_exclusion':
        raise ValueError('PR4 missing native support collision-pair proof')
    cloth = value.get('cloth', {})
    if cloth.get('path') != '/World/DustCloth':
        raise ValueError('PR4 support geometry cloth identity differs')
    surfaces = value.get('surfaces')
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError('PR4 missing native support collision surfaces')
    if origin is not None and {b.get('path') for b in surfaces} != set(source_support_paths(origin)):
        raise ValueError('PR4 native support geometry source identity differs')
    for body in [cloth] + surfaces:
        offset = body.get('rest_offset_m')
        if (body.get('collision_enabled') is not True or body.get('rest_offset_source') != REST_SOURCE
                or type(offset) not in (float, int) or not np.isfinite(offset) or offset < 0):
            raise ValueError('PR4 missing enabled collision/actual rest-offset support proof')
    for body in surfaces:
        if (body.get('static') is not True or body.get('geometry_source') != GEOMETRY_SOURCE
                or not isinstance(body.get('path'), str) or not body['path'].startswith('/World/')
                or body.get('approximation') not in ('none', 'convexHull')):
            raise ValueError('PR4 missing actual static support geometry proof')
        points, triangles = np.asarray(body.get('points'), float), np.asarray(body.get('triangles'))
        if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all()
                or triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles)
                or triangles.dtype.kind not in 'iu' or triangles.min() < 0 or triangles.max() >= len(points)):
            raise ValueError('PR4 invalid measured support collision topology')
    return value


def support_witness(points, geometry):
    """Find an upward actual face intersecting the cloth's effective shell.

    Signed distance is bounded on both sides by actual rigid rest +/- cloth
    rest. Thus neither speculative-contact gaps nor nodes deep under the table
    qualify. There is no invented distance, support fraction, or force gate.
    """
    validate_support(geometry)
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('PR4 invalid native FEM support points')
    cloth_rest = geometry['cloth']['rest_offset_m']
    for body in geometry['surfaces']:
        vertices = np.asarray(body['points'], float)
        for triangle_id, indices in enumerate(body['triangles']):
            a, b, c = vertices[indices]
            ab, ac = b-a, c-a
            normal = np.cross(ab, ac)
            norm = np.linalg.norm(normal)
            if norm == 0 or normal[2] <= 0:
                continue
            normal /= norm
            distance = (points-a) @ normal
            rigid_rest = body['rest_offset_m']
            candidates = np.flatnonzero((distance >= rigid_rest-cloth_rest) & (distance <= rigid_rest+cloth_rest))
            if not len(candidates):
                continue
            projected = points[candidates] - distance[candidates, None]*normal - a
            aa, bb, cc = ab@ab, ab@ac, ac@ac
            denominator = aa*cc-bb*bb
            u = ((projected@ab)*cc-(projected@ac)*bb)/denominator
            v = ((projected@ac)*aa-(projected@ab)*bb)/denominator
            inside = (u >= 0) & (v >= 0) & (u+v <= 1)
            if inside.any():
                node = int(candidates[np.flatnonzero(inside)[0]])
                return dict(supported=True, method='native_collision_geometry_rest_surface_intersection',
                            support_path=body['path'], triangle_id=triangle_id, node_id=node,
                            signed_face_distance_m=float(distance[node]), cloth_rest_offset_m=cloth_rest,
                            rigid_rest_offset_m=rigid_rest, fem_contact_force_claimed=False)
    return dict(supported=False, method='native_collision_geometry_rest_surface_intersection',
                fem_contact_force_claimed=False)


def _composition_errors(stage):
    """Read real Pcp errors on both current USD and Isaac's USD 24.5.

    This follows the existing asset_relocation prim-index fallback. Layer
    stacks are also inspected: missing sublayers need not be prim-index errors.
    Missing introspection APIs propagate; they never mean an empty error list.
    """
    if hasattr(stage, 'GetCompositionErrors'):
        return list(stage.GetCompositionErrors())
    from pxr import Usd
    prims = [stage.GetPseudoRoot(), *stage.TraverseAll()]
    for prototype in stage.GetPrototypes():
        prims.extend(Usd.PrimRange(prototype))
    errors = [error for prim in prims for error in prim.GetPrimIndex().localErrors]
    cache = stage._GetPcpCache()
    stacks = [cache.layerStack]
    for layer in cache.GetUsedLayers():
        stacks.extend(cache.FindAllLayerStacksUsingLayer(layer))
    for stack in stacks:
        errors.extend(stack.localErrors)
    return list({str(error): error for error in errors}.values())


def capture_support(stage, origin):
    """Read composed source triangles/offsets without modifying the USD stage."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cloth = stage.GetPrimAtPath(origin['body_path'])
    paths = source_support_paths(origin)
    surfaces = [stage.GetPrimAtPath(path) for path in paths]
    bodies = [cloth] + surfaces
    if not all(bodies):
        raise ValueError('PR4 missing composed native support geometry')
    errors = [str(e) for e in _composition_errors(stage)]
    if any(path in error for path in paths+[origin['body_path']] for error in errors):
        raise ValueError('PR4 unresolved native support geometry composition')

    def ancestors(prim):
        while prim and not prim.IsPseudoRoot():
            yield prim
            prim = prim.GetParent()

    # Source currently has no relevant groups. Until group filtering can be
    # completely reconstructed, refuse ambiguous native pair qualification.
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.CollisionGroup):
            query = UsdPhysics.CollisionGroup(prim).GetCollidersCollectionAPI().ComputeMembershipQuery()
            if any(query.IsPathIncluded(b.GetPath()) for b in bodies):
                raise ValueError('PR4 native support collision-group proof unavailable')
    for i, body in enumerate(bodies):
        others = bodies[1:] if i == 0 else bodies[:1]
        for ancestor in ancestors(body):
            for target in ancestor.GetRelationship('physics:filteredPairs').GetTargets():
                if any(other.GetPath().HasPrefix(target) for other in others):
                    raise ValueError('PR4 source excludes cloth/support collisions')

    def collision_data(prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('PR4 missing native support CollisionAPI')
        collision = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        offset = prim.GetAttribute('physxCollision:restOffset')
        if not collision.Get() or not offset or not offset.HasAuthoredValueOpinion() or offset.GetTimeSamples():
            raise ValueError('PR4 missing enabled collision/actual rest-offset support proof')
        for ancestor in ancestors(prim):
            attr = ancestor.GetAttribute('physics:collisionEnabled')
            if attr and attr.Get() is False:
                raise ValueError('PR4 native support collision disabled')
        return dict(path=str(prim.GetPath()), collision_enabled=True,
                    rest_offset_m=float(offset.Get()), rest_offset_source=REST_SOURCE)

    result = dict(schema=SCHEMA, pair_filter_check='no_source_pair_exclusion',
                  cloth=collision_data(cloth), surfaces=[])
    cache = UsdGeom.XformCache()
    for prim in surfaces:
        for ancestor in ancestors(prim):
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get():
                raise ValueError('PR4 source support must be native static collision geometry')
            if any(a.GetTimeSamples() for a in ancestor.GetAttributes() if a.GetName().startswith('xformOp:')):
                raise ValueError('PR4 animated support geometry requires per-frame evidence')
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError('PR4 source support requires actual Mesh collision faces')
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetPointsAttr().GetTimeSamples():
            raise ValueError('PR4 animated support mesh requires per-frame evidence')
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), float)
        points = np.asarray(mesh.GetPointsAttr().Get(), float) @ matrix[:3, :3] + matrix[3, :3]
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        if approximation not in ('none', 'convexHull'):
            raise ValueError('PR4 unsupported native support collision approximation')
        flip = (mesh.GetOrientationAttr().Get() == 'leftHanded') != (np.linalg.det(matrix[:3, :3]) < 0)
        triangles, cursor = [], 0
        # Numerical mesh validation only: based on source float precision, not
        # a scoring-distance allowance. No tolerance is added to contact tests.
        numeric = np.finfo(np.float32).eps*max(1., float(np.abs(points).max()))*8
        for count in counts:
            face = indices[cursor:cursor+count]; cursor += count
            if count < 3:
                raise ValueError('PR4 invalid native support polygon')
            for k in range(1, count-1):
                tri = [face[0], face[k], face[k+1]]
                if flip:tri[1], tri[2] = tri[2], tri[1]
                a, b, c = points[tri]; normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
                if length == 0:continue
                normal /= length
                # Fan triangulation is conservative only for actual planar,
                # convex source faces. ConvexHull triangles must be hull faces.
                if np.max(np.abs((points[face]-a)@normal)) > numeric:
                    raise ValueError('PR4 nonplanar source collision polygon')
                for j in range(count):
                    edge_a, edge_b = points[face[j]], points[face[(j+1)%count]]
                    if flip:edge_a, edge_b = edge_b, edge_a
                    if np.min(np.cross(edge_b-edge_a, points[face]-edge_a)@normal) < -numeric:
                        raise ValueError('PR4 nonconvex source collision polygon')
                if approximation == 'convexHull' and np.max((points-a)@normal) > numeric:
                    raise ValueError('PR4 authored support face is not on actual collision hull')
                triangles.append(tri)
        if cursor != len(indices):
            raise ValueError('PR4 native support topology differs')
        result['surfaces'].append(dict(collision_data(prim), static=True, geometry_source=GEOMETRY_SOURCE,
                                      approximation=approximation, points=points.tolist(), triangles=triangles))
    return validate_support(result, origin)
