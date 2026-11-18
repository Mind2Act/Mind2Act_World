"""Preserve the selected PR4 source's four same-mesh schema migrations.

The shared X5 owner calls this only while replacing the prepared asset. No pad,
geometry, drive, solver, material, or new collision value is introduced.
"""
import hashlib

import numpy as np

from mind2act_eval.contracts import to_json
from mind2act_eval.sim.physics_profile import profile_for_instance


ATTRIBUTES = ('physics:approximation', 'physics:collisionEnabled',
              'physxCollision:contactOffset', 'physxCollision:restOffset')


def _bindings():
    return [dict(source_xform=f'/World/Robot{side}/Asset/{link}/collisions/{link}/node_STL_BINARY_',
                 same_mesh=f'/World/Robot{side}/Asset/{link}/collisions/{link}/node_STL_BINARY_/mesh')
            for side in ('Left', 'Right') for link in ('link7', 'link8')]


def _geometry(stage, binding):
    from pxr import UsdGeom
    mesh = stage.GetPrimAtPath(binding['same_mesh'])
    body = stage.GetPrimAtPath(binding['source_xform'].split('/collisions/')[0])
    if not mesh or not mesh.IsA(UsdGeom.Mesh) or not body:
        raise ValueError('PR4 source finger geometry differs: ' + binding['same_mesh'])
    digest = hashlib.sha256()
    for name, dtype in (('points', '<f8'), ('faceVertexCounts', '<i8'), ('faceVertexIndices', '<i8')):
        value = mesh.GetAttribute(name).Get()
        if value is None:
            raise ValueError('PR4 source finger geometry missing: ' + name)
        array = np.asarray(value, dtype=dtype)
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    matrix = UsdGeom.XformCache().ComputeRelativeTransform(mesh, body)[0]
    digest.update(np.asarray(matrix, dtype='<f8').tobytes())
    return digest.hexdigest()


def capture(stage, spec):
    """Read only, before replacement; the actual runner-bound source selects this."""
    if spec is None or spec.task_id != 'PR4':
        return None
    from pxr import UsdGeom, UsdPhysics
    profile_for_instance(spec)  # Pinned physical configuration only; case admission is task-owned.
    expected = _bindings()
    actual = spec.config.get('source_case', {}).get('origin', {}).get('collider_schema_migration')
    if to_json(actual) != expected:
        raise ValueError('PR4 source finger schema migration paths differ')
    meshes = []
    for binding in expected:
        parent = stage.GetPrimAtPath(binding['source_xform'])
        mesh = stage.GetPrimAtPath(binding['same_mesh'])
        if (not parent or not parent.IsA(UsdGeom.Xform) or not mesh
                or parent.HasAPI(UsdPhysics.CollisionAPI) or parent.HasAPI(UsdPhysics.MeshCollisionAPI)
                or not mesh.HasAPI(UsdPhysics.CollisionAPI) or not mesh.HasAPI(UsdPhysics.MeshCollisionAPI)):
            raise ValueError('PR4 prepared source finger schema binding missing')
        attributes = []
        for name in ATTRIBUTES:
            attr = mesh.GetAttribute(name)
            if (not attr or not attr.HasAuthoredValueOpinion() or attr.GetTimeSamples()
                    or attr.GetConnections() or attr.Get() is None):
                raise ValueError('PR4 source finger collision attribute missing: ' + name)
            attributes.append((name, attr.GetTypeName(), attr.Get()))
        if (UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is not True
                or UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() != 'convexHull'):
            raise ValueError('PR4 source finger convex collision binding differs')
        meshes.append(dict(binding=binding, geometry_sha256=_geometry(stage, binding), attributes=attributes))
    return dict(spec=spec, meshes=meshes)


def restore(stage, source):
    """Restore original schemas/scalars only after all four geometries match."""
    from pxr import UsdPhysics
    for row in source['meshes']:
        binding = row['binding']
        if _geometry(stage, binding) != row['geometry_sha256']:
            raise ValueError('PR4 source finger geometry differs: ' + binding['same_mesh'])
        parent = stage.GetPrimAtPath(binding['source_xform'])
        if not parent.HasAPI(UsdPhysics.CollisionAPI) or not parent.HasAPI(UsdPhysics.MeshCollisionAPI):
            raise ValueError('PR4 official finger collision schema binding missing')
    for row in source['meshes']:
        parent = stage.GetPrimAtPath(row['binding']['source_xform'])
        mesh = stage.GetPrimAtPath(row['binding']['same_mesh'])
        parent.RemoveAPI(UsdPhysics.CollisionAPI); parent.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        UsdPhysics.CollisionAPI.Apply(mesh); UsdPhysics.MeshCollisionAPI.Apply(mesh)
        for name, dtype, value in row['attributes']:
            attr = mesh.CreateAttribute(name, dtype)
            attr.Set(value)
            if attr.Get() != value:
                raise ValueError('PR4 source finger collision attribute readback differs: ' + name)
    spec = source['spec']
    return dict(schema='mind2act_eval.pr4_source_collision_binding.v1', geometry_unchanged=True,
                source_spec_sha256=spec.config['source_spec_sha256'],
                source_config_sha256=spec.config['source_config_sha256'], scene_sha256=spec.config['scene_sha256'],
                meshes=[dict(**row['binding'], geometry_sha256=row['geometry_sha256'],
                             attributes={name: value for name, _, value in row['attributes']})
                        for row in source['meshes']])
