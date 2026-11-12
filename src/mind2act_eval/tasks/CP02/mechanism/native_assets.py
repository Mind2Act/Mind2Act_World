# Ported from cases/CP/CP02/runtime/assets.py; SHA256 a023966e77b1feb5d5c9d2a297d989191baf373551cddd52a5ac51a240120a16
"""Canonical CP02 authored assets and physical setup."""
import math
import numpy as np
from pxr import Gf,Sdf,UsdGeom,UsdShade,UsdPhysics
from mind2act_eval.sim.stage_authoring import copy_local_prim, remove_composed_prim
ROOT="/World/CP02V1"

def deduplicate_cover_support(stage):
    """Remove a static collider overlapping the rotating cover-support ring.

    Visual geometry remains authored. Cover support must be measured against
    the actual CakeAssembly rather than relabelling static-platter forces.
    """
    from pxr import Usd
    plate=stage.GetPrimAtPath(ROOT+'/Machine/Platter')
    cake=stage.GetPrimAtPath(ROOT+'/CakeAssembly')
    if not plate or not plate.IsA(UsdGeom.Mesh) or not plate.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Expected authored Machine/Platter mesh collider')
    parent=plate
    while parent:
        if parent.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Expected static Machine/Platter, not another rigid body')
        parent=parent.GetParent()
    if not cake or not cake.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Missing rotating CakeAssembly rigid body')
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default'])
    def bounds(prim):
        box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.asarray(box.GetMin()),np.asarray(box.GetMax())
    lo,hi=bounds(plate);rows=[]
    for i in range(48):
        path=ROOT+'/CakeAssembly/CoverSupport'+str(i);prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Cube) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Missing authored rotating cover support: '+path)
        low,high=bounds(prim);overlap=np.minimum(hi,high)-np.maximum(lo,low)
        if not np.isfinite(overlap).all() or not np.all(overlap>0):
            raise ValueError('Support ring no longer overlaps static platter; inspect changed geometry')
        rows.append(dict(path=path,world_min_m=low.tolist(),world_max_m=high.tolist(),
                         overlap_extent_m=overlap.tolist(),top_minus_platter_top_m=float(high[2]-hi[2])))
    api=UsdPhysics.CollisionAPI(plate);before=api.GetCollisionEnabledAttr().Get()
    api.CreateCollisionEnabledAttr(False)
    return dict(schema='cp02-hard-cover-support-topology-v1',disabled_collider=str(plate.GetPath()),
        collision_enabled_before=before,collision_enabled_after=False,
        platter_world_min_m=lo.tolist(),platter_world_max_m=hi.tolist(),supports=rows,
        rotating_support_body=str(cake.GetPath()),render_geometry_unchanged=True,
        support_flag_override=False,original240tick_requirement_unchanged=True,
        reason='Static authored platter and rotating support ring occupy overlapping volume at the same top surface; remove duplicate static contact only for Hard')

def restore_authored(stage, manifest):
    """Copy the pinned cake assets, including mesh/material closure, into CP02.

 Source files are visual templates: their transforms are not the fixed case.
 Only the two identified asset roots are replaced; station prims are retained.
 """
    from pxr import Usd
    spec = manifest['spec']
    world = stage.GetPrimAtPath('/World')
    candidates = [p for p in world.GetChildren() if p.GetChild('CakeAssembly') and p.GetChild('Source')]
    props = [p for p in world.GetChildren() if p.GetChild('Box') and p.GetChild('Lid')]
    if len(candidates) != 1 or len(props) != 1 or candidates[0] == props[0]:
        raise RuntimeError('CP02 requires one pinned cake asset root and one prop root')
    roots = [candidates[0], props[0]]
    original_center = UsdGeom.Xformable(candidates[0].GetChild('CakeAssembly')).GetOrderedXformOps()[0].Get()
    UsdGeom.Xform.Define(stage, ROOT)
    path_map = {p.GetPath(): Sdf.Path(ROOT) for p in roots}
    for parent in roots:
        for child in parent.GetChildren():
            dest = Sdf.Path(ROOT).AppendChild(child.GetName())
            if stage.GetPrimAtPath(dest):
                raise RuntimeError('Duplicate CP02 asset child: ' + str(dest))
            copy_local_prim(stage, child.GetPath(), dest)
    external = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        for rel in prim.GetRelationships():
            if rel.GetName().startswith('material:binding'):
                external.update((p for p in rel.GetTargets() if p.HasPrefix('/World/Materials')))
    UsdGeom.Scope.Define(stage, ROOT + '/Looks')
    for i, old in enumerate(sorted(external, key=str)):
        new = Sdf.Path(ROOT + '/Looks/Authored_' + str(i))
        copy_local_prim(stage, old, new)
        path_map[old] = new

    def remap(path):
        for old, new in sorted(path_map.items(), key=lambda x: -len(str(x[0]))):
            if path.HasPrefix(old):
                return path.ReplacePrefix(old, new)
        return path
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        for rel in prim.GetRelationships():
            old = rel.GetTargets()
            new = [remap(p) for p in old]
            if old != new:
                rel.SetTargets(new)
        for attr in prim.GetAttributes():
            old = attr.GetConnections()
            new = [remap(p) for p in old]
            if old != new:
                attr.SetConnections(new)
    for old in roots:
        remove_composed_prim(stage, old.GetPath())

    def translate(name, delta):
        p = stage.GetPrimAtPath(ROOT + '/' + name)
        if not p:
            return
        xf = UsdGeom.Xformable(p)
        ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if ops:
            ops[0].Set(ops[0].Get() + Gf.Vec3d(*delta))
        else:
            UsdGeom.Xform.Define(stage, p.GetPath()).AddTranslateOp().Set(Gf.Vec3d(*delta))
    shift = spec['ingredient_shift_xy_m']
    for name in ('IngredientTray', 'Parchment', 'Box', 'Lid'):
        translate(name, [*shift, 0.0])
    center = spec['center_xy_m']
    translate('Machine', [center[0] - original_center[0], center[1] - original_center[1], 0.0])
    v = dict(manifest['v4'])
    v['lid_pivot'] = list(stage.GetPrimAtPath(ROOT + '/Lid').GetAttribute('xformOp:translate').Get())
    v['lid_handle_local'] = list(stage.GetPrimAtPath(ROOT + '/Lid/Handle').GetAttribute('xformOp:translate').Get())
    v['cover_target'] = [*center, spec['assembly_z_m']]
    buttons = {name: list(pos) for name, pos in v['buttons'].items()}
    if buttons:
        delta = None
        for name, pos in buttons.items():
            old = stage.GetPrimAtPath(ROOT + '/' + name + 'Button').GetAttribute('xformOp:translate').Get()
            move = [pos[i] - old[i] for i in range(3)]
            if delta is not None and (not np.allclose(delta, move)):
                raise RuntimeError('Non-rigid CP02 controls binding')
            delta = move
            translate(name + 'Button', move)
        translate('Controls', delta)
    if spec['difficulty'] == 'hard':
        old = stage.GetPrimAtPath(ROOT + '/Cover').GetAttribute('xformOp:translate').Get()
        delta = [v['cover_source'][i] - old[i] for i in range(3)]
        translate('Cover', delta)
        translate('CoverMat', delta)
        from .cover_material import apply_opaque_cover
        apply_opaque_cover(stage)
    v['asset_mode'] = 'pinned authored mesh closure rebased to CP02V1'
    return v

def physicalize(stage, manifest, collision, rigid, cylinder):
    """Dynamic hinged lid, sprung momentary switches, and hollow free cover for CP02."""
    from pxr import Usd
    v = manifest['v4']
    level = manifest['difficulty']
    from .box_clearance import verify_rear_extension
    v['box_rear_clearance'] = verify_rear_extension(stage)
    v['lid_pivot'] = list(stage.GetPrimAtPath(ROOT + '/Lid').GetAttribute('xformOp:translate').Get())
    v['lid_handle_local'] = list(stage.GetPrimAtPath(ROOT + '/Lid/Handle').GetAttribute('xformOp:translate').Get())
    for path in [ROOT + '/Box', ROOT + '/Controls', ROOT + '/CoverMat']:
        prim = stage.GetPrimAtPath(path)
        if prim:
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Cube) or child.IsA(UsdGeom.Cylinder):
                    collision(child)
    lid = stage.GetPrimAtPath(ROOT + '/Lid')
    rigid(lid, 0.055)
    UsdPhysics.FilteredPairsAPI.Apply(lid).CreateFilteredPairsRel().SetTargets([p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT + '/Box')) if p.HasAPI(UsdPhysics.CollisionAPI)])
    mass = UsdPhysics.MassAPI(lid)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, -0.074, 0.003))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00011, 0.00063, 0.00073))
    from .box_clearance import update_lid_mass_properties
    update_lid_mass_properties(stage)
    for child in Usd.PrimRange(lid):
        if child.IsA(UsdGeom.Cube) or child.IsA(UsdGeom.Cylinder) or child.IsA(UsdGeom.Sphere):
            collision(child)
    joint = UsdPhysics.RevoluteJoint.Define(stage, ROOT + '/LidHinge')
    joint.CreateBody0Rel().SetTargets([])
    joint.CreateBody1Rel().SetTargets([ROOT + '/Lid'])
    joint.CreateAxisAttr('X')
    joint.CreateLocalPos0Attr(Gf.Vec3f(*v['lid_pivot']))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(-105.0)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateExcludeFromArticulationAttr(True)
    damping = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    damping.CreateStiffnessAttr(0.0)
    damping.CreateDampingAttr(0.003)
    damping.CreateMaxForceAttr(0.03)
    damping.CreateTargetVelocityAttr(0.0)
    for name, pos in v['buttons'].items():
        path = ROOT + '/' + name + 'Button'
        node = stage.GetPrimAtPath(path)
        rigid(node, 0.005)
        UsdPhysics.FilteredPairsAPI.Apply(node).CreateFilteredPairsRel().SetTargets([p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT + '/Controls')) if p.HasAPI(UsdPhysics.CollisionAPI)])
        collision(stage.GetPrimAtPath(path + '/Cap'))
        j = UsdPhysics.PrismaticJoint.Define(stage, ROOT + '/' + name + 'Slide')
        j.CreateBody0Rel().SetTargets([])
        j.CreateBody1Rel().SetTargets([path])
        j.CreateAxisAttr('Z')
        j.CreateLocalPos0Attr(Gf.Vec3f(pos[0], pos[1], pos[2] + 0.015))
        j.CreateLocalPos1Attr(Gf.Vec3f(0))
        j.CreateLocalRot0Attr(Gf.Quatf(1))
        j.CreateLocalRot1Attr(Gf.Quatf(1))
        j.CreateLowerLimitAttr(-0.006)
        j.CreateUpperLimitAttr(0.0)
        j.CreateExcludeFromArticulationAttr(True)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
        d.CreateStiffnessAttr(200.0)
        d.CreateDampingAttr(1.0)
        d.CreateMaxForceAttr(5.0)
        d.CreateTargetPositionAttr(0.0)
    if level == 'hard':
        path = ROOT + '/Cover'
        node = stage.GetPrimAtPath(path)
        rigid(node, 0.095)
        mass = UsdPhysics.MassAPI(node)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.055))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00075, 0.00075, 0.0013))
        for name in ['HandleStem', 'Handle']:
            collision(stage.GetPrimAtPath(path + '/' + name))
        cylinder(path + '/Collision/Roof', 0.127, 0.003, (0, 0, 0.1205))
        for i in range(48):
            t = i * 2 * math.pi / 48
            c = UsdGeom.Cube.Define(stage, path + f'/Collision/Wall{i}')
            c.CreateSizeAttr(1)
            c.AddTranslateOp().Set(Gf.Vec3d(0.1285 * math.cos(t), 0.1285 * math.sin(t), 0.0595))
            c.AddRotateZOp().Set(math.degrees(t))
            c.AddScaleOp().Set(Gf.Vec3d(0.003, 2 * 0.13 * math.tan(math.pi / 48) * 1.01, 0.119))
            c.MakeInvisible()
            collision(c.GetPrim())
    for i in range(48):
        t = i * 2 * math.pi / 48
        c = UsdGeom.Cube.Define(stage, '/World/CP02V1/CakeAssembly' + f'/CoverSupport{i}')
        c.CreateSizeAttr(1)
        c.AddTranslateOp().Set(Gf.Vec3d(0.127 * math.cos(t), 0.127 * math.sin(t), -0.002))
        c.AddRotateZOp().Set(math.degrees(t))
        c.AddScaleOp().Set(Gf.Vec3d(0.021, 0.019, 0.004))
        c.MakeInvisible()
        collision(c.GetPrim())
    if level=='hard':return deduplicate_cover_support(stage)
