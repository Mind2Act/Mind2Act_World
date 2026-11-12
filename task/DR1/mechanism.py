"""DR1 native mechanism on the prepared FR1 scene: belt surface-velocity conveyor and kinematic parcels.

Ported from the source builder `cases_unified/FR1/world.py` (source_id fr1-retained-train-x5-120hz-v1,
world.py sha256 f890d88c8a1f...). Parcels keep the collection roles exactly: kinematic, parked out of
view, released to dynamic at tick 1. Robot, clock and cameras belong
to robot/ and sim/. EnableCCD on the parcels is authored exactly as the collection source does (its
PhysX 'kinematic with CCD' warning is a collection-original trait, recorded in docs/ports/DR1.md); no
velocity is ever written to a kinematic parcel. Reference-carrier frustum scans of the source are validation-only and are skipped.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from mind2act_eval.sim.stage_authoring import copy_local_prim, remove_composed_prim


@dataclass
class Handles:
    package_roots: list = field(default_factory=list)
    rbapis: list = field(default_factory=list)
    dims: list = field(default_factory=list)
    ref_root: object = None
    speed_attr: object = None
    head_camera: object = None
    camera_intrinsics0: list = field(default_factory=list)
    fixed_matrices0: dict = field(default_factory=dict)


def author_scene(stage, spec, lay):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdShade, Sdf
    cfg = spec.config
    h = Handles()
    belt_bounds = cfg['belt_bounds_m']
    belt_y = (belt_bounds[2] + belt_bounds[3]) / 2
    belt_width = belt_bounds[3] - belt_bounds[2]
    speed = float(cfg['parameters']['speed_m_s'])
    for path in lay['removed_prims'][:2]:
        remove_composed_prim(stage, path)
    tm = lay['tray_material']
    shader = UsdShade.Shader(stage.GetPrimAtPath(tm['shader']))
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(*tm['diffuse_rgb']))
    shader.GetInput('metallic').Set(tm['metallic'])
    shader.GetInput('roughness').Set(tm['roughness'])
    h.head_camera = UsdGeom.Camera(stage.GetPrimAtPath(lay['head_camera']))
    if not h.head_camera:
        raise RuntimeError('public HeadCamera missing')
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/FR1')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

    g = lay['goods']
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    park = g['parking_xy_z']
    source_paths = {i['identity']: cfg['geometry_paths'][i['identity']] for i in cfg['stream_goods']}
    ops = []
    for i, item in enumerate(cfg['stream_goods']):
        src = stage.GetPrimAtPath(source_paths[item['identity']])
        if not src:
            raise RuntimeError('manifest stream prim missing: ' + source_paths[item['identity']])
        UsdGeom.Imageable(src).MakeVisible()
        cache.Clear()
        center = np.asarray(cache.ComputeWorldBound(src).ComputeAlignedRange().GetMidpoint(), float)
        visual_world = UsdGeom.XformCache().GetLocalToWorldTransform(src)
        path = '/World/FullStream/Goods_%02d' % i
        pkg = UsdGeom.Xform.Define(stage, path)
        op = pkg.AddTranslateOp()
        op.Set(Gf.Vec3d(park[0] + g['parking_dx_m'] * i, park[1], park[2]))
        copy_local_prim(stage, src.GetPath(), path + '/Visual')
        visual = UsdGeom.Xformable(stage.GetPrimAtPath(path + '/Visual'))
        visual.ClearXformOpOrder()
        visual.AddTransformOp().Set(visual_world * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(-center))))
        h.package_roots.append(path)
        h.dims.append(np.asarray(item['size_m'], float))
        ops.append(op)
    remove_composed_prim(stage, '/World/FR1/StreamGoods')

    h.ref_root = stage.GetPrimAtPath(lay['ref_carrier'])
    if not h.ref_root:
        raise RuntimeError('reference carrier missing')
    UsdGeom.Imageable(h.ref_root).MakeInvisible()

    cm = lay['contact_material']
    material = UsdShade.Material.Define(stage, '/World/FullContactMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(cm['static_friction'])
    pm.CreateDynamicFrictionAttr(cm['dynamic_friction'])
    pm.CreateRestitutionAttr(cm['restitution'])

    def collision(prim, approximation='none'):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(cm['contact_offset_m'])
        api.CreateRestOffsetAttr(cm['rest_offset_m'])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')

    for prefix in lay['static_prefixes']:
        prim = stage.GetPrimAtPath(prefix)
        if not prim:
            continue
        for child in Usd.PrimRange(prim):
            text = str(child.GetPath())
            if '/OriginalBelt/' in text or 'Marker' in text:
                continue
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Cube) or child.IsA(UsdGeom.Cylinder):
                collision(child)
    bs = lay['belt_surface']
    belt = UsdGeom.Cube.Define(stage, '/World/BeltContactSurface')
    belt.CreateSizeAttr(1)
    belt.AddTranslateOp().Set(Gf.Vec3d(0, belt_y, bs['center_z_m']))
    belt.AddScaleOp().Set(Gf.Vec3f(belt_bounds[1] - belt_bounds[0], belt_width, bs['thickness_m']))
    belt.MakeInvisible()
    collision(belt.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(belt.GetPrim())
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt.GetPrim())
    surface.CreateSurfaceVelocityEnabledAttr(True)
    surface.CreateSurfaceVelocityLocalSpaceAttr(False)
    h.speed_attr = surface.CreateSurfaceVelocityAttr(Gf.Vec3f(speed, 0, 0))

    for path in h.package_roots:
        pkg = stage.GetPrimAtPath(path)
        UsdGeom.Imageable(pkg).MakeVisible()
        for child in Usd.PrimRange(pkg):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if child.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(child).CreateJointEnabledAttr(False)
            if child.IsA(UsdGeom.Mesh):
                visible = UsdGeom.Imageable(child).ComputeVisibility() != UsdGeom.Tokens.invisible
                if visible and 'collision' not in str(child.GetPath()).lower():
                    collision(child, g['collision_approximation'])
                elif child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        body = UsdPhysics.RigidBodyAPI.Apply(pkg)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(True)  # collection role: kinematic until released at tick 1
        h.rbapis.append(body)
        UsdPhysics.MassAPI.Apply(pkg).CreateMassAttr(g['mass_kg'])
        phys = PhysxSchema.PhysxRigidBodyAPI.Apply(pkg)
        phys.CreateSolverPositionIterationCountAttr(g['position_iterations'])
        phys.CreateSolverVelocityIterationCountAttr(g['velocity_iterations'])
        phys.CreateSleepThresholdAttr(g['sleep_threshold'])
        phys.CreateEnableCCDAttr(True)
        PhysxSchema.PhysxContactReportAPI.Apply(pkg).CreateThresholdAttr(0.)
        UsdGeom.Imageable(pkg).MakeInvisible()
    return h


def snapshot_fixed(stage, lay, h):
    from pxr import UsdGeom
    h.camera_intrinsics0 = [float(h.head_camera.GetFocalLengthAttr().Get()),
                            float(h.head_camera.GetHorizontalApertureAttr().Get()),
                            float(h.head_camera.GetVerticalApertureAttr().Get())]
    cache = UsdGeom.XformCache()
    h.fixed_matrices0 = {p: np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)))
                         for p in lay['fixed_paths'] if stage.GetPrimAtPath(p)}


def release_parcel(stage, h, bodies, item, index):
    """Tick-1 activation exactly as the source Timeline: place, show, make dynamic."""
    from pxr import UsdGeom
    yaw = np.radians(item['spawn_yaw_deg']) / 2.
    bodies[index].set_world_pose(np.asarray(item['spawn_center']), np.array([np.cos(yaw), 0., 0., np.sin(yaw)]))
    UsdGeom.Imageable(stage.GetPrimAtPath(h.package_roots[index])).MakeVisible()
    h.rbapis[index].GetKinematicEnabledAttr().Set(False)


def parcel_extent(quaternion_wxyz, dims):
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.abs(rot) @ dims


def subscribe_contacts(table):
    from pxr import PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType

    def report(headers, data):
        for header in headers:
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.collider0, header.collider1))
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            impulse = max((float(np.linalg.norm(p.impulse)) for p in points), default=0.)
            separation = min((float(p.separation) for p in points), default=1.)
            sign = 1. if '/World/FullStream/Goods_' in pair[0] else -1.
            normal = max((sign * float(p.normal[2]) for p in points), default=None)
            table.report(pair, header.type != ContactEventType.CONTACT_LOST, impulse, separation, normal)
    return get_physx_simulation_interface().subscribe_contact_report_events(report)
