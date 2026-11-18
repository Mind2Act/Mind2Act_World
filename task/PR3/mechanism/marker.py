"""PR3 station physics around the prepared scene: dynamic marker + passive nib, paper board proxy,
saddle, static collision of the station, faint reference path texture, live ink.

Ported from the baseline (source_id pr3-original-source-mainline-pro-sourcepath,
cases_unified/PR3/world.py). Not ported: every robot patch (articulation, drives, fixed base, X5 pad
settings, configure_stage) — those belong to robot/ — and the simulation-app / World setup, which sim/ owns.
The prepared scene is an instance asset; nothing here recomputes geometry from builder source.
"""
import numpy as np

from . import physx
from .live_ink import LivePaperInk
from .passive_nib import install_passive_nib

BOARD = "/World/MarkerPhysics/BoardTop"
MARKER = "/World/MarkerDynamic"


def select_level(stage, level):
    from pxr import Usd, UsdGeom, UsdPhysics
    selected = f"/World/Marker_{level}"
    if not stage.GetPrimAtPath(selected):
        raise ValueError(f"scene lacks expected difficulty root {selected}")
    for difficulty in ("easy", "medium", "hard"):
        root = stage.GetPrimAtPath(f"/World/Marker_{difficulty}")
        if not root:
            continue
        if difficulty == level:
            UsdGeom.Imageable(root).MakeVisible()
            continue
        UsdGeom.Imageable(root).MakeInvisible()
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    return selected


def author_marker(stage, selected_root, layout):
    """Dynamic marker body with copied visuals, cylinder colliders, passive nib and physics material."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    marker, contact = layout["marker"], layout["contact"]
    offset = contact["contact_offset_m"]
    center = np.asarray(marker["center_world_m"])
    body = UsdGeom.Xform.Define(stage, MARKER)
    body.AddTranslateOp().Set(Gf.Vec3d(*center))
    body_prim = body.GetPrim()
    physx.dynamic_body(body_prim, marker["mass_kg"], marker["solver_position_iterations"], marker["solver_velocity_iterations"])
    visuals = UsdGeom.Xform.Define(stage, MARKER + "/Visuals")
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(body_prim).GetInverse()
    root_layer = stage.GetRootLayer()
    names = set(layout["marker_visual_names"])

    def is_visual(child):
        return child.GetName() in names or child.GetName().startswith("CapPocketClip")

    copied = []
    source = stage.GetPrimAtPath(selected_root)
    for child in source.GetChildren():
        if not is_visual(child):
            continue
        destination = visuals.GetPath().AppendChild(child.GetName())
        matrix = cache.GetLocalToWorldTransform(child)
        matrix.SetTranslateOnly(matrix.ExtractTranslation() + Gf.Vec3d(0, 0, float(center[2] - .789)))
        if not Sdf.CopySpec(root_layer, child.GetPath(), root_layer, destination):
            raise RuntimeError(f"failed to CopySpec marker visual {child.GetPath()}")
        UsdGeom.Xformable(stage.GetPrimAtPath(destination)).MakeMatrixXform().Set(matrix * inverse)
        copied.append(str(destination))
    if not copied:
        raise RuntimeError("marker hierarchy inspection found no marker visuals")
    for child in list(source.GetChildren()):
        if is_visual(child):
            stage.RemovePrim(child.GetPath())

    colliders = UsdGeom.Xform.Define(stage, MARKER + "/Colliders")

    def cylinder(name, spec):
        prim = UsdGeom.Cylinder.Define(stage, f"{colliders.GetPath()}/{name}")
        prim.CreateRadiusAttr(spec["radius"])
        prim.CreateHeightAttr(spec["length"])
        prim.AddTranslateOp().Set(Gf.Vec3d(spec["x"], 0.0, 0.0))
        prim.AddRotateYOp().Set(90.0)
        return prim.GetPrim()

    c = layout["colliders"]
    barrel, cap, neck = cylinder("Barrel", c["barrel"]), cylinder("Cap", c["cap"]), cylinder("TipNeck", c["neck"])
    physx.collision(barrel, "convexHull", offset, torsionalPatchRadius=contact["barrel_torsional_patch_radius_m"],
                    minTorsionalPatchRadius=contact["barrel_min_torsional_patch_radius_m"])
    physx.collision(cap, "convexHull", offset)
    physx.collision(neck, "convexHull", offset)
    tip = UsdGeom.Sphere.Define(stage, str(colliders.GetPath()) + "/Tip")
    tip.CreateRadiusAttr(c["tip"]["radius"])
    tip.AddTranslateOp().Set(Gf.Vec3d(c["tip"]["x"], 0.0, 0.0))
    physx.collision(tip.GetPrim(), "convexHull", offset)
    UsdGeom.Imageable(colliders.GetPrim()).MakeInvisible()
    nib_body, tip_path = install_passive_nib(stage, center, offset)
    material = UsdShade.Material.Define(stage, "/World/MarkerPhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(contact["static_friction"])
    api.CreateDynamicFrictionAttr(contact["dynamic_friction"])
    api.CreateRestitutionAttr(contact["restitution"])
    for path in (str(barrel.GetPath()), str(cap.GetPath()), str(neck.GetPath()), tip_path):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material, materialPurpose="physics")
    return dict(nib_body=nib_body, tip=tip_path)


def write_reference_texture(path, target_xy_mm):
    """Faint reference path drawn on the printed paper (generated; the source scene asset is unchanged)."""
    from PIL import Image, ImageDraw
    texture = Image.new('RGB', (2200, 1425), (245, 245, 237))
    draw = ImageDraw.Draw(texture)
    coords = [(round((x + 220) * 5), round((142.5 - y) * 5)) for x, y in target_xy_mm]
    draw.line(coords, fill=(90, 120, 106), width=13, joint='curve')
    path.parent.mkdir(parents=True, exist_ok=True)
    texture.save(path)
    with Image.open(path) as check:
        check.verify()
    return path


def bind_reference_texture(stage, selected_root, level, texture_path):
    from pxr import Sdf, Usd, UsdShade
    printed = stage.GetPrimAtPath(f"{selected_root}/PrintedTarget_{level}")
    bound, _ = UsdShade.MaterialBindingAPI(printed).ComputeBoundMaterial()
    nodes = [p for p in Usd.PrimRange(bound.GetPrim())
             if p.IsA(UsdShade.Shader) and UsdShade.Shader(p).GetIdAttr().Get() == 'UsdUVTexture']
    if not nodes:
        raise RuntimeError('Cannot locate paper texture material')
    for node in nodes:
        UsdShade.Shader(node).GetInput('file').Set(Sdf.AssetPath(str(texture_path)))


def add_station_physics(stage, selected_root, layout):
    """Board proxy, saddle strips and static collision of the clamp and station (baseline 'none' triangle meshes)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    offset = layout["contact"]["contact_offset_m"]
    paper_z, size = layout["paper"]["top_z_m"], layout["paper"]["board_proxy_size_m"]
    board = UsdGeom.Cube.Define(stage, BOARD)
    board.CreateSizeAttr(1.0)
    board.AddTranslateOp().Set(Gf.Vec3d(0.0, layout["paper"]["board_proxy_center_y_m"], paper_z - 0.0015))
    board.AddScaleOp().Set(Gf.Vec3f(*size))
    physx.collision(board.GetPrim(), "convexHull", offset)

    def static_tree(root_path, approximation="none"):
        root = stage.GetPrimAtPath(root_path)
        paths = []
        if not root:
            return paths
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                physx.collision(prim, approximation, offset)
                paths.append(str(prim.GetPath()))
        return paths

    saddle = static_tree(f"{selected_root}/SiliconeRest")
    # Raise and widen the visible support to match the physical groove.
    for name in ("ConcaveSaddle0", "ConcaveSaddle1"):
        visible = UsdGeom.Mesh(stage.GetPrimAtPath(selected_root + "/" + name))
        points = []
        for point in visible.GetPointsAttr().Get():
            x, y, z = map(float, point)
            dy = y + .171
            if abs(dy) <= .01021:
                y = -.171 + dy * .012 / .0102
                z = .796 + (z - .789) * .012 / .0102 if z <= .78901 and z > .7741 else z + .007
            elif z > .7741:
                z += .007
            points.append((x, y, z))
        visible.GetPointsAttr().Set(points)
    for k, x in enumerate((-.032, .042)):  # convex strips reproduce the U-saddle
        for j in range(32):
            angles = np.array([j, j + 1]) * np.pi / 32
            yy = -.171 + .012 * np.cos(angles)
            zz = .796 - .012 * np.sin(angles)
            points = [(xx, y, z) for xx in (x - .007, x + .007) for y, z in [(yy[0], .774), (yy[1], .774), (yy[1], zz[1]), (yy[0], zz[0])]]
            mesh = UsdGeom.Mesh.Define(stage, f"/World/MarkerPhysics/Saddle{k}Strip{j}")
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([4] * 6)
            mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
            physx.collision(mesh.GetPrim(), "convexHull", offset)
            saddle.append(str(mesh.GetPath()))
    clamp = []
    for child in stage.GetPrimAtPath(selected_root).GetChildren():
        if child.GetName().startswith(tuple(layout["clamp_prefixes"])):
            clamp.extend(static_tree(str(child.GetPath())))
    station = [BOARD, *saddle, *clamp]
    for root in layout["static_roots"]:
        station.extend(static_tree(root))
    if not station:
        raise RuntimeError("station collision audit found no static geometry")
    return station


def add_ink(stage, paper_z, width_m):
    return LivePaperInk(stage, '/World/LivePaperInk', paper_z, width_m)
