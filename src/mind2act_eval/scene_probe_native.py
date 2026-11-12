"""Lazy USD/PhysX implementation for the restricted scene-probe projection.

The transform convention (world metres, wxyz, finalized pose preferred) follows
the trusted mind2act_random/clutter_spawn.py baseline. Its dynamic/zero-velocity,
mass-cap and friction setup is applied only under the new clutter namespace.
Task/robot physics and all existing collision geometry are never rewritten.
"""
from __future__ import annotations

import math
from pathlib import Path

from .scene_probe import APPEARANCE_SOURCE_SHA256, CLUTTER_SOURCE_SHA256, FLOOR_SOURCE_SHA256, ROOT, sha256


def _asset(root, relative, expected):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
        raise ValueError("trusted probe asset escapes root, is missing, or changed")
    return path


def physics_modes(stage):
    """Read actual authored task body/joint modes, never author them."""
    from pxr import UsdPhysics
    result = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(ROOT + "/"):
            continue
        attrs = {a.GetName(): repr(a.Get()) for a in prim.GetAttributes()
                 if a.GetName().startswith(("physics:", "physx"))}
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint) or attrs:
            result[path] = {"type": prim.GetTypeName(), "apis": list(prim.GetAppliedSchemas()), "attributes": attrs}
    return result


def spawn_clutter(stage, inputs, asset_root):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    if stage.GetPrimAtPath(ROOT):
        raise ValueError("probe namespace already exists")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        raise ValueError("finalized layout poses require the original metre stage")
    before = physics_modes(stage)
    UsdGeom.Xform.Define(stage, ROOT).SetResetXformStack(True)
    material = UsdShade.Material.Define(stage, ROOT + "/PhysicsMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.6)
    physics.CreateDynamicFrictionAttr(1.5)
    physics.CreateRestitutionAttr(0.)
    receipt = []
    for item in inputs.clutter:
        asset = _asset(asset_root, item.asset_path, item.asset_sha256)
        if asset.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise ValueError("clutter catalog asset must be USD")
        source = Usd.Stage.Open(str(asset))
        if source is None or not source.GetDefaultPrim():
            raise ValueError("trusted clutter USD lacks a default prim")
        base = Path(asset_root).resolve(strict=True)
        # Inspect loaded composition layers for path confinement. Do not turn
        # unresolved SDK/MDL/texture tokens into an extra dependency-closure gate.
        for dependency in (p.realPath for p in source.GetUsedLayers() if p.realPath):
            # Package-internal paths remain bound to the already verified package.
            outer = dependency.split("[", 1)[0]
            if outer and not Path(outer).resolve().is_relative_to(base):
                raise ValueError("clutter dependency escapes trusted asset root")
        root = ROOT + "/" + item.name
        xform = UsdGeom.Xform.Define(stage, root)
        xform.GetPrim().GetReferences().AddReference(str(asset), source.GetDefaultPrim().GetPath())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*item.position))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(item.orientation[0], Gf.Vec3d(*item.orientation[1:])))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*item.scale))
        bodies = [p for p in Usd.PrimRange(xform.GetPrim()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) > 1:
            raise ValueError("trusted clutter must not contain multiple rigid bodies")
        added_body_api = not bodies
        if added_body_api:
            bodies = [xform.GetPrim()]
            UsdPhysics.RigidBodyAPI.Apply(bodies[0])
        body = UsdPhysics.RigidBodyAPI(bodies[0])
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        body.CreateVelocityAttr(Gf.Vec3f(0))
        body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        UsdPhysics.MassAPI.Apply(bodies[0]).CreateMassAttr(item.mass_kg)
        if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(xform.GetPrim())):
            raise ValueError("clutter asset has no authored collider; no runtime shape repair permitted")
        for prim in Usd.PrimRange(xform.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        receipt.append({"name": item.name, "root": root, "body_path": str(bodies[0].GetPath()),
                        "asset": str(asset), "sha256": item.asset_sha256,
                        "pose_source": "settled", "orientation_convention": "wxyz", "frame": "world_metres",
                        "physics_policy": "fixed_source_clutter_spawn_defaults", "source_sha256": CLUTTER_SOURCE_SHA256,
                        "mass_kg": item.mass_kg, "static_friction": .6, "dynamic_friction": 1.5, "restitution": 0.,
                        "zero_velocity": True, "added_clutter_rigid_body_api": added_body_api,
                        "collision_geometry_changed": False})
    if physics_modes(stage) != before:
        raise ValueError("clutter insertion changed protected task/robot physics")
    return receipt


def _material(stage, path, name, root):
    from pxr import Sdf, UsdShade
    material = UsdShade.Material.Define(stage, root)
    shader = UsdShade.Shader.Define(stage, root + "/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(path)), "mdl")
    shader.SetSourceAssetSubIdentifier(name, "mdl")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateVolumeOutput("mdl"),
                   material.CreateDisplacementOutput("mdl")):
        output.ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _uvs(prim):
    """Trusted baseline convention: keep authored st; otherwise add render UVs."""
    from pxr import Gf, Sdf, UsdGeom
    api = UsdGeom.PrimvarsAPI(prim)
    if api.GetPrimvar("st") and api.GetPrimvar("st").Get() is not None:
        return
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        uv, cursor = [], 0
        for count in counts:
            face = [points[i] for i in indices[cursor:cursor + count]]
            cursor += count
            if count < 3:
                raise ValueError("degenerate native surface face")
            normal = Gf.Cross(face[1]-face[0], face[2]-face[0])
            axis = max(range(3), key=lambda i: abs(normal[i]))
            axes = (0, 1) if axis == 2 else ((0, 2) if axis == 1 else (1, 2))
            uv.extend((float(p[axes[0]])/.5, float(p[axes[1]])/.5) for p in face)
    elif prim.IsA(UsdGeom.Cube):
        uv = [(0, 0), (1, 0), (1, 1), (0, 1)]*6
    else:
        raise ValueError("untextured surface needs the source Mesh/Cube UV convention")
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uv)


def apply_appearance(stage, inputs, asset_root):
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    before = physics_modes(stage)
    rows = []
    for surface, material_row in (("table", inputs.layout["Appearance"]["material"]), ("floor", inputs.layout["ground"])):
        pin = inputs.catalog["materials"][material_row["id"]]
        path = _asset(asset_root, pin["path"], pin["sha256"])
        if path.suffix.lower() != ".mdl":
            raise ValueError("probe appearance currently requires cataloged MDL")
        material = _material(stage, path, pin["mdl_name"], ROOT + "/" + surface.title() + "Material")
        for target in inputs.catalog[surface + "_targets"]:
            prim = stage.GetPrimAtPath(target)
            if not prim or not prim.IsA(UsdGeom.Gprim) or prim.IsInstanceProxy():
                raise ValueError("explicit native surface must be a nonproxy Gprim")
            if any(p.IsA(UsdGeom.Subset) for p in Usd.PrimRange(prim)):
                raise ValueError("surface subsets require explicit binding review")
            _uvs(prim)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            physical = binding.ComputeBoundMaterial(materialPurpose="physics")[0]
            if physical and physical.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                binding.Bind(physical, materialPurpose="physics", bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            for purpose in ("", "preview", "full"):
                effective = binding.ComputeBoundMaterial(materialPurpose=purpose)[0]
                if not effective or effective.GetPath() != material.GetPath():
                    raise ValueError("native render binding blocked")
            if physical:
                effective = binding.ComputeBoundMaterial(materialPurpose="physics")[0]
                if not effective or effective.GetPath() != physical.GetPath():
                    raise ValueError("appearance changed effective surface physics material")
            rows.append({"surface": surface, "target": target, "material": material_row["id"], "sha256": pin["sha256"],
                         "source_sha256": FLOOR_SOURCE_SHA256 if surface == "floor" else APPEARANCE_SOURCE_SHA256})
    settings = inputs.layout["Appearance"]["light"]
    kind = settings["type"]
    target = inputs.catalog["dome_light" if kind == "Dome" else "key_light"]
    prim = stage.GetPrimAtPath(target)
    if not prim or not prim.HasAPI(UsdLux.LightAPI):
        raise ValueError("explicit native key light missing")
    original = UsdLux.LightAPI(prim)
    if (kind == "Dome") != prim.IsA(UsdLux.DomeLight):
        raise ValueError("Dome must use the existing HDR dome; key must be interchangeable")
    intensity = float(original.GetIntensityAttr().Get())
    exposure = float(original.GetExposureAttr().Get())
    def area(p):
        if p.IsA(UsdLux.RectLight):
            v = UsdLux.RectLight(p)
            return v.GetWidthAttr().Get() * v.GetHeightAttr().Get()
        if p.IsA(UsdLux.DiskLight):
            return math.pi * UsdLux.DiskLight(p).GetRadiusAttr().Get() ** 2
        if p.IsA(UsdLux.SphereLight):
            return 4 * math.pi * UsdLux.SphereLight(p).GetRadiusAttr().Get() ** 2
        if p.IsA(UsdLux.CylinderLight):
            v = UsdLux.CylinderLight(p)
            return 2 * math.pi * v.GetRadiusAttr().Get() * v.GetLengthAttr().Get()
        return None
    source_area = area(prim)
    if not math.isfinite(intensity) or intensity <= 0 or (source_area is not None and
                                                        (not math.isfinite(source_area) or source_area <= 0)):
        raise ValueError("invalid native emitter intensity/area")
    if kind in {"Rect", "Disk", "Sphere", "Cylinder"} and (source_area is None or source_area <= 0):
        raise ValueError("area-light conversion needs the trusted native emitter area")
    power = intensity if source_area is None or original.GetNormalizeAttr().Get() else intensity * source_area
    prim.SetTypeName(kind + "Light")
    light = UsdLux.LightAPI(prim)
    light.GetNormalizeAttr().Set(kind not in {"Dome", "Distant"})
    light.GetIntensityAttr().Set((350. if kind == "Distant" else power) * settings["intensity_scale"])
    light.GetExposureAttr().Set(exposure)
    light.GetColorAttr().Set(Gf.Vec3f(*settings["color_rgb"]))
    light.GetEnableColorTemperatureAttr().Set(False)
    if kind == "Distant":
        UsdLux.DistantLight(prim).GetAngleAttr().Set(1.)
    elif kind == "Rect":
        obj = UsdLux.RectLight(prim)
        obj.GetWidthAttr().Set(math.sqrt(source_area))
        obj.GetHeightAttr().Set(math.sqrt(source_area))
    elif kind in {"Disk", "Sphere"}:
        obj = (UsdLux.DiskLight if kind == "Disk" else UsdLux.SphereLight)(prim)
        obj.GetRadiusAttr().Set(math.sqrt(source_area / (math.pi if kind == "Disk" else 4 * math.pi)))
    elif kind == "Cylinder":
        obj = UsdLux.CylinderLight(prim)
        obj.GetLengthAttr().Set(math.sqrt(source_area))
        obj.GetRadiusAttr().Set(math.sqrt(source_area) / (2 * math.pi))
    if physics_modes(stage) != before:
        raise ValueError("visual projection changed protected task/robot physics")
    rows.append({"surface": "light", "target": target, "kind": kind,
                 "source_sha256": APPEARANCE_SOURCE_SHA256, "source_area": source_area,
                 "existing_hdr_retained": kind == "Dome"})
    return rows


def live_body_state(body):
    """Refuse USD pose fallback when the PhysX view did not register the actor."""
    from .scene_probe import _vector
    if not body.is_physics_handle_valid():
        raise RuntimeError("native clutter PhysX handle is not valid; USD-only poses cannot certify physics")
    position, orientation = body.get_world_pose()
    linear, angular = body.get_linear_velocity(), body.get_angular_velocity()
    if any(value is None for value in (position, orientation, linear, angular)):
        raise RuntimeError("native clutter PhysX state readback is missing")
    return (_vector(list(map(float, position)), 3), _vector(list(map(float, orientation)), 4, quaternion=True),
            _vector(list(map(float, linear)), 3), _vector(list(map(float, angular)), 3))


class NativeMonitor:
    def __init__(self, ctx, clutter, *, floor_targets):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import PhysxSchema, Usd, UsdGeom
        self.ctx, self.clutter = ctx, clutter
        self.floor_targets = tuple(floor_targets)
        self.bodies = {}
        for row in clutter:
            prim = ctx.sim.stage.GetPrimAtPath(row["body_path"])
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            handle = SingleRigidPrim(prim_path=row["body_path"], name="probe_"+row["name"])
            handle.initialize(ctx.world.physics_sim_view)
            live_body_state(handle)
            self.bodies[row["name"]] = handle
        self.ledger = ctx.sim.contacts

    def sample(self):
        from pxr import Usd, UsdGeom
        from omni.physx import get_physx_interface
        # Same readback flush used by the trusted PR6 source builder. It does
        # not step physics, and prevents stale USD bounds when Fabric is active.
        get_physx_interface().update_transformations(True, True, True, False)
        rows = {}
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        for row in self.clutter:
            body = self.bodies[row["name"]]
            position, orientation, linear, angular = live_body_state(body)
            bbox = cache.ComputeWorldBound(self.ctx.sim.stage.GetPrimAtPath(row["root"])).ComputeAlignedRange()
            if bbox.IsEmpty():
                raise RuntimeError("clutter has no observable world bound")
            contacts = [{"pair": list(pair), "separation_m": values[2], "impulse_ns": values[1]}
                        for pair, values in self.ledger.pairs.items()
                        if values[0] and any(p == row["root"] or p.startswith(row["root"]+"/") for p in pair)]
            rows[row["name"]] = {"position": list(map(float, position)), "orientation": list(map(float, orientation)),
                                  "linear_velocity_m_s": list(linear), "angular_velocity_rad_s": list(angular),
                                  "physics_handle_valid": True,
                                  "bottom_z": float(bbox.GetMin()[2]),
                                  "floor_contact": any(any(p == target or p.startswith(target+"/")
                                                            for target in self.floor_targets for p in c["pair"])
                                                       for c in contacts),
                                  "contacts": contacts}
        return rows
