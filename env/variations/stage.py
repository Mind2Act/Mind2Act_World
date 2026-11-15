"""Restricted USD authoring. No World, solver, stepping, camera or random API."""
from __future__ import annotations

import math
from pathlib import Path

from . import digest, file_digest


def _under(path, root):
    return path == root or path.startswith(root + "/")


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return tuple(_stable(v) for v in value)
    # str() of an individual USD value is semantic; repr() of a dict of USD
    # values can include Python wrapper addresses, which are not stage facts.
    return str(value)


def _physics_material_path(prim):
    from pxr import UsdPhysics, UsdShade
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")[0]
    if not material:
        return None
    p = material.GetPrim()
    physical = p.HasAPI(UsdPhysics.MaterialAPI) or any(
        schema.startswith("Physx") and "Material" in schema for schema in p.GetAppliedSchemas())
    # A render-only allPurpose fallback is not a PhysicsMaterial. Its visual
    # binding may change, while every real physics material stays protected.
    return str(p.GetPath()) if physical else None


def _snapshot(stage, targets):
    """All pre-existing properties except granted visual/light opinions."""
    from pxr import Usd, UsdShade
    surfaces = (*targets.table_surfaces, *targets.floor_surfaces)
    result = {"stage_metadata": _stable(stage.GetPseudoRoot().GetAllMetadata())}
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if _under(path, targets.clutter_root):
            continue
        surface = any(_under(path, root) for root in surfaces)
        light = path in (targets.key_light, targets.dome_light)
        metadata = dict(prim.GetAllAuthoredMetadata() if light else prim.GetAllMetadata())
        if light:
            metadata.pop("typeName", None)
        if surface:
            schemas = prim.GetAppliedSchemas()
            metadata["apiSchemas"] = tuple(s for s in schemas if s != "MaterialBindingAPI")
        props = {}
        for prop in prim.GetAuthoredProperties() if light else prim.GetProperties():
            name = prop.GetName()
            if (surface and name in ("material:binding", "material:binding:preview", "material:binding:full")) or (light and name in ("inputs:color", "inputs:intensity")):
                continue
            if isinstance(prop, Usd.Attribute):
                prop_metadata = prop.GetAllAuthoredMetadata() if light else prop.GetAllMetadata()
                props[name] = (_stable(prop_metadata), str(prop.Get()),
                               [(t, str(prop.Get(t))) for t in prop.GetTimeSamples()],
                               tuple(map(str, prop.GetConnections())))
            else:
                props[name] = (_stable(prop.GetAllMetadata()), tuple(map(str, prop.GetTargets())))
        # A visual binding must never displace an inherited physics binding.
        result[path] = (_stable(metadata), props, _physics_material_path(prim))
    return result


def _check_pin(pin):
    if file_digest(pin.path) != pin.sha256:
        raise ValueError("variation asset changed between resolve and apply")
    for path, expected in pin.dependencies:
        if file_digest(path) != expected:
            raise ValueError("variation dependency changed between resolve and apply")


def _asset_stage(pin):
    from pxr import Ar
    context = Ar.ResolverContext(Ar.DefaultResolverContext([str(p) for p in pin.search_paths]))
    with Ar.ResolverContextBinder(context):
        return _inspect_asset_stage(pin, context)


def _inspect_asset_stage(pin, context):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdUtils
    _check_pin(pin)
    stage = Usd.Stage.Open(str(pin.path), context)
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError("distractor must have an explicit USD default prim")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(pin.path))
    if unresolved:
        raise ValueError("unresolved distractor dependencies: " + str(unresolved))
    pinned = {pin.path.resolve(), *(p.resolve() for p, _ in pin.dependencies)}
    # Kit can return its SDK installation path for a module hard-linked into
    # the pinned closure. The checked physical file, rather than its spelling,
    # is the capability. Equal bytes in a different file remain unpinned.
    identities = {(s.st_dev, s.st_ino) for p in pinned for s in (p.stat(),)}
    # Dependencies inside a USDZ are covered by the outer file's content pin.
    actual = [layer.realPath for layer in layers] + list(map(str, assets))
    from .mdl_closure import sdk_aliases
    aliases = sdk_aliases(pin, layers, actual)
    for value in actual:
        outer = Path(value.split("[", 1)[0]).resolve()
        outer = aliases.get(outer, outer)
        if outer not in pinned:
            actual_stat = outer.stat()
            if (actual_stat.st_dev, actual_stat.st_ino) not in identities:
                raise ValueError("distractor has an unpinned dependency: " + str(outer))
    root = stage.GetDefaultPrim()
    # AddReference below composes only this default-prim subtree. Exported Kit
    # viewport/render siblings are not imported and grant no stage capability.
    imported = tuple(Usd.PrimRange.AllPrims(root))
    for prim in imported:
        if (prim.IsA(UsdGeom.Camera) or prim.IsA(UsdPhysics.Joint)
                or prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                or prim.HasAPI(UsdLux.LightAPI)):
            raise ValueError("distractor contains a camera/light/joint/articulation/physics scene")
        for prop in prim.GetProperties():
            if hasattr(prop, "GetTimeSamples") and prop.GetTimeSamples():
                raise ValueError("distractor contains animation")
            paths = prop.GetConnections() if hasattr(prop, "GetConnections") else prop.GetTargets()
            if any(not p.GetPrimPath().HasPrefix(root.GetPath()) for p in paths):
                raise ValueError("distractor relationship escapes its default prim")
    # Runtime effects and imported executable schemas are outside this module.
    for prim in imported:
        if any(s.startswith(("OmniScripting", "PhysxVehicle")) for s in prim.GetAppliedSchemas()):
            raise ValueError("unsupported distractor runtime schema")
    return stage


def _check_exclusions(stage, prim, excluded):
    from itertools import product
    from pxr import Gf, Usd, UsdGeom
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(prim)
    box = bound.GetRange()
    if box.IsEmpty():
        raise ValueError("distractor has no finite geometry bounds")
    corners = [tuple(bound.GetMatrix().Transform(Gf.Vec3d(*p))) for p in product(*zip(box.GetMin(), box.GetMax()))]
    if not all(math.isfinite(x) for p in corners for x in p):
        raise ValueError("invalid distractor geometry bounds")
    # The delivery uses projected footprints, not a rotated object's larger
    # world AABB. A convex hull + separating axes retains those clear corners.
    points = sorted(set((p[0], p[1]) for p in corners))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def half(values):
        result = []
        for point in values:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result
    polygon = half(points)[:-1] + half(reversed(points))[:-1]
    if len(polygon) < 3:
        raise ValueError("distractor has no finite projected footprint")
    axes = [(1., 0.), (0., 1.)] + [(b[1]-a[1], a[0]-b[0]) for a,b in zip(polygon, polygon[1:]+polygon[:1])]
    warnings = []
    for index, (a, b) in enumerate(excluded):
        if max(p[2] for p in corners) < a[2] or min(p[2] for p in corners) > b[2]:
            continue
        rectangle = [(a[0],a[1]),(a[0],b[1]),(b[0],a[1]),(b[0],b[1])]
        def separated(axis):
            left = [sum(x*y for x,y in zip(p,axis)) for p in polygon]
            right = [sum(x*y for x,y in zip(p,axis)) for p in rectangle]
            return max(left) < min(right) or max(right) < min(left)
        if not any(separated(axis) for axis in axes):
            warnings.append({"kind": "source_exclusion_overlap", "zone_index": index,
                             "severity": "warning", "geometry_intersection_established": False})
    return warnings


def _material(stage, root, name, pin):
    from pxr import Sdf, UsdShade
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    _check_pin(pin)
    material = UsdShade.Material.Define(stage, root + "/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
    shader.CreateImplementationSourceAttr("sourceAsset")
    shader.SetSourceAsset(Sdf.AssetPath(relative_asset_path(stage.GetEditTarget().GetLayer(), pin.path)), "mdl")
    shader.SetSourceAssetSubIdentifier(pin.metadata["mdl_name"], "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _spawn(stage, plan, targets):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from mind2act_eval.sim.asset_relocation import relative_asset_path
    rows = []
    for item in plan.layout.objects:
        pin = plan.assets[item.asset_key]
        _asset_stage(pin)
        path = targets.clutter_root + "/" + item.name
        prim = stage.DefinePrim(path)
        reference = relative_asset_path(stage.GetEditTarget().GetLayer(), pin.path)
        if not prim.GetReferences().AddReference(reference):
            raise ValueError("cannot compose pinned distractor")
        from .mdl_closure import localize_bindings
        localize_bindings(prim, pin, stage.GetEditTarget().GetLayer())
        xform = UsdGeom.Xformable(prim)
        if not xform:
            raise ValueError("distractor default prim is not transformable")
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(True)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "variation").Set(Gf.Vec3d(*item.position))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "variation").Set(Gf.Quatd(item.orientation[0], Gf.Vec3d(*item.orientation[1:])))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "variation").Set(Gf.Vec3d(*item.scale))
        bodies = [p for p in Usd.PrimRange(xform.GetPrim()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1 or bodies[0] != xform.GetPrim():
            raise ValueError("requires one baked rigid body at distractor default prim")
        body = UsdPhysics.RigidBodyAPI(bodies[0])
        if body.GetKinematicEnabledAttr().Get() or body.GetRigidBodyEnabledAttr().Get() is False:
            raise ValueError("distractor baked rigid body is disabled or kinematic")
        # Only this new distractor receives the effective archived mass and rest
        # velocity. Inherited PhysicsMaterial and friction opinions stay intact.
        if "mass" in item.physics:
            UsdPhysics.MassAPI.Apply(bodies[0]).CreateMassAttr(float(item.physics["mass"]))
        body.CreateVelocityAttr(Gf.Vec3f(0))
        body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        warnings = _check_exclusions(stage, xform.GetPrim(), plan.binding["forbidden_aabbs"])
        rows.append({"path": path, "asset_sha256": pin.sha256,
                     "position": item.position, "orientation_wxyz": item.orientation,
                     "physics": dict(item.physics), "warnings": warnings})
    return rows


def _visual_prims(stage, path):
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(path)
    if not root:
        raise ValueError("missing explicitly granted background surface: " + path)
    return [p for p in Usd.PrimRange(root)
            if p == root or p.IsA(UsdGeom.Gprim) or p.IsA(UsdGeom.Subset)]


def _appearance(stage, plan, targets):
    from pxr import Gf, UsdLux, UsdShade
    layout, root = plan.layout, targets.clutter_root
    bindings, skipped = [], []
    roles = [("floor", layout.ground, targets.floor_surfaces)]
    if layout.appearance["table_enabled"]:
        roles.append(("table", layout.appearance["material"]["id"], targets.table_surfaces))
    for role, key, paths in roles:
        if not paths:
            if role == "table":
                skipped.append({"role": "table", "reason": "builder declares no table in the clean scene"})
                continue
            raise ValueError("builder did not grant " + role + " visual targets")
        material = _material(stage, root, role, plan.assets[key])
        for path in paths:
            for prim in _visual_prims(stage, path):
                api = UsdShade.MaterialBindingAPI.Apply(prim)
                # RTX consumes the default material purpose. full/preview alone
                # leave its original binding untouched, even on a leaf Mesh.
                for purpose in ("", "preview", "full"):
                    api.Bind(material, materialPurpose=purpose)
                    bindings.append({"target": str(prim.GetPath()), "purpose": purpose,
                        "material_path": str(material.GetPath()), "asset_sha256": plan.assets[key].sha256})
    light_row = None
    if layout.appearance["light_enabled"]:
        data = layout.appearance["light"]
        path = targets.dome_light if data["type"] == "Dome" else targets.key_light
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdLux.LightAPI):
            raise ValueError("missing explicitly granted authored light: " + path)
        intensity = UsdLux.LightAPI(prim).GetIntensityAttr().Get()
        if intensity is None or not math.isfinite(intensity):
            raise ValueError("authored light has no finite baseline intensity")
        light = getattr(UsdLux, data["type"] + "Light").Define(stage, path)
        light.CreateColorAttr(Gf.Vec3f(*data["color_rgb"]))
        light.CreateIntensityAttr(float(intensity * data["intensity_scale"]))
        light_row = {"target": path, "type": data["type"], "color_rgb": tuple(data["color_rgb"]),
                     "baseline_intensity": intensity, "intensity_scale": data["intensity_scale"]}
    return {"visual_bindings": bindings, "skipped_visuals": skipped, "light": light_row}


def verify_appearance(stage, targets, receipt):
    """Check effective post-init opinions, including zero-distractor layouts."""
    from pxr import Gf, Sdf, UsdLux, UsdShade
    layer_path = Path(targets.runtime_layer).resolve()
    if not any(layer.realPath and Path(layer.realPath).resolve() == layer_path
               for layer in stage.GetLayerStack()):
        raise ValueError("variation layer no longer composed after initialization")
    rows = []
    for binding in receipt["visual_bindings"]:
        prim = stage.GetPrimAtPath(binding["target"])
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
            materialPurpose=binding["purpose"])[0] if prim else None
        actual = str(material.GetPath()) if material else None
        if actual != binding["material_path"]:
            raise ValueError("variation visual binding not effective after initialization: " + binding["target"])
        rows.append({"target": binding["target"], "purpose": binding["purpose"], "material_path": actual})
    light_readback = None
    if receipt["light"]:
        expected = receipt["light"]
        prim = stage.GetPrimAtPath(expected["target"])
        if not prim or prim.GetTypeName() != expected["type"] + "Light":
            raise ValueError("variation visual light type lost after initialization")
        light = UsdLux.LightAPI(prim)
        intensity = expected["baseline_intensity"] * expected["intensity_scale"]
        if light.GetIntensityAttr().GetTypeName() == Sdf.ValueTypeNames.Float:
            intensity = float(Gf.Vec3f(intensity)[0])  # exact authored float storage
        color = Gf.Vec3f(*expected["color_rgb"])
        if light.GetIntensityAttr().Get() != intensity or light.GetColorAttr().Get() != color:
            raise ValueError("variation visual light opinions lost after initialization")
        light_readback = {"target": expected["target"], "type": prim.GetTypeName(),
                          "color_rgb": tuple(color), "intensity": intensity}
    return {"status": "verified", "layer_composed": True, "effective_bindings": rows,
            "light": light_readback, "readback": "post_initialization_usd", "pixels_verified": False}


def apply(plan, stage, targets):
    from dataclasses import replace
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
    from mind2act_eval.sim.asset_relocation import audit_generated_layers
    if plan.spec.mode == "clean":
        return {"schema": "mind2act_eval.variation_receipt.v1", "mode": "clean", "plan_sha256": plan.sha256,
                "authored": False, "physics_steps_added": 0}
    if stage.GetPrimAtPath(targets.clutter_root):
        raise ValueError("variation namespace already exists; apply is once-only")
    for prim in stage.TraverseAll():
        p = str(prim.GetPath())
        for root in (*targets.table_surfaces, *targets.floor_surfaces):
            if _under(p, root) and (prim.IsA(UsdGeom.Camera) or prim.HasAPI(UsdLux.LightAPI)
                    or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint)):
                raise ValueError("visual capability includes a protected dynamic/camera/light prim")
    before = _snapshot(stage, targets)
    if targets.runtime_layer is None:
        raise ValueError("random variation requires a builder-owned runtime layer path")
    path = Path(targets.runtime_layer).resolve()
    if path.suffix != ".usda":
        raise ValueError("variation runtime layer must be a loose .usda file")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(path))
    session = stage.GetSessionLayer()
    original = list(session.subLayerPaths)
    session.subLayerPaths = [layer.identifier, *original]
    try:
        with Usd.EditContext(stage, layer):
            UsdGeom.Xform.Define(stage, targets.clutter_root).SetResetXformStack(True)
            clutter = _spawn(stage, plan, targets)
            appearance = _appearance(stage, plan, targets)
        after = _snapshot(stage, targets)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        if changed:
            raise ValueError("variation modified protected stage properties: " + ", ".join(changed[:12]))
        layer.Save()
        portability = audit_generated_layers([path])
        if not portability["passed"]:
            raise ValueError("nonportable variation runtime layer: " + str(portability["issues"]))
    except BaseException:
        session.subLayerPaths = original
        # Keep this failed attempt's authored layer for diagnosis, detached from
        # the live stage. Never overwrite it on a retry.
        layer.Save()
        raise
    return {"schema": "mind2act_eval.variation_receipt.v1", "mode": "random",
            "plan_sha256": plan.sha256, "variation": plan.spec, "base_spec_sha256": plan.instance_sha256,
            "composition_sha256": digest(layer.ExportToString()), "protected_before_sha256": digest(before),
            "protected_after_sha256": digest(after), "protected_diff": [],
            "targets": replace(targets, runtime_layer=None), "runtime_layer": path.name,
            "generated_layer_portability": {k: v for k, v in portability.items() if k != "issues"},
            "settled_provenance": plan.layout.provenance, "clutter": clutter, **appearance,
            "physics_steps_added": 0, "source_warmup": "unchanged builder initialization"}


def verify_settled(plan, stage, targets, *, native=False):
    from pxr import UsdGeom, Usd
    position_tolerance = plan.binding["initialization_pose_tolerance_m"]
    angle_tolerance = plan.binding["initialization_pose_tolerance_rad"]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in (position_tolerance, angle_tolerance)):
        raise ValueError("requires frozen source initialization pose tolerances")
    rows = []
    for item in plan.layout.objects:
        path = targets.clutter_root + "/" + item.name
        if native:
            # Direct PhysX tensor reads: no SDK prim registration/reset/sensor writes.
            from isaacsim.core.simulation_manager import SimulationManager
            view = SimulationManager.get_physics_sim_view().create_rigid_body_view([path])
            if view.count != 1:
                raise ValueError("native settled body missing: " + path)
            values = view.get_transforms()
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            elif hasattr(values, "numpy"):
                values = values.numpy()
            value = values[0]
            position, orientation = value[:3], value[[6, 3, 4, 5]]
        else:
            transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            position = tuple(transform.ExtractTranslation())
            quat = transform.ExtractRotationQuat()
            orientation = (quat.GetReal(), *quat.GetImaginary())
        delta = math.dist(position, item.position)
        dot = abs(sum(a*b for a, b in zip(orientation, item.orientation)))
        angle = 2 * math.acos(min(1., max(0., dot)))
        # Frozen admission tolerance; never expanded after a failed sample.
        if delta > position_tolerance or angle > angle_tolerance:
            raise ValueError("settled pose changed during original task initialization: " + path)
        rows.append({"path": path, "position_error_m": delta, "angle_error_rad": angle})
    return {"status": "verified", "readback": "native_physx" if native else "composed_usd",
            "position_tolerance_m": position_tolerance, "angle_tolerance_rad": angle_tolerance, "objects": rows,
            "physics_steps_added": 0}
