"""Retain only the four acquisition pads across official X5 restore.

The station projection removes /World/CP09, including its two physics materials.
Those two exact source declarations are reconstructed; render materials must
already exist and survive unchanged. No task drive/solver opinions are copied.
"""
import hashlib

from .task_collision_overlay import expected_descriptor, material_paths, validate_actual


def define_physics_material(stage, path):
    from pxr import UsdPhysics, UsdShade
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(2.)
    api.CreateDynamicFrictionAttr(1.8)
    api.CreateRestitutionAttr(0.)
    return material


def capture(stage, spec):
    if spec is None or spec.task_id != 'CP09':
        return None
    from pxr import Sdf, Usd
    descriptor = expected_descriptor()
    # Work on an isolated composed copy: malformed/missing pads or bindings
    # must fail before any live-stage replacement or material authoring.
    composed = stage.Flatten()
    source = Usd.Stage.Open(composed)
    projected_materials = []
    for side in ('Left', 'Right'):
        physics, _ = material_paths(side)
        if not source.GetPrimAtPath(physics):
            define_physics_material(source, physics)
            projected_materials.append(physics)
    validate_actual(source, descriptor)
    layer = Sdf.Layer.CreateAnonymous('CP09-source-pads.usda')
    paths = [row['path'] for row in descriptor['pads']] + projected_materials
    for path in paths:
        Sdf.CreatePrimInLayer(layer, Sdf.Path(path).GetParentPath())
        if not Sdf.CopySpec(composed, path, layer, path):
            raise ValueError('cannot capture exact CP09 source pad: ' + path)
    return dict(layer=layer, descriptor=descriptor, paths=paths,
                projected_materials=projected_materials)


def restore(stage, captured):
    from pxr import Sdf
    paths, layer = captured['paths'], captured['layer']
    for row in captured['descriptor']['pads']:
        path = Sdf.Path(row['path'])
        if stage.GetPrimAtPath(path) or not stage.GetPrimAtPath(path.GetParentPath()):
            raise ValueError('CP09 official replacement has unexpected pad/parent: ' + str(path))
    for path in paths:
        if stage.GetPrimAtPath(path):
            raise ValueError('CP09 projected material unexpectedly occupied: ' + path)
    for path in paths:
        Sdf.CreatePrimInLayer(stage.GetEditTarget().GetLayer(), Sdf.Path(path).GetParentPath())
        if not Sdf.CopySpec(layer, path, stage.GetEditTarget().GetLayer(), path):
            raise ValueError('cannot restore exact CP09 source pad: ' + path)
    validate_actual(stage, captured['descriptor'])
    return dict(schema='mind2act.cp09-source-pad-binding.v2',
                source_sha256=captured['descriptor']['source_sha256'],
                acquisition_version=captured['descriptor']['acquisition_version'],
                geometry_asset_sha256=captured['descriptor']['geometry_asset_sha256'],
                geometry_sha256=captured['descriptor']['geometry_sha256'],
                restored_pad_paths=[r['path'] for r in captured['descriptor']['pads']],
                reconstructed_projected_physics_materials=captured['projected_materials'],
                captured_usd_sha256=hashlib.sha256(layer.ExportToString().encode()).hexdigest(),
                operation='restore exact source pads before strict appearance restoration')
