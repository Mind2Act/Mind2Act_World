"""Preserve composed source render bindings across canonical robot restoration.

No material, shader, color, geometry or physics value is invented here. A valid
source material must survive with the same composed shader graph; otherwise
restoration fails instead of silently substituting a finish. Unresolved source
bindings remain explicitly unresolved, except the known out-of-scope viewport
binding on the shared camera, cleared with an explicit render-only receipt.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

# Investigation provenance, not a claim that this capture helper executes here.
PR1_CAPTURE_HELPER_SHA256 = 'f78940b9c7bbea66ba76b57bc14ce82125ae539afdb2767ba2c70c7d3202f16a'


def repair_camera_material_binding(stage, roots):
    """Clear only the shared camera's nonexistent viewport helper target.

    Author in the session layer; the pinned asset and physics bindings stay
    intact. A valid source target or any other unresolved target is untouched.
    This cannot substitute a finish: effective material must remain identical.
    """
    from pxr import Sdf, Usd, UsdShade
    target = Sdf.Path('/OmniKit_Viewport_LightRig/Looks/Ground_Material')
    repairs = []
    for root in roots:
        mesh = stage.GetPrimAtPath(str(root) + '/camera/visuals/camera/World/mesh')
        if not mesh or stage.GetPrimAtPath(target):
            continue
        rel = mesh.GetRelationship('material:binding')
        if not rel or rel.GetTargets() != [target]:
            continue
        material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
        before = _material_signature(material) if material else None
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            override = mesh.CreateRelationship('material:binding', custom=False)
            override.SetTargets([])
            material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
            after = _material_signature(material) if material else None
            if after != before:
                override.SetTargets([target])
                raise ValueError('camera orphan repair would change effective source material')
        repairs.append(dict(relationship=str(rel.GetPath()), before_targets=[str(target)],
                            after_targets=[], operation='clear nonexistent shared viewport target in session',
                            effective_material_sha256_before=before, effective_material_sha256_after=after))
    return repairs


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _material_signature(material):
    from pxr import Usd
    stage = material.GetPrim().GetStage()
    pending = [str(prim.GetPath()) for prim in Usd.PrimRange(material.GetPrim())]
    visited, graph = set(), []
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        prim = stage.GetPrimAtPath(path)
        if not prim:
            graph.append((path, 'missing_connected_prim', []))
            continue
        attributes = []
        for attr in prim.GetAttributes():
            connections = attr.GetConnections()
            attributes.append((attr.GetName(), str(attr.GetTypeName()), str(attr.Get()),
                               [str(target) for target in connections]))
            pending.extend(str(target.GetPrimPath()) for target in connections)
        graph.append((path, prim.GetTypeName(), attributes))
    # A shared shader may live outside the Material subtree; forwarding and
    # cycles must not hide a removed node or an altered source input.
    graph.sort(key=lambda row: row[0])
    return hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()


def capture_bindings(stage, roots):
    """Snapshot effective render relationships while the prepared source exists."""
    from pxr import Usd, UsdShade, UsdPhysics
    bindings, unresolved, file_hashes = [], [], {}
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        if not prim:
            raise ValueError('source robot root missing: ' + root)
        for owner in Usd.PrimRange(prim):
            for rel in owner.GetRelationships():
                name = rel.GetName()
                if not (name == 'material:binding' or name.startswith('material:binding:')):
                    continue
                if name.endswith(':physics'):
                    continue
                targets = rel.GetTargets()
                materials = [UsdShade.Material(stage.GetPrimAtPath(path)) for path in targets
                             if path.IsPrimPath() and stage.GetPrimAtPath(path).IsA(UsdShade.Material)]
                if len(materials) != 1:
                    unresolved.append(dict(relationship=str(rel.GetPath()), targets=[str(p) for p in targets],
                                           reason='no unique valid source render material'))
                    continue
                material = materials[0]
                if material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                    unresolved.append(dict(relationship=str(rel.GetPath()), targets=[str(p) for p in targets],
                                           reason='physics material excluded from render-only restoration'))
                    continue
                sources = []
                for spec in material.GetPrim().GetPrimStack():
                    layer = spec.layer
                    path = Path(layer.realPath) if layer.realPath else None
                    if path and path.is_file() and str(path) not in file_hashes:
                        file_hashes[str(path)] = _sha(path)
                    sources.append(dict(layer=layer.identifier, prim=str(spec.path),
                                        file_sha256=file_hashes.get(str(path)) if path else None))
                bindings.append(dict(relationship=str(rel.GetPath()),
                    before_targets=[str(p) for p in targets], material=str(material.GetPath()),
                    strength=str(UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel)),
                    material_sha256_before=_material_signature(material), source_layers=sources))
    return dict(bindings=bindings, unresolved_before=unresolved)


def restore_bindings(stage, snapshot, *, source_asset):
    """Reapply original targets only after all surviving materials are checked."""
    from pxr import Sdf, UsdShade
    rows = deepcopy(snapshot['bindings'])
    for row in rows:
        relationship = Sdf.Path(row['relationship'])
        owner = stage.GetPrimAtPath(relationship.GetPrimPath())
        material = UsdShade.Material(stage.GetPrimAtPath(row['material']))
        if not owner or not material:
            raise ValueError('source appearance target did not survive robot restore: ' + row['relationship'])
        if _material_signature(material) != row['material_sha256_before']:
            raise ValueError('source material graph changed during robot restore: ' + row['material'])
        for target in row['before_targets']:
            path = Sdf.Path(target)
            if not stage.GetObjectAtPath(path):
                raise ValueError('source binding target did not survive robot restore: ' + target)
    for row in rows:
        path = Sdf.Path(row['relationship'])
        owner = stage.GetPrimAtPath(path.GetPrimPath())
        UsdShade.MaterialBindingAPI.Apply(owner)
        rel = owner.CreateRelationship(path.name, custom=False)
        rel.SetTargets([Sdf.Path(target) for target in row['before_targets']])
        rel.SetMetadata('bindMaterialAs', row['strength'])
        row['after_targets'] = [str(target) for target in rel.GetTargets()]
        material = UsdShade.Material(stage.GetPrimAtPath(row['material']))
        row['material_sha256_after'] = _material_signature(material)
    return dict(schema='mind2act_eval.robot_source_appearance.v1',
        operation='retain original source render bindings; no shader authoring',
        material_digest_scope='material descendants and recursively connected prims',
        render_validation='unreviewed', source_asset=str(Path(source_asset).resolve()),
        source_asset_sha256=_sha(source_asset), implementation_sha256=_sha(__file__),
        pr1_capture_reference_helper_sha256=PR1_CAPTURE_HELPER_SHA256,
        capture_reference_scope='PR1 audit provenance only; no universal material override',
        bindings=rows, unresolved_before=deepcopy(snapshot['unresolved_before']))
