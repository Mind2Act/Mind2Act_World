"""Real CPU USD composition; no private asset or native renderer required."""
from pathlib import Path
import hashlib

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from mind2act_eval.robot import ROOTS, X5Robot


def test_camera_orphan_repair_changes_only_session_render_target():
    from mind2act_eval.robot.appearance import repair_camera_material_binding
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage,'/X5A/camera/visuals/camera/World/mesh').GetPrim()
    target = '/OmniKit_Viewport_LightRig/Looks/Ground_Material'
    mesh.CreateRelationship('material:binding').SetTargets([target])
    mesh.CreateRelationship('material:binding:physics').SetTargets(['/Physics/Original'])
    original = stage.GetRootLayer().ExportToString()
    rows = repair_camera_material_binding(stage,['/X5A'])
    assert len(rows)==1 and rows[0]['effective_material_sha256_before']==rows[0]['effective_material_sha256_after']
    assert mesh.GetRelationship('material:binding').GetTargets()==[]
    assert mesh.GetRelationship('material:binding:physics').GetTargets()==[Sdf.Path('/Physics/Original')]
    assert stage.GetRootLayer().ExportToString()==original
    assert repair_camera_material_binding(stage,['/X5A'])==[]


def test_camera_repair_keeps_valid_source_and_unrelated_missing_targets():
    from mind2act_eval.robot.appearance import repair_camera_material_binding
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage,'/X5A/camera/visuals/camera/World/mesh').GetPrim()
    target = '/OmniKit_Viewport_LightRig/Looks/Ground_Material'
    UsdShade.Material.Define(stage,target)
    rel = mesh.CreateRelationship('material:binding')
    for value in [target,'/OtherMissingMaterial']:
        rel.SetTargets([value])
        assert repair_camera_material_binding(stage,['/X5A'])==[]
        assert rel.GetTargets()==[Sdf.Path(value)]


def scene(tmp_path, monkeypatch):
    asset = tmp_path / 'asset.usda'
    source = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(source, '/X5A')
    source.SetDefaultPrim(root.GetPrim())
    for link in ('link3', 'camera'):
        mesh = UsdGeom.Mesh.Define(source, '/X5A/' + link + '/visuals/mesh')
        # Reproduce a real reference-out-of-scope source binding.
        mesh.GetPrim().CreateRelationship('material:binding').SetTargets(['/Outside/Material'])
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    for index, path in enumerate(ROOTS):
        UsdGeom.Xform.Define(stage, str(Sdf.Path(path).GetParentPath()))
        prim = UsdGeom.Xform.Define(stage, path)
        prim.GetPrim().GetReferences().AddReference(str(asset))
        prim.AddTranslateOp().Set(Gf.Vec3d(index, 2, 3))
    # Two deliberately different authored source materials, not a white override.
    for side, color in [('left', (.8, .7, .6)), ('right', (.1, .3, .9))]:
        material = UsdShade.Material.Define(stage, '/World/Materials/' + side)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for path, side in zip(ROOTS, ('left', 'right')):
        for link in ('link3', 'camera'):
            mesh = stage.GetPrimAtPath(path + '/' + link + '/visuals/mesh')
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                UsdShade.Material(stage.GetPrimAtPath('/World/Materials/' + side)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    monkeypatch.setattr('mind2act_eval.robot.verify_asset', lambda path, expected: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return stage, X5Robot(usd_path=asset), asset


def bindings(stage):
    return {path + '/' + link: str(UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath(path + '/' + link + '/visuals/mesh')).ComputeBoundMaterial()[0].GetPath())
        for path in ROOTS for link in ('link3', 'camera')}


def test_restore_preserves_actual_source_materials_strength_and_mounts(tmp_path, monkeypatch):
    stage, robot, asset = scene(tmp_path, monkeypatch)
    original = bindings(stage)
    source_bytes = asset.read_bytes()
    mounts = {path: UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation() for path in ROOTS}
    restored = robot.restore_asset(stage)
    assert restored["asset_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert robot._asset_sha256 == restored["asset_sha256"]
    assert bindings(stage) == original
    for path in ROOTS:
        assert UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation() == mounts[path]
        for link in ('link3', 'camera'):
            rel = stage.GetPrimAtPath(path + '/' + link + '/visuals/mesh').GetRelationship('material:binding')
            assert rel.GetMetadata('bindMaterialAs') == UsdShade.Tokens.strongerThanDescendants
    assert asset.read_bytes() == source_bytes
    # Idempotent reconstruction must retain each arm's distinct source, not whiten it.
    robot.restore_asset(stage)
    assert bindings(stage) == original


def test_live_receipt_records_before_after_without_claiming_render_acceptance(tmp_path, monkeypatch):
    stage, robot, _ = scene(tmp_path, monkeypatch)
    robot._native_receipt = {'schema': 'native-test-snapshot'}
    robot.restore_asset(stage)
    receipt = robot.native_receipt()['appearance']
    assert receipt['render_validation'] == 'unreviewed'
    assert len(receipt['bindings']) == 4
    assert all(row['before_targets'] == row['after_targets'] for row in receipt['bindings'])
    assert all(row['material_sha256_before'] == row['material_sha256_after'] for row in receipt['bindings'])
    receipt['bindings'][0]['after_targets'].clear()
    assert robot.native_receipt()['appearance']['bindings'][0]['after_targets']


def test_unresolved_source_does_not_invent_id10_or_material(tmp_path, monkeypatch):
    stage, robot, _ = scene(tmp_path, monkeypatch)
    for path in ROOTS:
        for link in ('link3', 'camera'):
            stage.GetPrimAtPath(path + '/' + link + '/visuals/mesh').RemoveProperty('material:binding')
    # These are the asset's ignored, outside-reference targets, not a request
    # to create a universal white material for other tasks.
    robot.restore_asset(stage)
    receipt = robot.native_receipt()['appearance']
    assert receipt['bindings'] == []
    assert receipt['render_validation'] == 'unreviewed'
    assert not stage.GetPrimAtPath('/World/Materials/X5SourceShell')


def test_material_graph_drift_fails_instead_of_silently_changing_finish(tmp_path, monkeypatch):
    from mind2act_eval.robot.appearance import capture_bindings, restore_bindings
    stage, _, asset = scene(tmp_path, monkeypatch)
    captured = capture_bindings(stage, ROOTS)
    shader = UsdShade.Shader(stage.GetPrimAtPath('/World/Materials/left/Shader'))
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(1., 1., 1.))
    with pytest.raises(ValueError, match='material graph changed'):
        restore_bindings(stage, captured, source_asset=asset)


def test_missing_source_target_fails_before_any_binding_write(tmp_path, monkeypatch):
    from mind2act_eval.robot.appearance import capture_bindings, restore_bindings
    stage, _, asset = scene(tmp_path, monkeypatch)
    captured = capture_bindings(stage, ROOTS)
    for path in ROOTS:
        for link in ('link3', 'camera'):
            stage.GetPrimAtPath(path + '/' + link + '/visuals/mesh').RemoveProperty('material:binding')
    stage.RemovePrim('/World/Materials/right')
    with pytest.raises(ValueError, match='target did not survive'):
        restore_bindings(stage, captured, source_asset=asset)
    # Preflight must not partially restore the other arm's valid source.
    assert stage.GetPrimAtPath(ROOTS[0] + '/link3/visuals/mesh').GetRelationship('material:binding').GetTargets() == []


def test_physics_material_binding_is_outside_this_render_operation(tmp_path, monkeypatch):
    from mind2act_eval.robot.appearance import capture_bindings
    stage, _, _ = scene(tmp_path, monkeypatch)
    mesh = stage.GetPrimAtPath(ROOTS[0] + '/link3/visuals/mesh')
    mesh.CreateRelationship('material:binding:physics').SetTargets(['/World/Materials/left'])
    assert all(not row['relationship'].endswith(':physics') for row in capture_bindings(stage, ROOTS)['bindings'])


def test_existing_episode_writer_gets_current_source_binding_receipt(tmp_path, monkeypatch):
    import json
    from test_runner import fixture
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.contracts import ConsumerKind
    from mind2act_eval.runner import EpisodeLimits, run_episode
    stage, robot, _ = scene(tmp_path, monkeypatch)
    spec, task, ctx, _ = fixture(tmp_path)
    # Use the actual X5 getter and existing episode writer; simulation/action
    # transport stays explicitly CPU demo, not a native robot claim.
    ctx.robot.native_receipt = robot.native_receipt
    robot.restore_asset(stage)
    result = run_episode(spec, task, ctx, DemoAdapter(), consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(1, 10))
    assert result.infra_valid, result.error
    stored = json.loads((tmp_path / 'robot-receipt.json').read_text())['appearance']
    assert len(stored['bindings']) == 4
    assert stored['render_validation'] == 'unreviewed'
    assert stored['source_asset_sha256']


def test_all_purpose_physics_material_cannot_enter_render_restoration(tmp_path, monkeypatch):
    from pxr import UsdPhysics
    from mind2act_eval.robot.appearance import capture_bindings
    stage, _, _ = scene(tmp_path, monkeypatch)
    material = stage.GetPrimAtPath('/World/Materials/left')
    UsdPhysics.MaterialAPI.Apply(material).CreateStaticFrictionAttr(.7)
    captured = capture_bindings(stage, ROOTS)
    assert all(row['material'] != '/World/Materials/left' for row in captured['bindings'])
    assert len(captured['bindings']) == 2
    assert len(captured['unresolved_before']) == 2
    assert all('physics material excluded' in row['reason'] for row in captured['unresolved_before'])


def test_shader_outside_material_removed_by_robot_restore_is_rejected(tmp_path, monkeypatch):
    stage, robot, _ = scene(tmp_path, monkeypatch)
    shader = UsdShade.Shader.Define(stage, ROOTS[0] + '/SourceShader')
    shader.CreateIdAttr('UsdPreviewSurface')
    material = UsdShade.Material(stage.GetPrimAtPath('/World/Materials/left'))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    with pytest.raises(ValueError, match='material graph changed'):
        robot.restore_asset(stage)


def test_surviving_external_shader_inputs_are_part_of_the_digest(tmp_path, monkeypatch):
    from mind2act_eval.robot.appearance import capture_bindings, restore_bindings
    stage, robot, asset = scene(tmp_path, monkeypatch)
    shader = UsdShade.Shader.Define(stage, '/World/SourceShaders/Shell')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.4, .5, .6))
    material = UsdShade.Material(stage.GetPrimAtPath('/World/Materials/left'))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    # The external graph survives a normal restore and must not be recolored.
    robot.restore_asset(stage)
    assert shader.GetInput('diffuseColor').Get() == Gf.Vec3f(.4, .5, .6)
    captured = capture_bindings(stage, ROOTS)
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(.9, .1, .2))
    with pytest.raises(ValueError, match='material graph changed'):
        restore_bindings(stage, captured, source_asset=asset)


def test_external_nodegraph_forwarding_cycles_are_bounded_and_deterministic(tmp_path, monkeypatch):
    from mind2act_eval.robot.appearance import capture_bindings
    stage, _, _ = scene(tmp_path, monkeypatch)
    first = UsdShade.NodeGraph.Define(stage, '/World/SourceGraphs/First')
    second = UsdShade.NodeGraph.Define(stage, '/World/SourceGraphs/Second')
    a = first.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    b = second.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    a.ConnectToSource(b); b.ConnectToSource(a)
    material = UsdShade.Material(stage.GetPrimAtPath('/World/Materials/left'))
    material.CreateSurfaceOutput().ConnectToSource(a)
    assert capture_bindings(stage, ROOTS) == capture_bindings(stage, ROOTS)
