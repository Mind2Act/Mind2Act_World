"""Real USD restore/prepare lifecycle; usd-core records PhysX metadata only.

The four-pad fixture was emitted by the SHA-bound acquisition function, not
the candidate overlay. The official robot fixture supplies only links/joints;
this is a CPU source-hierarchy regression, not a native/visual certificate.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math

import pxr
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import pytest

from mind2act_eval.robot import ROOTS, PROFILE, X5Robot
from mind2act_eval.robot.appearance import capture_bindings
from mind2act_eval.robot.task_collision_overlay import expected_descriptor, snapshot
from test_cp09_robot_overlay import selected


@pytest.fixture(autouse=True)
def physx_metadata(monkeypatch):
    def api(namespace, token):
        class API:
            _schema_token = token
            def __init__(self, prim): self.prim = prim
            @classmethod
            def Apply(cls, prim):
                schemas = list(prim.GetMetadata('apiSchemas').GetAppliedItems()) if prim.GetMetadata('apiSchemas') else []
                prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(schemas + [cls._schema_token]))
                return cls(prim)
            def __getattr__(self, name):
                key = name[6:-4] if name.startswith('Create') else name[3:-4]
                key = namespace + ':' + key[:1].lower() + key[1:]
                if name.startswith('Get'): return lambda: self.prim.GetAttribute(key)
                def create(value):
                    typ = Sdf.ValueTypeNames.Bool if type(value) is bool else (
                        Sdf.ValueTypeNames.Int if type(value) is int else Sdf.ValueTypeNames.Float)
                    attr = self.prim.CreateAttribute(key, typ, custom=False); attr.Set(value); return attr
                return create
        return API
    schema = SimpleNamespace(**{name: api(namespace, name) for name, namespace in (
        ('PhysxCollisionAPI', 'physxCollision'), ('PhysxContactReportAPI', 'physxContactReport'),
        ('PhysxArticulationAPI', 'physxArticulation'), ('PhysxRigidBodyAPI', 'physxRigidBody'),
        ('PhysxJointAPI', 'physxJoint'))})
    original_has, original_remove = Usd.Prim.HasAPI, Usd.Prim.RemoveAPI
    monkeypatch.setattr(Usd.Prim, 'HasAPI', lambda prim, cls: (
        cls._schema_token in prim.GetMetadata('apiSchemas').GetAppliedItems()
        if hasattr(cls, '_schema_token') and prim.GetMetadata('apiSchemas') else
        False if hasattr(cls, '_schema_token') else original_has(prim, cls)))
    monkeypatch.setattr(Usd.Prim, 'RemoveAPI', lambda prim, cls: (
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([
            t for t in prim.GetMetadata('apiSchemas').GetAppliedItems() if t != cls._schema_token]))
        if hasattr(cls, '_schema_token') else original_remove(prim, cls)))
    monkeypatch.setattr(pxr, 'PhysxSchema', schema, raising=False)


def source_stage(tmp_path, monkeypatch, difficulty='easy', *, projected=True):
    asset = tmp_path / 'official.usda'
    official = Usd.Stage.CreateNew(str(asset))
    official.SetDefaultPrim(UsdGeom.Xform.Define(official, '/X5').GetPrim())
    UsdGeom.Xform.Define(official, '/X5/base_link')
    for i in range(1, 9):
        UsdGeom.Xform.Define(official, '/X5/link' + str(i))
        (UsdPhysics.RevoluteJoint if i <= 6 else UsdPhysics.PrismaticJoint).Define(official, '/X5/joint' + str(i))
    official.GetRootLayer().Save()
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(Sdf.Layer.FindOrOpen(str(Path(__file__).parent / 'fixtures/cp09_source_pads.usda')))
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    for root in ROOTS:
        stage.GetPrimAtPath(root).GetReferences().AddReference(str(asset))
    if projected: stage.RemovePrim('/World/CP09')  # Actual station projection removes this subtree.
    monkeypatch.setattr('mind2act_eval.robot.verify_asset', lambda *_: None)  # Fixture asset only.
    robot = X5Robot(usd_path=asset)
    spec = selected(difficulty); robot.bind_task_instance(spec, tmp_path)
    return stage, robot, spec


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_actual_prepare_restores_four_source_pads_before_strict_appearance(tmp_path, monkeypatch, difficulty):
    stage, robot, spec = source_stage(tmp_path, monkeypatch, difficulty)
    before = capture_bindings(stage, ROOTS)
    pads_before = {p: row for p, row in snapshot(stage).items() if p.endswith('/CP09RubberPad')}
    assert len(before['bindings']) == 4 and len(pads_before) == 4
    robot.prepare_stage(stage)
    assert {p: snapshot(stage)[p] for p in pads_before} == pads_before
    appearance = robot.native_receipt()['appearance']
    source_receipt = robot.native_receipt()['source_pad_binding']
    assert source_receipt['source_sha256'] == expected_descriptor(spec)['source_sha256']
    assert len(source_receipt['restored_pad_paths']) == 4
    assert source_receipt['reconstructed_projected_physics_materials'] == [
        '/World/CP09/GripMaterialLeft', '/World/CP09/GripMaterialRight']
    assert len(appearance['bindings']) == 4
    assert all(r['before_targets'] == r['after_targets'] and
               r['material_sha256_before'] == r['material_sha256_after'] for r in appearance['bindings'])
    for root in ROOTS:
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(root + '/joint1'), 'angular')
        assert drive.GetStiffnessAttr().Get() == pytest.approx(4400 * math.pi / 180)
        assert drive.GetMaxForceAttr().Get() == 100
    receipt = robot.apply_task_collision_overlay(stage, expected_descriptor(spec))
    assert receipt['official_drive_solver_limits_unchanged']
    assert receipt['added_robot_prims'] == []
    robot.verify_task_collision_overlay(stage)
    stage.RemovePrim(expected_descriptor(spec)['pads'][0]['path'])
    with pytest.raises(ValueError, match='pad missing'): robot.verify_task_collision_overlay(stage)


@pytest.mark.parametrize('mutation', ['missing', 'scale', 'binding', 'shader', 'extra', 'physics', 'missing_visual', 'connected_visual'])
def test_bad_source_fails_before_official_robot_replacement(tmp_path, monkeypatch, mutation):
    stage, robot, spec = source_stage(tmp_path, monkeypatch)
    path = expected_descriptor(spec)['pads'][0]['path']; prim = stage.GetPrimAtPath(path)
    if mutation == 'missing': stage.RemovePrim(path)
    if mutation == 'scale': prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(1.))
    if mutation == 'binding': prim.GetRelationship('material:binding').SetTargets(['/Wrong/Material'])
    if mutation == 'shader': stage.GetAttributeAtPath('/World/Materials/CP09FingerRubberLeft/Shader.inputs:roughness').Set(.5)
    if mutation == 'extra': UsdGeom.Cube.Define(stage, path + '/extra')
    if mutation == 'physics': prim.GetRelationship('material:binding:physics').SetTargets(['/Wrong/Physics'])
    if mutation == 'missing_visual': stage.RemovePrim('/World/Materials/CP09FingerRubberLeft')
    if mutation == 'connected_visual':
        shader = UsdShade.Shader.Define(stage, '/Texture')
        output = shader.CreateOutput('rgb', Sdf.ValueTypeNames.Color3f)
        UsdShade.Shader(stage.GetPrimAtPath('/World/Materials/CP09FingerRubberLeft/Shader')).GetInput('diffuseColor').ConnectToSource(output)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): robot.prepare_stage(stage)
    assert stage.GetRootLayer().ExportToString() == before


def test_unbound_missing_owner_still_fails(tmp_path, monkeypatch):
    stage, robot, _ = source_stage(tmp_path, monkeypatch)
    robot._active_task_spec = None
    with pytest.raises(ValueError, match='target did not survive'): robot.restore_asset(stage)


def test_source_visual_graph_must_survive_and_no_late_overlay_waiver(tmp_path, monkeypatch):
    stage, robot, spec = source_stage(tmp_path, monkeypatch)
    robot.prepare_stage(stage)
    stage.GetAttributeAtPath('/World/Materials/CP09FingerRubberLeft/Shader.inputs:roughness').Set(.5)
    with pytest.raises(ValueError): robot.apply_task_collision_overlay(stage, expected_descriptor(spec))


def test_original_physics_material_must_match_and_is_never_overwritten(tmp_path, monkeypatch):
    stage, robot, spec = source_stage(tmp_path, monkeypatch, projected=False)
    prim = stage.GetPrimAtPath('/World/CP09/GripMaterialLeft')
    prim.GetAttribute('physics:staticFriction').Set(.7)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='friction differs'): robot.prepare_stage(stage)
    assert stage.GetRootLayer().ExportToString() == before
    prim.GetAttribute('physics:staticFriction').Set(2.)
    robot.prepare_stage(stage)
    assert robot.native_receipt()['source_pad_binding']['reconstructed_projected_physics_materials'] == []
    robot.apply_task_collision_overlay(stage, expected_descriptor(spec))


def test_official_angular_units_are_required_not_old_unscaled_fixture(tmp_path, monkeypatch):
    stage, robot, spec = source_stage(tmp_path, monkeypatch)
    robot.prepare_stage(stage)
    UsdPhysics.DriveAPI(stage.GetPrimAtPath(ROOTS[0] + '/joint1'), 'angular').GetStiffnessAttr().Set(4400.)
    with pytest.raises(ValueError, match='nonofficial X5 drives'):
        robot.apply_task_collision_overlay(stage, expected_descriptor(spec))


def test_missing_retained_pad_cannot_be_rebuilt_by_later_overlay(tmp_path, monkeypatch):
    stage, robot, spec = source_stage(tmp_path, monkeypatch)
    robot.prepare_stage(stage)
    path = expected_descriptor(spec)['pads'][0]['path']; stage.RemovePrim(path)
    with pytest.raises(ValueError, match='pad missing'):
        robot.apply_task_collision_overlay(stage, expected_descriptor(spec))
    assert not stage.GetPrimAtPath(path)


def test_strict_appearance_still_rejects_material_lost_during_restore(tmp_path, monkeypatch):
    from mind2act_eval.robot import cp09_source_pads
    stage, robot, _ = source_stage(tmp_path, monkeypatch)
    restore = cp09_source_pads.restore
    def lose_material(stage, captured):
        receipt = restore(stage, captured)
        stage.RemovePrim('/World/Materials/CP09FingerRubberLeft')
        return receipt
    monkeypatch.setattr(cp09_source_pads, 'restore', lose_material)
    with pytest.raises(ValueError, match='source appearance target did not survive'):
        robot.prepare_stage(stage)
