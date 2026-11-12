"""Real USD collider selection around the native scene-query boundary."""
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from mind2act_eval.variations import station_targets
from mind2act_eval.variations import geometry


def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/EvalVariation')
    cube = UsdGeom.Cube.Define(stage, '/World/EvalVariation/box')
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return stage


def test_support_and_own_clutter_hits_do_not_reject(monkeypatch):
    stage = scene()
    monkeypatch.setattr(geometry, '_query_shape', lambda path: [
        '/World/Worktable/top', '/World/Floor', '/World/EvalVariation/box'])
    result = geometry.verify_geometry(stage, station_targets())
    assert result['status'] == 'verified'
    assert len(result['shapes']) == 1
    assert result['occlusion_verified'] is False


def test_mechanism_outside_generic_task_namespace_still_rejects(monkeypatch):
    stage = scene()
    monkeypatch.setattr(geometry, '_query_shape', lambda path: ['/World/Carousel/rotor/tray0'])
    with pytest.raises(ValueError, match='task/mechanism collision geometry.*Carousel'):
        geometry.verify_geometry(stage, station_targets())


@pytest.mark.parametrize('approximation, fatal', [('convexHull', True), ('none', False),
                                                ('convexDecomposition', False)])
def test_convex_query_does_not_mislabel_nonconvex_mesh_hit(monkeypatch, approximation, fatal):
    stage = scene()
    stage.RemovePrim('/World/EvalVariation/box')
    owner = UsdGeom.Xform.Define(stage, '/World/EvalVariation/object').GetPrim()
    UsdPhysics.CollisionAPI.Apply(owner)
    UsdPhysics.MeshCollisionAPI.Apply(owner).CreateApproximationAttr(approximation)
    UsdGeom.Mesh.Define(stage, '/World/EvalVariation/object/mesh')
    monkeypatch.setattr(geometry, '_query_shape', lambda path: ['/World/Task/target'])
    if fatal:
        with pytest.raises(ValueError, match='task/mechanism collision geometry'):
            geometry.verify_geometry(stage, station_targets())
    else:
        result = geometry.verify_geometry(stage, station_targets())
        assert result['status'] == 'requires_visual_review'
        assert result['unconfirmed_mesh_overlaps'][0]['collisions'] == ['/World/Task/target']


def test_disabled_and_visual_only_shapes_are_not_queried(monkeypatch):
    stage = scene()
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/EvalVariation/box')).CreateCollisionEnabledAttr(False)
    UsdGeom.Cube.Define(stage, '/World/EvalVariation/label')
    def unexpected(path):
        pytest.fail('noncollider was queried: '+path)
    monkeypatch.setattr(geometry, '_query_shape', unexpected)
    assert geometry.verify_geometry(stage, station_targets())['shapes'] == []
