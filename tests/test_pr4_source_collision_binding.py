"""CPU schema restoration through the real shared asset replacement seam."""
from dataclasses import replace

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from mind2act_eval.contracts import ClockSpec, InstanceSpec
from mind2act_eval.robot import X5Robot
from mind2act_eval.tasks.PR4.scene import collision_hulls
from test_physics_profile import SOURCES, identity_spec


def source_stage(tmp_path, monkeypatch, *, changed_geometry=False):
    stage = Usd.Stage.CreateInMemory()
    official = Usd.Stage.CreateNew(str(tmp_path / 'official.usda'))
    official.SetDefaultPrim(UsdGeom.Xform.Define(official, '/X5').GetPrim())
    bindings = []
    points = np.array([[.01 if i & 1 else -.01, .02 if i & 2 else -.02,
                        .03 if i & 4 else -.03] for i in range(8)])
    for side in ('Left', 'Right'):
        UsdGeom.Xform.Define(stage, '/World/Robot' + side + '/Asset')
        for link in ('link7', 'link8'):
            tail = link + '/collisions/' + link + '/node_STL_BINARY_'
            parent = '/World/Robot' + side + '/Asset/' + tail
            bindings.append(dict(source_xform=parent, same_mesh=parent + '/mesh'))
            for target, path, is_source in ((stage, parent, True), (official, '/X5/' + tail, False)):
                if side == 'Right' and not is_source:
                    continue
                xform = UsdGeom.Xform.Define(target, path)
                xform.AddTranslateOp().Set((.1, .2, .3))
                mesh = UsdGeom.Mesh.Define(target, path + '/mesh')
                mesh.CreatePointsAttr(points if is_source or not changed_geometry else points * 2)
                mesh.CreateFaceVertexCountsAttr([3]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
                collider = mesh.GetPrim() if is_source else xform.GetPrim()
                UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('convexHull')
                collider.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0004)
                collider.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    official.GetRootLayer().Save()
    import mind2act_eval.robot as robot_module
    monkeypatch.setattr(robot_module, 'verify_asset', lambda *_: None)
    robot = X5Robot(usd_path=tmp_path / 'official.usda')
    # No Kit configuration in this CPU test; asset restoration is the real method.
    monkeypatch.setattr(robot, 'configure_stage', lambda _: {'cpu_fixture': True})
    return stage, robot, bindings


def selected_spec(row, bindings):
    spec = identity_spec(row)
    return replace(spec, config={**dict(spec.config), 'source_case': {
        'origin': {'collider_schema_migration': bindings}}})


@pytest.mark.parametrize('row', SOURCES)
def test_selected_pr4_source_mesh_collision_survives_shared_prepare(tmp_path, monkeypatch, row):
    stage, robot, bindings = source_stage(tmp_path, monkeypatch)
    source_hulls = collision_hulls(stage)
    robot.bind_task_instance(selected_spec(row, bindings), tmp_path)
    robot.prepare_stage(stage)
    assert collision_hulls(stage) == source_hulls
    for entry in bindings:
        parent = stage.GetPrimAtPath(entry['source_xform'])
        mesh = stage.GetPrimAtPath(entry['same_mesh'])
        assert not parent.HasAPI(UsdPhysics.CollisionAPI)
        assert not parent.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert mesh.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'convexHull'
        assert mesh.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.0004)
        assert mesh.GetAttribute('physxCollision:restOffset').Get() == 0.
    receipt = robot.native_receipt()['source_collision_binding']
    assert receipt['source_spec_sha256'] == row[2]
    assert len(receipt['meshes']) == 4
    assert receipt['geometry_unchanged'] is True


def test_selected_pr4_rejects_different_official_mesh(tmp_path, monkeypatch):
    stage, robot, bindings = source_stage(tmp_path, monkeypatch, changed_geometry=True)
    robot.bind_task_instance(selected_spec(SOURCES[0], bindings), tmp_path)
    with pytest.raises(ValueError, match='PR4 source finger geometry differs'):
        robot.prepare_stage(stage)


@pytest.mark.parametrize('mutation', ['path', 'missing_source_collision'])
def test_selected_pr4_rejects_unqualified_binding_before_asset_write(tmp_path, monkeypatch, mutation):
    stage, robot, bindings = source_stage(tmp_path, monkeypatch)
    spec = selected_spec(SOURCES[0], bindings)
    if mutation == 'path':
        bindings[0]['same_mesh'] = '/World/Unrelated/mesh'
        spec = selected_spec(SOURCES[0], bindings)
    else:
        stage.GetPrimAtPath(bindings[0]['same_mesh']).RemoveAPI(UsdPhysics.CollisionAPI)
    before = stage.GetRootLayer().ExportToString()
    robot.bind_task_instance(spec, tmp_path)
    with pytest.raises(ValueError):
        robot.prepare_stage(stage)
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('bound', [False, True])
def test_other_tasks_keep_official_collision_schema(tmp_path, monkeypatch, bound):
    stage, robot, _ = source_stage(tmp_path, monkeypatch)
    if bound:
        robot.bind_task_instance(InstanceSpec('P1', 'fixture', 'easy', 'fixture', 'v1', 'test', 'v1',
                                             clock=ClockSpec(120, 30, 30)), tmp_path)
    robot.prepare_stage(stage)
    parent = stage.GetPrimAtPath('/World/RobotLeft/Asset/link7/collisions/link7/node_STL_BINARY_')
    assert parent.HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath(str(parent.GetPath()) + '/mesh').HasAPI(UsdPhysics.CollisionAPI)


def test_schema_geometry_restore_does_not_admit_a_different_case(tmp_path,monkeypatch):
    from mind2act_eval.tasks.PR4.source_admission import validate_case_source
    stage,robot,bindings=source_stage(tmp_path,monkeypatch)
    spec=replace(selected_spec(SOURCES[0],bindings),instance_id='another-case',seed=991)
    robot.bind_task_instance(spec,tmp_path)
    robot.prepare_stage(stage)
    assert robot.native_receipt()['source_collision_binding']['geometry_unchanged']
    with pytest.raises(ValueError):validate_case_source(spec)
