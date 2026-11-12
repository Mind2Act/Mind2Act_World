"""The approved PR2 exception changes four existing contacts and no X5 mechanics."""
from dataclasses import replace
import json
import math

import pytest

pytest.importorskip('pxr')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.robot import PROFILE, X5Robot
from mind2act_eval.robot.task_collision_overlay import snapshot


def selected(level='easy'):
    source = ('pr2-hard-directional-grasp-run925de9-v1' if level == 'hard' else
              'pr2-em-edge-side-grasp-run925de9-v1')
    return InstanceSpec('PR2', 'selected-' + level, level, source,
                        'pr2-measured-stability-20260930-v2', 'test', 'v1',
                        config=dict(initial_volt=100., target_volt=140., fixture_y_offset_m=0.))


def stage():
    st = Usd.Stage.CreateInMemory()
    for side in ('Left', 'Right'):
        root = '/World/Robot' + side + '/Asset'
        prim = UsdGeom.Xform.Define(st, root).GetPrim()
        for name, value in [('solverPositionIterationCount', 8), ('solverVelocityIterationCount', 0)]:
            prim.CreateAttribute('physxArticulation:' + name, Sdf.ValueTypeNames.Int).Set(value)
        prim.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(True)
        for i in range(1, 9):
            joint = (UsdPhysics.RevoluteJoint if i <= 6 else UsdPhysics.PrismaticJoint).Define(st, root + '/joint' + str(i))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular' if i <= 6 else 'linear')
            props = PROFILE['arm' if i <= 6 else 'finger']; scale = math.pi / 180 if i <= 6 else 1.
            drive.CreateStiffnessAttr(props['stiffness'] * scale)
            drive.CreateDampingAttr(props['damping'] * scale); drive.CreateMaxForceAttr(100.)
        for i in (7, 8):
            p = root + '/link%d/collisions/link%d/node_STL_BINARY_' % (i, i)
            mesh = UsdGeom.Mesh.Define(st, p)
            mesh.CreatePointsAttr([(0, 0, 0), (.01, 0, 0), (0, .01, 0)])
            mesh.CreateFaceVertexCountsAttr([3]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mat = UsdShade.Material.Define(st, '/World/PrecisionContactMaterial')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(.9); api.CreateDynamicFrictionAttr(.7); api.CreateRestitutionAttr(0.)
    return st


def prepared(spec, tmp_path):
    robot = X5Robot(); robot.bind_task_instance(spec, tmp_path)
    robot._configured = robot._mounted = True
    return robot


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_pr2_source_contacts_are_a_verified_x5_exception(tmp_path, level):
    from mind2act_eval.robot.pr2_source_contacts import expected_descriptor
    spec = selected(level); st = stage(); before = snapshot(st)
    robot = prepared(spec, tmp_path); descriptor = expected_descriptor(spec)
    receipt = robot.apply_task_collision_overlay(st, descriptor)
    assert receipt['task_id'] == 'PR2' and receipt['added_robot_prims'] == []
    assert receipt['official_drive_solver_limits_unchanged']
    after = snapshot(st)
    changed = {path for path in before if before[path] != after[path]}
    assert changed == set(descriptor['colliders']) and len(changed) == 4
    for path in changed:
        prim = st.GetPrimAtPath(path)
        assert prim.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.00015)
        assert prim.GetAttribute('physxCollision:restOffset').Get() == 0.
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        assert str(material.GetPath()) == '/World/PrecisionContactMaterial'
    robot.verify_task_collision_overlay(st)
    st.GetPrimAtPath(descriptor['colliders'][0]).GetAttribute('physxCollision:contactOffset').Set(.02)
    with pytest.raises(ValueError): robot.verify_task_collision_overlay(st)


@pytest.mark.parametrize('change', ['task', 'source', 'descriptor', 'drive', 'missing'])
def test_pr2_exception_never_authorizes_other_tasks_or_mechanics(tmp_path, change):
    from mind2act_eval.robot.pr2_source_contacts import expected_descriptor
    spec = selected(); desc = expected_descriptor(spec); st = stage()
    if change == 'task': spec = replace(spec, task_id='PR1')
    if change == 'source': spec = replace(spec, source_id='other-source')
    if change == 'descriptor': desc = {**desc, 'static_friction': 2.}
    if change == 'drive':
        UsdPhysics.DriveAPI(st.GetPrimAtPath('/World/RobotLeft/Asset/joint7'), 'linear').GetMaxForceAttr().Set(200.)
    if change == 'missing': st.RemovePrim(desc['colliders'][0])
    robot = prepared(spec, tmp_path); before = snapshot(st)
    with pytest.raises(ValueError): robot.apply_task_collision_overlay(st, desc)
    assert snapshot(st) == before


def test_pr2_large_mesh_audit_is_bounded_and_keeps_full_snapshot_hashes(tmp_path):
    from mind2act_eval.robot.pr2_source_contacts import expected_descriptor
    from mind2act_eval.robot.task_collision_overlay import digest
    spec = selected(); st = stage()
    mesh = UsdGeom.Mesh.Define(st, '/World/RobotLeft/Asset/large_visual')
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(.1234567, .2345678, .3456789)] * 30000))
    before = snapshot(st)
    assert len(json.dumps(before)) > 1024 * 1024
    robot = prepared(spec, tmp_path)
    receipt = robot.apply_task_collision_overlay(st, expected_descriptor(spec))
    after = snapshot(st)
    for name, values in [('before', before), ('after', after)]:
        path = tmp_path / 'robot-contact-overlay' / 'PR2' / (name + '.json')
        assert path.stat().st_size < 1024
        audit = json.loads(path.read_text())
        assert audit['schema'] == 'mind2act.robot_snapshot_summary.v1'
        assert audit['full_values_stored'] is False
        assert audit['snapshot_sha256'] == receipt[name + '_snapshot_sha256'] == digest(values)
        assert audit['prim_count'] == len(values)
        assert audit['attribute_count'] == sum(len(p['attrs']) for p in values.values())
    assert set(receipt['changed_robot_prims']) == set(expected_descriptor(spec)['colliders'])
    robot.verify_task_collision_overlay(st)


@pytest.mark.parametrize('change', ['last_mesh_point', 'time_sample'])
def test_pr2_full_guard_rejects_geometry_and_sample_changes(change):
    from mind2act_eval.robot.pr2_source_contacts import SourceContactGuard, expected_descriptor
    st = stage(); spec = selected()
    mesh = UsdGeom.Mesh.Define(st, '/World/RobotLeft/Asset/large_visual')
    points = Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * 30000)
    attr = mesh.CreatePointsAttr(points)
    guard = SourceContactGuard(st, expected_descriptor(spec))
    points[-1] = Gf.Vec3f(1, 2, 3)
    if change == 'last_mesh_point': attr.Set(points)
    else: attr.Set(points, Usd.TimeCode(1))
    with pytest.raises(ValueError, match='unauthorized X5 semantics'):
        guard.authorize_after(st)
