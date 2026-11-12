"""CPU USD regression for the actual shared X5 restore on station sublayers."""
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import pytest

from mind2act_eval.robot import ROOTS, X5Robot
from test_robot_appearance import scene, bindings


@pytest.mark.parametrize('thin', [False, True])
def test_restore_replaces_local_station_subtree_without_mutating_sources(tmp_path, monkeypatch, thin):
    prepared, _, asset = scene(tmp_path, monkeypatch)
    # Pristine bodies, poses and joint endpoints must all replace the frozen
    # station opinions, rather than repairing just rigidBodyEnabled.
    pristine = Usd.Stage.Open(str(asset))
    for name in ('base_link', 'link3'):
        body = UsdGeom.Xform.Define(pristine, '/X5A/' + name)
        body.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    joint = UsdPhysics.FixedJoint.Define(pristine, '/X5A/Joint')
    joint.CreateBody0Rel().SetTargets(['/X5A/base_link'])
    joint.CreateBody1Rel().SetTargets(['/X5A/link3'])
    joint.CreateLocalPos0Attr(Gf.Vec3f(.1, .2, .3))
    joint.CreateJointEnabledAttr(True)
    pristine.GetRootLayer().Save()
    for root in ROOTS:
        for name in ('base_link', 'link3'):
            body = prepared.GetPrimAtPath(root + '/' + name)
            UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Set(False)
            UsdGeom.Xformable(body).GetOrderedXformOps()[0].Set(Gf.Vec3d(9, 8, 7))
        joint = UsdPhysics.FixedJoint(prepared.GetPrimAtPath(root + '/Joint'))
        joint.GetBody1Rel().SetTargets([root + '/base_link'])
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(9))
        joint.GetJointEnabledAttr().Set(False)
        UsdGeom.Xform.Define(prepared, root + '/StationOnly')
    UsdGeom.Camera.Define(prepared, '/World/StationCamera').CreateFocalLengthAttr(42)
    station = tmp_path / 'station.usda'
    prepared.GetRootLayer().Export(str(station))
    if thin:
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths = [str(station)]
        assert all(not stage.GetRootLayer().GetPrimAtPath(root) for root in ROOTS)
        # An unrelated runtime opinion must survive consolidation as well.
        stage.GetPrimAtPath('/World/StationCamera').GetAttribute('focalLength').Set(51)
    else:
        stage = Usd.Stage.Open(str(station))
    sources = {path: path.read_bytes() for path in (asset, station)}
    before_bindings = bindings(stage)
    mounts = {root: UsdGeom.Xformable(stage.GetPrimAtPath(root)).ComputeLocalToWorldTransform(0)
              for root in ROOTS}
    robot = X5Robot(usd_path=asset)
    for _ in range(2):  # A second restore must retain source appearance and mounts.
        robot.restore_asset(stage)
        for root in ROOTS:
            assert UsdGeom.Xformable(stage.GetPrimAtPath(root)).ComputeLocalToWorldTransform(0) == mounts[root]
            for name in ('base_link', 'link3'):
                body = stage.GetPrimAtPath(root + '/' + name)
                assert UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is True
                assert UsdGeom.Xformable(body).GetLocalTransformation() == UsdGeom.Xformable(
                    pristine.GetPrimAtPath('/X5A/' + name)).GetLocalTransformation()
            assert not stage.GetPrimAtPath(root + '/StationOnly')
            joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root + '/Joint'))
            assert joint.GetJointEnabledAttr().Get() is True
            assert joint.GetBody1Rel().GetTargets() == [Sdf.Path(root + '/link3')]
            assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(.1, .2, .3)
            assert stage.GetPrimAtPath(root).GetMetadata('references').GetAddedOrExplicitItems()
        assert bindings(stage) == before_bindings
        assert stage.GetPrimAtPath('/World/StationCamera').GetAttribute('focalLength').Get() == (51 if thin else 42)
        assert all(path.read_bytes() == content for path, content in sources.items())
    assert stage.GetRootLayer().subLayerPaths == []
