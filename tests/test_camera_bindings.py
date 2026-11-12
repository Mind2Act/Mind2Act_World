import json
from types import SimpleNamespace

import pytest

from mind2act_eval.sim import IsaacSimulator

pytest.importorskip("pxr")


def camera_stage():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdRender

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/Wrist")
    move = parent.AddTranslateOp()
    move.Set((1., 2., 3.))
    path = "/World/Wrist/Camera"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.AddTranslateOp().Set((0., 0., .1))
    product = UsdRender.Product.Define(stage, "/Render/Wrist")
    product.CreateCameraRel().SetTargets([path])

    def pose():
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
        rotation = matrix.ExtractRotationQuat()
        return matrix.ExtractTranslation(), (rotation.GetReal(), *rotation.GetImaginary())

    sensor = SimpleNamespace(prim_path=path, get_render_product_path=lambda: "/Render/Wrist",
                             get_world_pose=pose)
    # IsaacSimulator.stage is a native-context property. Reuse the real readback
    # method on a holder with a real composed USD stage.
    holder = SimpleNamespace(stage=stage, _cameras={"left_wrist": sensor},
                             _camera_config={"left_wrist": {"path": path}})
    return holder, move, product


def test_binding_receipt_detects_render_product_pointing_at_another_camera():
    from pxr import UsdGeom

    holder, _, product = camera_stage()
    UsdGeom.Camera.Define(holder.stage, "/World/Overview")
    product.GetCameraRel().SetTargets(["/World/Overview"])

    row = IsaacSimulator._camera_bindings(holder)["left_wrist"]

    assert row["status"] == "observed"
    assert row["requested_prim"] == row["sensor_prim"] == "/World/Wrist/Camera"
    assert row["render_camera_targets"] == ["/World/Overview"]
    assert row["render_camera_matches_sensor"] is False


def test_binding_receipt_reads_current_composed_wrist_pose_each_time():
    holder, move, _ = camera_stage()
    before = IsaacSimulator._camera_bindings(holder)["left_wrist"]
    move.Set((2., 4., 6.))
    after = IsaacSimulator._camera_bindings(holder)["left_wrist"]

    assert before["world_position_m"] == pytest.approx([1., 2., 3.1])
    assert after["world_position_m"] == pytest.approx([2., 4., 6.1])
    assert after["usd_world_matrix"][3][:3] == pytest.approx(after["world_position_m"])
    assert after["parent_prim"] == "/World/Wrist"
    assert after["reset_xform_stack"] is False
    assert before["render_camera_matches_sensor"] is after["render_camera_matches_sensor"] is True


@pytest.mark.parametrize("component", ["position", "orientation", "matrix"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "positive_inf", "negative_inf"])
def test_nonfinite_camera_readback_does_not_abort_verified_receipt(tmp_path, component, value):
    from pxr import UsdGeom, UsdRender

    holder, move, _ = camera_stage()
    sensor = holder._cameras["left_wrist"]
    position, orientation = [1., 2., 3.1], [1., 0., 0., 0.]
    if component == "matrix":
        # Keep the SDK pose finite so this case reaches the real USD matrix.
        move.Set((value, 2., 3.))
    elif component == "position":
        position[0] = value
    else:
        orientation[0] = value
    sensor.get_world_pose = lambda: (position, orientation)

    UsdGeom.Camera.Define(holder.stage, "/World/HeadCamera")
    product = UsdRender.Product.Define(holder.stage, "/Render/Head")
    product.CreateCameraRel().SetTargets(["/World/HeadCamera"])
    holder._cameras["head"] = SimpleNamespace(
        prim_path="/World/HeadCamera", get_render_product_path=lambda: "/Render/Head",
        get_world_pose=lambda: ([0., 0., 0.], [1., 0., 0., 0.]))
    holder._camera_config["head"] = {"path": "/World/HeadCamera"}
    holder._output_dir = tmp_path
    holder._camera_bindings = lambda: IsaacSimulator._camera_bindings(holder)

    # Exercise the strict JSON writer used by ready(), not just the readback.
    IsaacSimulator._write_camera_receipt(
        holder, "verified", 1, {name: {"valid": True} for name in holder._cameras})
    receipt = json.loads((tmp_path / "camera-receipt.json").read_text())

    assert receipt["status"] == "verified"
    assert receipt["initial_frame_encoded"] is True
    assert all(row["valid"] for row in receipt["cameras"].values())
    bad = receipt["bindings"]["left_wrist"]
    assert bad["status"] == "unavailable"
    assert "nonfinite" in bad["error"]
    assert receipt["bindings"]["head"]["status"] == "observed"
    assert receipt["bindings"]["head"]["render_camera_matches_sensor"] is True
