import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from mind2act_eval.contracts import InstanceSpec, NativeEvidence
from mind2act_eval.tasks import TaskRegistry
from mind2act_eval.tasks.CP03 import DEFINITION
from mind2act_eval.tasks.CP03.evidence import StablePlacement, dish_target, euler_xyz_deg, quat_matrix
from mind2act_eval.tasks.CP03.instances import instance_document, verify_assets
from mind2act_eval.tasks.CP03.mechanism.observable_quality import PlateQuality, plate_tilt_deg
from mind2act_eval.tasks.CP03.scorer import RULE_VERSION, score

ROOT = Path(__file__).resolve().parents[2] / "src/mind2act_eval/tasks/CP03"


def native(level, satisfied, wrong=0, success=False):
    n = {"easy": 4, "medium": 7, "hard": 9}[level]
    return NativeEvidence("CP03", "i", 1, dict(
        credited_plates=min(satisfied,{'easy': 3, 'medium': 5, 'hard': 7}[level]),
        difficulty=level, scoring_version=RULE_VERSION, required_steps=n, steps_satisfied=satisfied, wrong_or_extra_items=wrong,
        task_success=success))


def test_registers():
    assert TaskRegistry([DEFINITION]).get("CP03").task_id == "CP03"


@pytest.mark.parametrize("level,n", [("easy", 4), ("medium", 7), ("hard", 9)])
def test_success_100_partial_uses_core_cap(level, n):
    core_n = {'easy': 3, 'medium': 5, 'hard': 7}[level]
    cap = dict(easy=15,medium=10,hard=7)[level]
    assert score(native(level, n, success=True)).value == 100
    assert score(native(level, 0)).value == 0
    assert score(native(level, 2)).value == pytest.approx(cap * 2 / core_n)
    assert score(native(level, n - 1)).value <= cap
    assert score(native(level, n)).components["completion"] == {"k": core_n, "n": core_n, "ratio": 1.0}


def test_wrong_plates_subtract_and_floor():
    assert score(native("medium", 5, wrong=1)).value == pytest.approx(10 * 4 / 5)
    assert score(native("easy", 1, wrong=2)).value == 0


def test_inconsistent_evidence_rejected():
    with pytest.raises(ValueError):
        score(native("easy", 5))
    with pytest.raises(ValueError):
        score(native("easy", 3, success=True))
    with pytest.raises(ValueError):
        score(native("hard", 9, wrong=1, success=True))
    bad = NativeEvidence("CP03", "i", 1, dict(difficulty="easy", required_steps=7, steps_satisfied=0,
                                             wrong_or_extra_items=0, task_success=False))
    with pytest.raises(ValueError):
        score(bad)


def test_quaternion_helpers_match_scipy():
    r = pytest.importorskip("scipy.spatial.transform").Rotation
    q = np.array([0.8, 0.3, -0.4, 0.2]); q = q / np.linalg.norm(q)
    rot = r.from_quat(q[[1, 2, 3, 0]])
    assert np.allclose(quat_matrix(q), rot.as_matrix())
    assert np.allclose(euler_xyz_deg(q), rot.as_euler("xyz", degrees=True))


def test_terminal_support_does_not_need_tracker_manipulation_or_wait():
    for relaxed, expected in ((False, True), (True, True)):
        tracker = StablePlacement(5, .004, require_manipulation=not relaxed)
        for _ in range(5):
            tracker.update([0, 0, 0], bilateral=False, elevated=False, released=True, supported=True, correct=True)
        assert tracker.complete is expected


def test_dish_target_is_terminal_geometry_only():
    assert dish_target(0., .8, 1., 0., .05, .8)
    assert not dish_target(.03, .8, 1., 0., .05, .8)


def test_plate_quality_windows():
    protocol = {"max_transport_tilt_deg": 3.0, "transient_peak_limit_deg": 6.0,
                "max_contiguous_excursion_s": .05, "max_total_excursion_s": .1}
    q = PlateQuality(protocol, 0.0)
    for tick in range(1, 20):
        q.observe(tick, z=.1, tilt_deg=1.0, fingers=[True, True], slot_mouth_intersection=False, slot_support=False)
    q.observe(20, z=.1, tilt_deg=1.0, fingers=[True, True], slot_mouth_intersection=True, slot_support=False)
    q.observe(21, z=.1, tilt_deg=1.0, fingers=[False, False], slot_mouth_intersection=True, slot_support=True)
    assert q.result()["tilt_quality_pass"]
    with pytest.raises(ValueError):
        q.observe(25, z=0, tilt_deg=0, fingers=[False, False], slot_mouth_intersection=False, slot_support=False)


def test_plate_tilt_axis():
    assert plate_tilt_deg([1, 0, 0, 0]) == pytest.approx(90.0)


def native_spec(level="easy"):
    n = {"easy": 3, "medium": 5, "hard": 7}[level]
    return {"task_id": "CP03", "difficulty": level, "case_id": f"CP03_{level}_x", "instruction": "Load plates.",
            "binding": {"expected_units": n, "declared_version": "v"},
            "config": {"source_closure": {"layers/scene.usda": "aa", "tex/CP03-YCB029/texture_map.png": "bb"},
                       "native_inputs": {
                           "args": {"scene": "layers/scene.usda", "output": "assigned_by_runner"},
                           "setup": {"parameters": {"scene_sha256": "s", "empty_slot_indices": [1]}},
                           "config": {"task": "CP03", "difficulty": level, "count": n, "scene_sha256": "s",
                                      "speed_m_s": .012},
                           "manifest": {"levels": {level: {"mode": 1, "speed": .012}}}}}}


def doc(level="easy"):
    return instance_document(native_spec(level), x5_asset="x5/ARX.usd", asset_version="a1", config_version="c1")


def test_instance_document_roundtrips_into_a_frozen_spec():
    spec = InstanceSpec.from_dict(doc("hard") | {"clock": {"physics_hz": 120, "control_hz": 120, "rgb_hz": 30}})
    assert spec.source_id == "cp03-hard-b-20261028" and spec.assets["station"] == "layers/scene.usda"
    assert len(spec.config["robot_rest"]) == 8


def test_instance_document_rejects_mismatches():
    bad = native_spec("easy"); bad["binding"]["expected_units"] = 5
    with pytest.raises(ValueError):
        instance_document(bad, x5_asset="x", asset_version="a", config_version="c")
    bad = native_spec("easy"); bad["config"]["native_inputs"]["config"]["difficulty"] = "hard"
    with pytest.raises(ValueError):
        instance_document(bad, x5_asset="x", asset_version="a", config_version="c")
    bad = native_spec("easy"); del bad["config"]["source_closure"]["tex/CP03-YCB029/texture_map.png"]
    with pytest.raises(ValueError):
        instance_document(bad, x5_asset="x", asset_version="a", config_version="c")


def test_missing_texture_is_an_error_not_a_fallback(tmp_path):
    (tmp_path / "scene.usda").write_bytes(b"x")

    class Resolver:
        def resolve(self, asset_id):
            path = tmp_path / ("scene.usda" if asset_id == "station" else "texture_map.png")
            if not path.exists():
                raise FileNotFoundError(asset_id)
            return path
    document = doc()
    document["config"]["asset_sha256"]["layers/scene.usda"] = __import__("hashlib").sha256(b"x").hexdigest()
    with pytest.raises(FileNotFoundError):
        verify_assets(Resolver(), document)


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_no_legacy_imports_or_robot_drive_calls():
    for path in ROOT.rglob("*.py"):
        text = path.read_text()
        for module in _imports(ast.parse(text)):
            assert not module.startswith(("physco", "cases", "mind2act_env", "mind2act_benchmark",
                                          "runtime", "engine", "mind2act.")), f"{module} in {path.name}"
        for token in ("set_joint_positions", "set_joint_velocities", "apply_action", "SimulationApp(",
                      "set_gains", "set_max_efforts", "set_armatures", "FixedBase"):
            assert token not in text, f"{token} in {path.name}"


def test_deep_copy_of_document_is_stable():
    assert copy.deepcopy(doc()) == doc()


def test_private_absolute_paths_rejected():
    bad = native_spec()
    bad["config"]["native_inputs"]["manifest"]["source_scene"] = "/root/share/x/scene.usda"
    with pytest.raises(ValueError):
        instance_document(bad, x5_asset="x", asset_version="a", config_version="c")


def test_mount_and_park_go_through_the_shared_robot_service():
    assert "ctx.robot.mount_bases" in (ROOT / "mechanism/rack_scene.py").read_text()
    assert "ctx.robot.park_arm" in (ROOT / "scene.py").read_text()
