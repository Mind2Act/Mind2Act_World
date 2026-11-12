import ast
from pathlib import Path

import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.tasks import TaskRegistry
from mind2act_eval.tasks.CP02 import DEFINITION
from mind2act_eval.tasks.CP02.evidence import MeasuredTask, quat_matrix, roll_deg
from mind2act_eval.tasks.CP02.scorer import score

ROOT = Path(__file__).resolve().parents[2] / "src/mind2act_eval/tasks/CP02"
RELAXED = {"evaluation_protocol": "physco-task-centric-v2", "protocol": {"max_food_gripper_tilt_deg": 30.0}}


def spec(level, n):
    return {"difficulty": level, "speed_deg_s": 10.0, "demonstration_speed_deg_s": 10.0,
            "acceptance": {"stability_seconds": 0.1}, "region_inner_outer_radius_m": [.0235,.0275],
            "items": [{"identity": f"m{i}", "target_cake_local_m": [0.01 * i, 0.0, 0.02]} for i in range(n)]}


def raw(n, angle_deg=0.0, placed=True, lid_deg=0.0):
    half = np.radians(angle_deg) / 2
    lid = np.radians(-lid_deg) / 2
    bodies = [{"position": [0.01 * i, 0, 0.02] if placed else [1, 1, 1], "quaternion_wxyz": [1, 0, 0, 0],
               "identity": f"m{i}", "bilateral": False, "touch_robot": False, "supported": True,
               "held_tilt_deg": 0.0} for i in range(n)]
    return {"rotor": {"position": [0, 0, 0], "quaternion_wxyz": [np.cos(half), 0, 0, np.sin(half)],
                      "angular_velocity_rad_s": [0, 0, 1]},
            "bodies": bodies,
            "lid": {"quaternion_wxyz": [np.cos(lid), np.sin(lid), 0, 0], "robot_contact": False},
            "buttons": {}, "lid_angle_deg": lid_deg, "lid_contact": False,
            "start_pressed": False, "stop_pressed": False}


def native(level, satisfied, wrong=0, success=False):
    n = {"easy": 4, "medium": 8, "hard": 10}[level]
    return NativeEvidence("CP02", "i", 1, dict(
        credited_macarons=min(satisfied,{'easy': 4, 'medium': 6, 'hard': 6}[level]), core_wrong_or_extra_items=wrong, scoring_version=SCORING_VERSION,
        difficulty=level, required_steps=n, steps_satisfied=satisfied, wrong_or_extra_items=wrong,
        physical_task_success=success, task_process_pass=success))


def test_registers():
    assert TaskRegistry([DEFINITION]).get("CP02").task_id == "CP02"


@pytest.mark.parametrize("level,n", [("easy", 4), ("medium", 8), ("hard", 10)])
def test_success_100_partial_uses_core_cap(level, n):
    core_n = {'easy': 4, 'medium': 6, 'hard': 6}[level]
    cap = dict(easy=15,medium=10,hard=7)[level]
    assert score(native(level, n, success=True)).value == 100
    assert score(native(level, 0)).value == 0
    assert score(native(level, 2)).value == pytest.approx(cap * 2 / core_n)
    assert score(native(level, n - 1)).value <= cap
    assert score(native(level, n)).components["completion"] == {"k": core_n, "n": core_n, "ratio": 1.0}


def test_wrong_items_subtract_and_floor():
    assert score(native("medium", 5, wrong=2)).value == pytest.approx(10 * 3 / 6)
    assert score(native("hard", 1, wrong=4)).value == 0


def test_inconsistent_evidence_rejected():
    with pytest.raises(ValueError):
        score(native("easy", 5))
    with pytest.raises(ValueError):
        score(native("easy", 3, success=True))
    bad = NativeEvidence("CP02", "i", 1, dict(difficulty="easy", required_steps=8, steps_satisfied=0,
                                             wrong_or_extra_items=0, physical_task_success=False, task_process_pass=False))
    with pytest.raises(ValueError):
        score(bad)


def test_quaternion_helpers_match_scipy():
    r = pytest.importorskip("scipy.spatial.transform").Rotation
    q = np.array([0.8, 0.3, -0.4, 0.2])
    rot = r.from_quat(q[[1, 2, 3, 0]])
    assert np.allclose(quat_matrix(q), rot.as_matrix())
    assert roll_deg(q) == pytest.approx(rot.as_euler("xyz", degrees=True)[0])


def test_easy_all_decorations_stable_after_cue():
    n = 4
    task = MeasuredTask(spec("easy", n), RELAXED, raw(n))
    for tick in range(1, 500):
        task.update(tick, raw(n))
    grade = task.grade()
    assert grade["physical_task_success"] and all(p["stable"] for p in grade["placements"])
    assert grade["steps_satisfied"] == 4 and grade["required_steps"] == 4 and grade["wrong_or_extra_items"] == 0
    assert score(NativeEvidence("CP02", "i", 500, dict(grade, difficulty="easy"))).value == 100


def test_unplaced_decorations_get_no_credit():
    task = MeasuredTask(spec("easy", 2), RELAXED, raw(2, placed=False))
    for tick in range(1, 500):
        task.update(tick, raw(2, placed=False))
    assert not any(p["stable"] for p in task.grade()["placements"])


@pytest.mark.parametrize('cue_finished', [False, True])
def test_cue_contact_and_tilt_are_not_latched_as_formal_violations(cue_finished):
    initial = raw(1)
    initial['cover'] = dict(position=[0, 0, 0], bilateral=True, touches_food=True,
                            touch_robot=True, supported=False)
    initial['cover_target_world_m'] = [0, 0, 0]
    initial['cover_source_z_m'] = 0
    initial['cover_contains_cake_center'] = False
    initial['bodies'][0]['held_tilt_deg'] = 45
    config = dict(evaluation_protocol='source', protocol=dict(max_food_gripper_tilt_deg=30))
    task = MeasuredTask(spec('hard', 1), config, initial)
    task.timeline.clock['cue_finished'] = cue_finished
    task.update(1, initial)
    failures = task.timeline.controls['failures']
    assert not failures


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
                      "FixedBase"):
            assert token not in text, f"{token} in {path.name}"


def test_no_private_paths_in_package_data():
    for path in ROOT.rglob("*.json"):
        text = path.read_text()
        assert "/Users/" not in text and "/root/" not in text, path.name


def test_base_mount_goes_through_the_shared_robot_service():
    assert "ctx.robot.mount_bases" in (ROOT / "mechanism/cake_scene.py").read_text()


def test_setup_steps_count_only_before_their_items():
    task = MeasuredTask(spec("medium", 2), RELAXED, raw(2))
    task.lid_tick = 10
    task.timeline.controls.update(lid_opened=True, started=True, start_tick=50)
    task.item_first = {0: 40}
    got = task.steps()
    assert got["required_steps"] == 4 and got["steps_satisfied"] == 1  # Independent lid step.
    task.item_first = {0: 60}
    task.timeline.controls['start_order_valid'] = True
    assert task.steps()["steps_satisfied"] == 2
