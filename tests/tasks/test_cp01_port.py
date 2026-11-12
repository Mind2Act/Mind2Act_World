import ast
from pathlib import Path

from mind2act_eval.tasks import TaskRegistry
from mind2act_eval.tasks.CP01 import DEFINITION

ROOT = Path(__file__).resolve().parents[2] / "task/CP01"


def test_registers_through_explicit_registry():
    assert TaskRegistry([DEFINITION]).get("CP01").task_id == "CP01"


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_no_legacy_wrapper_or_robot_imports():
    for path in ROOT.rglob("*.py"):
        for module in _imports(ast.parse(path.read_text())):
            assert not module.startswith(("physco", "cases", "mind2act_env", "mind2act_benchmark",
                                          "runtime", "engine")), f"{module} in {path.name}"


def test_robot_values_only_via_ctx_robot():
    text = (ROOT / "mechanism/development_scene.py").read_text()
    for call in ("prepare_stage", "attach_cameras", "articulations", "rest_pose", "verify", "reset_to_rest"):
        assert f"ctx.robot.{call}" in text
    assert "HOME" not in text


def test_no_x5_driver_calls_in_task():
    for path in ROOT.rglob("*.py"):
        text = path.read_text()
        for token in ("restore_asset(", "configure_stage(", "verify_native(", "SimulationApp("):
            assert token not in text, f"{token} in {path.name}"


def test_no_direct_robot_handle_writes():
    text = (ROOT / "mechanism/development_scene.py").read_text()
    for token in ("set_joint_positions", "set_joint_velocities", "apply_action"):
        assert token not in text


def test_private_path_not_published():
    for path in ROOT.rglob("*.json"):
        assert "/Users/" not in path.read_text() and "/root/" not in path.read_text(), path.name
