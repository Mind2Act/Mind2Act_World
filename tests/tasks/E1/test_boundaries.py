import re
from pathlib import Path
import pytest
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.tasks import TaskRegistry
from mind2act_eval.tasks.E1 import DEFINITION

ROOT = Path(__file__).parents[3] / "task/E1"
SOURCES = {p: p.read_text() for p in ROOT.rglob("*.py")}


def spec(difficulty="easy"):
    return InstanceSpec(task_id="E1", instance_id="E__%s__episode_0001" % difficulty, difficulty=difficulty,
                        source_id="E-142-e-current-reconstruction-camera-v2", config_version="1", asset_version="1",
                        protocol_version="v1")


def test_registered_and_binding_mismatch_rejected():
    reg = TaskRegistry([DEFINITION])
    assert reg.get("E1") is DEFINITION
    with pytest.raises(Exception):
        reg.build(spec("hard"), spec("easy"), None)


@pytest.mark.parametrize("pattern", [
    r"align_stage", r"Carrier is missing", r"\bfrom cases\b", r"\bimport cases\b", r"physco_", r"mind2act_env",
    r"mind2act_benchmark", r"DriveAPI", r"CreateStiffnessAttr", r"CreateMaxForceAttr", r"exec\(", r"ast\.parse",
])
def test_no_forbidden_dependency(pattern):
    # docstrings may mention removed layers; code lines may not
    code_hits = [p.name for p, s in SOURCES.items() if any(re.search(pattern, line) for line in _code_lines(s))]
    assert code_hits == []


def _code_lines(text):
    in_doc = False
    for line in text.splitlines():
        if line.count('"""') % 2:
            in_doc = not in_doc
            continue
        if not in_doc and not line.strip().startswith("#"):
            yield line


def test_no_none_get_health_contract():
    assert "health" in SOURCES[ROOT / "scene.py"] and "Health(" in SOURCES[ROOT / "scene.py"]


def test_build_rejects_non_120hz_clock_before_touching_sim():
    from mind2act_eval.contracts import ClockSpec
    from mind2act_eval.tasks.E1.scene import build
    bad = InstanceSpec(task_id="E1", instance_id="i", difficulty="easy", source_id="s", config_version="1",
                       asset_version="1", protocol_version="v1", clock=ClockSpec(240, 30, 30))
    with pytest.raises(ValueError, match="120 Hz"):
        build(bad, None)


def test_observation_carries_spec_protocol_version():
    from types import SimpleNamespace
    from mind2act_eval.tasks._observation import observation
    robot = SimpleNamespace(observe=lambda: dict(left_arm=[0] * 6, right_arm=[0] * 6, left_gripper=1.0, right_gripper=1.0))
    s = InstanceSpec(task_id="E1", instance_id="i", difficulty="easy", source_id="s", config_version="1",
                     asset_version="1", protocol_version="v2")
    o = observation(s, robot, 120)
    assert o.protocol_version == "v2" and o.sim_time_s == 1.0
