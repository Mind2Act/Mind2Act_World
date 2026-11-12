import re
from pathlib import Path
import pytest
from mind2act_eval.contracts import ClockSpec, InstanceSpec
from mind2act_eval.tasks import TaskRegistry
from mind2act_eval.tasks.P1 import DEFINITION

ROOT = Path(__file__).parents[3] / "task/P1"
SOURCES = {p: p.read_text() for p in ROOT.rglob("*.py")}


def _code_lines(text):
    in_doc = False
    for line in text.splitlines():
        if line.count('"""') % 2:
            in_doc = not in_doc
            continue
        if not in_doc and not line.strip().startswith("#"):
            yield line


def spec(difficulty, **kw):
    return InstanceSpec(task_id="P1", instance_id="P__%s__1" % difficulty, difficulty=difficulty,
                        source_id="P-148-b-P-v6", config_version="1", asset_version="1", protocol_version="v1", **kw)


def test_registered_and_cross_difficulty_binding_rejected():
    reg = TaskRegistry([DEFINITION])
    assert reg.get("P1") is DEFINITION
    with pytest.raises(Exception):
        reg.build(spec("hard"), spec("easy"), None)


@pytest.mark.parametrize("pattern", [
    r"\bfrom cases\b", r"\bimport cases\b", r"physco_", r"mind2act_env", r"mind2act_benchmark", r"mind2act\.",
    r"exec\(", r"ast\.parse", r"expert", r"/home/", r"/root/", r"Desktop", r"CreateStiffnessAttr", r"CreateMaxForceAttr",
])
def test_no_forbidden_dependency_or_expert_input(pattern):
    hits = [p.name for p, s in SOURCES.items() if any(re.search(pattern, l) for l in _code_lines(s))]
    assert hits == []


def test_build_rejects_non_120hz_clock_before_touching_sim():
    from mind2act_eval.tasks.P1.scene import build
    with pytest.raises(ValueError, match="120 Hz"):
        build(spec("easy", clock=ClockSpec(240, 30, 30)), None)


def test_missing_card_is_an_asset_gap_not_a_fallback():
    text = SOURCES[ROOT / "scene.py"]
    assert "asset gap" in text and "no fallback" in text


def test_timings_recorded_per_phase():
    text = SOURCES[ROOT / "scene.py"]
    for key in ("task_hook_s", "measure_s", "containment_s", "contacts_s", "ledger_s", "p1_timings.json"):
        assert key in text
