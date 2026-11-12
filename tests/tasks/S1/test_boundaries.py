import re
from pathlib import Path
import pytest
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.tasks import TaskRegistry
from mind2act_eval.tasks.S1 import DEFINITION

ROOT = Path(__file__).parents[3] / "task/S1"
SOURCES = {p: p.read_text() for p in ROOT.rglob("*.py")}


def _code_lines(text):
    in_doc = False
    for line in text.splitlines():
        if line.count('"""') % 2:
            in_doc = not in_doc
            continue
        if not in_doc and not line.strip().startswith("#"):
            yield line


def spec(difficulty):
    return InstanceSpec(task_id="S1", instance_id="S__%s__1" % difficulty, difficulty=difficulty,
                        source_id="S-147-direct-continuous-340aff8-run925de9", config_version="1",
                        asset_version="1", protocol_version="v1")


def test_registered_and_cross_difficulty_binding_rejected():
    reg = TaskRegistry([DEFINITION])
    assert reg.get("S1") is DEFINITION
    with pytest.raises(Exception):
        reg.build(spec("hard"), spec("easy"), None)


@pytest.mark.parametrize("pattern", [
    r"\bfrom cases\b", r"\bimport cases\b", r"physco_", r"mind2act_env", r"mind2act_benchmark", r"mind2act\.",
    r"DriveAPI.*Robot", r"exec\(", r"ast\.parse", r"EnableCCD", r"SolverType", r"TGS", r"CreateStiffnessAttr\(4400",
    r"expert", r"\.get\(['\"]cue",
])
def test_no_forbidden_dependency_or_expert_input(pattern):
    hits = [p.name for p, s in SOURCES.items() if any(re.search(pattern, l) for l in _code_lines(s))
            if p.name != "test_boundaries.py"]
    assert hits == []


def test_timeline_has_no_defaults():
    from dataclasses import replace
    from mind2act_eval.tasks.S1.mechanism.schedule import timeline_config
    timeline = dict(cue_seconds=2., rotation_seconds=3., stop_seconds=.6, cue_gap_s=.2)
    for missing in timeline:
        incomplete = {key: value for key, value in timeline.items() if key != missing}
        with pytest.raises(ValueError, match="must give the timeline explicitly"):
            timeline_config(replace(spec("easy"), config=incomplete))
    assert timeline_config(replace(spec("easy"), config=timeline)) == timeline
    formal = dict(timeline, test_source=dict(instance_set="test50-r5", split="test"))
    assert timeline_config(replace(spec("easy"), config=formal)) == dict(timeline, cue_seconds=1., cue_gap_s=0.)
    assert formal["cue_seconds"] == 2. and formal["cue_gap_s"] == .2


def test_build_rejects_non_120hz_clock_before_touching_sim():
    from mind2act_eval.contracts import ClockSpec
    from mind2act_eval.tasks.S1.scene import build
    bad = InstanceSpec(task_id="S1", instance_id="i", difficulty="easy", source_id="s", config_version="1",
                       asset_version="1", protocol_version="v1", clock=ClockSpec(240, 30, 30))
    with pytest.raises(ValueError, match="120 Hz"):
        build(bad, None)


def test_result_metadata_reports_timeline():
    text = SOURCES[ROOT / "scene.py"]
    assert "def result_metadata" in text and "s1_timeline" in text
