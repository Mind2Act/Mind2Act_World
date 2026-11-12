"""Bind the training scene and the current public input without seed substitution."""
import json
from pathlib import Path

from mind2act_eval.contracts import to_json
from .mechanism.case import make_case

SOURCE_ID = "e2-fingertip-quality-v2-frozen"
RUBRIC = "e2-spec-20261030-v1"
CONFIG = Path(__file__).with_name("config")


def instruction():
    return json.loads((CONFIG / "instructions.json").read_text())["entries"][0]["en"]


def validate(spec):
    if spec.task_id != "E2" or spec.source_id != SOURCE_ID:
        raise ValueError("E2 requires the selected training source identity")
    if spec.config_version != RUBRIC:
        raise ValueError("E2 requires the current e2-spec-20261030-v1 rules")
    if spec.clock.physics_hz != 120 or spec.clock.control_hz not in (30, 120) or spec.clock.rgb_hz != 30:
        raise ValueError("E2 requires 120 Hz physics, 30/120 Hz control and 30 Hz RGB")
    case = make_case(spec.difficulty, spec.seed)
    if to_json(spec.config["source_case"]) != case:
        raise ValueError("E2 training source case/seed/difficulty binding changed")
    if spec.config["instruction"] != instruction():
        raise ValueError("E2 instruction differs from the delivered canonical input")
    if not {"station", "font", "x5"}.issubset(spec.assets):
        raise ValueError("E2 requires resolver-owned station/font/x5 assets")
    return case
