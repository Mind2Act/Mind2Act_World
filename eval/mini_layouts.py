"""Authoritative 720-episode Mind2Act Mini World catalogue and spec validation."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from types import MappingProxyType

CATALOGUE_SHA256 = "75466a41bb62a4fa65f4a70f17f2ea01b8aa102260fa2ee5276e2a96208b775a"
MINI_TASKS = ("M1", "M2", "DR1", "PR1", "CP02", "CP05", "CP06", "CP10")
MINI_DIFFICULTIES = ("easy", "medium", "hard")
MINI_MODES = ("clean", "random")
CASES_PER_CELL = 15
TOTAL_EPISODES = len(MINI_TASKS) * len(MINI_DIFFICULTIES) * len(MINI_MODES) * CASES_PER_CELL


@lru_cache(maxsize=1)
def catalogue():
    raw = files("mind2act_eval.env_cfg").joinpath("mini_layouts.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CATALOGUE_SHA256:
        raise ValueError("mini layout catalogue authority bytes differ")
    data = json.loads(raw)
    if (data.get("schema") != "mind2act_eval.mini_layouts.v1"
            or tuple(data.get("tasks", ())) != MINI_TASKS
            or tuple(data.get("difficulties", ())) != MINI_DIFFICULTIES
            or tuple(data.get("modes", ())) != MINI_MODES
            or data.get("cases_per_cell") != CASES_PER_CELL
            or data.get("total_episodes") != TOTAL_EPISODES
            or not isinstance(data.get("records"), list)
            or len(data["records"]) != TOTAL_EPISODES):
        raise ValueError("unsupported or incomplete mini layout catalogue")
    records = {(row["mode"], row["instance_id"]): MappingProxyType(dict(row)) for row in data["records"]}
    if len(records) != TOTAL_EPISODES:
        raise ValueError("duplicate mini layout identity")
    return MappingProxyType(records)


def validate(spec):
    """Verify that a Mini World spec matches the frozen 720-episode catalogue."""
    if (spec.config.get("layout_split") != "mini"
            or "test_source" in spec.config
            or "fixed50_origin" in spec.config
            or "train_source" in spec.config):
        raise ValueError("mini inputs must use layout_split='mini' without private source provenance")
    row = catalogue().get((spec.variation.mode, spec.instance_id))
    if row is None:
        raise ValueError(f"unrecognized mini layout instance: {spec.variation.mode}/{spec.instance_id}")
    if (spec.task_id, spec.difficulty, spec.seed) != (row["task_id"], row["difficulty"], row["seed"]):
        raise ValueError("mini layout task/difficulty/seed differs from catalogue")
    from mind2act_eval.spec_identity import spec_digest
    from mind2act_eval.variations import base_spec_digest
    if spec_digest(spec) != row["spec_sha256"] or base_spec_digest(spec) != row["base_spec_sha256"]:
        raise ValueError("mini layout differs from the released catalogue")
    if spec.variation.mode == "random":
        if (spec.variation.layout_id != row["layout_id"]
                or spec.variation.layout_sha256 != row["layout_sha256"]
                or spec.variation.base_spec_sha256 != row["base_spec_sha256"]
                or spec.variation.asset_manifest_sha256 != row["asset_manifest_sha256"]):
            raise ValueError("mini random variation pins differ from catalogue")
    return dict(row)
