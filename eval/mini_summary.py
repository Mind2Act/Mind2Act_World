"""Aggregate verified Mind2Act Mini World (720-episode) model results."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import json
import math
from math import isfinite
from pathlib import Path
import re

from mind2act_eval.budgets import task_policy_limits
from mind2act_eval.contracts import ConsumerKind, InstanceSpec, Result
from mind2act_eval.mini_layouts import (
    CATALOGUE_SHA256,
    CASES_PER_CELL,
    MINI_DIFFICULTIES,
    MINI_MODES,
    MINI_TASKS,
    catalogue,
    validate as validate_mini,
)
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.sealing import SEAL_NAME, validate_worker_seal
from mind2act_eval.tasks.M1.scorer import RULE_VERSION as M1_SCORING_VERSION

SCHEMA = "mind2act_eval.mini_summary.v1"
FAMILY_TASKS = {
    "mind": ("M1", "M2"),
    "act": ("DR1", "PR1"),
    "coupling": ("CP02", "CP05", "CP06", "CP10"),
}
FAMILY_WEIGHTS = {"mind": 0.25, "act": 0.25, "coupling": 0.50}
_ATTEMPT_RE = re.compile(r"^(episode-[0-9a-f]+)-attempt-(\d+)$")


def expected_scoring_version(task_id: str) -> str:
    return M1_SCORING_VERSION if task_id == "M1" else SCORING_VERSION


def _select_attempt_dirs(results_root: Path) -> list[Path]:
    root = Path(results_root).resolve(strict=True)
    grouped: dict[tuple[Path, str], tuple[int, Path]] = {}
    standalone: list[Path] = []
    for spec_path in sorted(root.rglob("spec.json")):
        attempt_dir = spec_path.parent
        if not (attempt_dir / "result.json").is_file() and not (attempt_dir / SEAL_NAME).is_file():
            continue
        match = _ATTEMPT_RE.match(attempt_dir.name)
        if match is not None:
            key = (attempt_dir.parent, match.group(1))
            attempt_num = int(match.group(2))
            prev = grouped.get(key)
            if prev is None or attempt_num > prev[0]:
                grouped[key] = (attempt_num, attempt_dir)
        else:
            standalone.append(attempt_dir)
    selected = [item[1] for item in sorted(grouped.values(), key=lambda x: str(x[1]))]
    selected.extend(standalone)
    return sorted(selected)


def _validate_loaded_result(attempt_dir: Path, spec: InstanceSpec, row: dict,
                            result: Result, *, strict_seal: bool) -> None:
    if not result.infra_valid or not result.model_tested or result.score is None:
        raise ValueError(f"invalid or unscored model episode in {attempt_dir}")
    if (result.task_id, result.instance_id) != (spec.task_id, spec.instance_id):
        raise ValueError(f"spec/result identity mismatch in {attempt_dir}")
    expected_sv = expected_scoring_version(spec.task_id)
    if result.scoring_version != expected_sv:
        raise ValueError(
            f"unexpected scoring_version in {attempt_dir}: {result.scoring_version} != {expected_sv}"
        )
    if result.policy_kind is None or result.budget_version is None:
        raise ValueError(f"scored model result requires explicit policy_kind and budget_version: {attempt_dir}")
    expected_limits, _ = task_policy_limits(spec, result.policy_kind, result.budget_version)
    if result.budget_limits:
        if (result.budget_limits.get("physics_ticks") != expected_limits["physics_ticks"]
                or result.budget_limits.get("wall_s") != expected_limits["wall_s"]):
            raise ValueError(f"result budget_limits differ from formal policy budget in {attempt_dir}")
    variation_meta = result.task_metadata.get("variation")
    if spec.variation.mode == "random":
        if (not isinstance(variation_meta, Mapping)
                or variation_meta.get("mode") != "random"
                or variation_meta.get("applied") is not True
                or variation_meta.get("layout_id") != row["layout_id"]
                or variation_meta.get("layout_sha256") != row["layout_sha256"]):
            raise ValueError(f"unverified random variation metadata in {attempt_dir}")
    elif variation_meta is not None:
        raise ValueError(f"unexpected variation metadata on clean episode in {attempt_dir}")
    score = float(result.score)
    if not isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"score outside [0, 100] in {attempt_dir}: {score}")
    score_path = attempt_dir / "score.json"
    if strict_seal or score_path.is_file():
        score_row = json.loads(score_path.read_text())
        if not isinstance(score_row, Mapping) or not math.isclose(
            float(score_row["value"]), score, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"score.json value disagrees with result.json in {attempt_dir}")
    if "task_success" in result.score_components:
        if bool(result.score_components["task_success"]) != (score == 100.0):
            raise ValueError(f"task_success disagrees with score in {attempt_dir}")


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _aggregate_subset(cell_means: dict[tuple[str, str, str], tuple[float, float]],
                      *, modes: tuple[str, ...], difficulties: tuple[str, ...]) -> dict:
    by_task = {}
    for task_id in MINI_TASKS:
        scores, srs = [], []
        for mode in modes:
            for diff in difficulties:
                cell = cell_means.get((mode, task_id, diff))
                if cell is not None:
                    scores.append(cell[0])
                    srs.append(cell[1])
        by_task[task_id] = {
            "score": _mean(scores),
            "sr": _mean(srs),
            "cells_present": len(scores),
            "cells_expected": len(modes) * len(difficulties),
        }
    by_family = {}
    for family, tasks in FAMILY_TASKS.items():
        fam_scores = [by_task[t]["score"] for t in tasks if by_task[t]["score"] is not None]
        fam_srs = [by_task[t]["sr"] for t in tasks if by_task[t]["sr"] is not None]
        by_family[family] = {
            "score": _mean(fam_scores) if len(fam_scores) == len(tasks) else None,
            "sr": _mean(fam_srs) if len(fam_srs) == len(tasks) else None,
            "tasks_present": len(fam_scores),
            "tasks_expected": len(tasks),
        }
    all_task_scores = [by_task[t]["score"] for t in MINI_TASKS if by_task[t]["score"] is not None]
    all_task_srs = [by_task[t]["sr"] for t in MINI_TASKS if by_task[t]["sr"] is not None]
    complete_families = all(by_family[f]["score"] is not None for f in FAMILY_TASKS)
    weighted_score = (
        sum(FAMILY_WEIGHTS[f] * by_family[f]["score"] for f in FAMILY_TASKS)
        if complete_families else None
    )
    weighted_sr = (
        sum(FAMILY_WEIGHTS[f] * by_family[f]["sr"] for f in FAMILY_TASKS)
        if complete_families else None
    )
    return {
        "weighted_score": weighted_score,
        "weighted_sr": weighted_sr,
        "unweighted_score": _mean(all_task_scores) if len(all_task_scores) == len(MINI_TASKS) else None,
        "unweighted_sr": _mean(all_task_srs) if len(all_task_srs) == len(MINI_TASKS) else None,
        "families": by_family,
        "tasks": by_task,
    }


def summarize_results(results_root: Path, *, mode_filter: str = "all",
                      strict_seal: bool = True, allow_incomplete: bool = False) -> dict:
    if mode_filter not in ("all", "clean", "random"):
        raise ValueError("mode_filter must be 'all', 'clean', or 'random'")
    target_modes = MINI_MODES if mode_filter == "all" else (mode_filter,)
    expected_records = {
        key: row for key, row in catalogue().items() if key[0] in target_modes
    }

    seen: dict[tuple[str, str], tuple[dict, Result, Path]] = {}
    failed_attempts: list[str] = []
    protocols: set[tuple[str, str]] = set()
    for attempt_dir in _select_attempt_dirs(results_root):
        spec = InstanceSpec.from_dict(json.loads((attempt_dir / "spec.json").read_text()))
        row = validate_mini(spec)
        key = (row["mode"], row["instance_id"])
        if key[0] not in target_modes:
            continue

        result_path = attempt_dir / "result.json"
        seal_path = attempt_dir / SEAL_NAME
        if allow_incomplete and result_path.is_file() and not seal_path.is_file():
            raw_res = Result(**json.loads(result_path.read_text()))
            if not raw_res.infra_valid:
                failed_attempts.append(f"{key[0]}/{key[1]}")
                continue

        if strict_seal:
            if not seal_path.is_file():
                raise ValueError(f"missing {SEAL_NAME} in {attempt_dir}")
            result, _ = validate_worker_seal(
                attempt_dir, expected_consumer=ConsumerKind.MODEL, strict=True
            )
        else:
            result = Result(**json.loads(result_path.read_text()))
            if result.consumer_kind != ConsumerKind.MODEL:
                raise ValueError(f"non-model consumer in {attempt_dir}: {result.consumer_kind}")
            if allow_incomplete and not result.infra_valid:
                failed_attempts.append(f"{key[0]}/{key[1]}")
                continue

        _validate_loaded_result(attempt_dir, spec, row, result, strict_seal=strict_seal)
        if key in seen:
            raise ValueError(
                f"duplicate episode result for {key[0]}/{key[1]}: {seen[key][2]} vs {attempt_dir}"
            )
        protocols.add((result.policy_kind, result.budget_version))
        seen[key] = (row, result, attempt_dir)

    if not seen:
        raise ValueError(f"no valid Mini World episode results found under {results_root}")
    if len(protocols) != 1:
        raise ValueError(f"mixed policy protocols in campaign: {sorted(protocols)}")
    policy_kind, budget_version = next(iter(protocols))

    missing_keys = sorted(set(expected_records) - set(seen))
    if missing_keys and not allow_incomplete:
        preview = ", ".join(f"{m}/{i}" for m, i in missing_keys[:5])
        raise ValueError(
            f"incomplete Mini World campaign ({len(seen)}/{len(expected_records)} episodes; "
            f"missing {len(missing_keys)}, e.g. {preview})"
        )

    cell_rows: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (mode, _instance_id), (row, result, _dir) in sorted(seen.items()):
        sc = float(result.score)
        sr = 100.0 if sc == 100.0 else 0.0
        cell_rows[(mode, row["task_id"], row["difficulty"])].append((sc, sr))

    cell_means: dict[tuple[str, str, str], tuple[float, float]] = {}
    cells_detail = {}
    for mode in target_modes:
        for task_id in MINI_TASKS:
            for diff in MINI_DIFFICULTIES:
                vals = cell_rows.get((mode, task_id, diff), [])
                cell_key = f"{mode}/{task_id}/{diff}"
                if len(vals) == CASES_PER_CELL or (vals and allow_incomplete):
                    mean_sc = sum(v[0] for v in vals) / len(vals)
                    mean_sr = sum(v[1] for v in vals) / len(vals)
                    cell_means[(mode, task_id, diff)] = (mean_sc, mean_sr)
                else:
                    mean_sc, mean_sr = None, None
                cells_detail[cell_key] = {
                    "episodes_present": len(vals),
                    "episodes_expected": CASES_PER_CELL,
                    "score": mean_sc,
                    "sr": mean_sr,
                }

    overall = _aggregate_subset(cell_means, modes=target_modes, difficulties=MINI_DIFFICULTIES)
    by_mode = {
        mode: _aggregate_subset(cell_means, modes=(mode,), difficulties=MINI_DIFFICULTIES)
        for mode in target_modes
    }
    by_difficulty = {
        diff: _aggregate_subset(cell_means, modes=target_modes, difficulties=(diff,))
        for diff in MINI_DIFFICULTIES
    }

    return {
        "schema": SCHEMA,
        "catalogue_sha256": CATALOGUE_SHA256,
        "scoring_version": SCORING_VERSION,
        "task_scoring_versions": {"M1": M1_SCORING_VERSION, "default": SCORING_VERSION},
        "mode_filter": mode_filter,
        "seal_verified": strict_seal,
        "complete": len(missing_keys) == 0,
        "policy_kind": policy_kind,
        "budget_version": budget_version,
        "episodes_present": len(seen),
        "episodes_expected": len(expected_records),
        "failed_attempts_skipped": sorted(failed_attempts),
        "missing_episodes": [f"{m}/{i}" for m, i in missing_keys],
        "weights": dict(FAMILY_WEIGHTS),
        "overall": overall,
        "by_mode": by_mode,
        "by_difficulty": by_difficulty,
        "cells": cells_detail,
    }


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def format_table(summary: dict) -> str:
    ov = summary["overall"]
    lines = [
        f"Mind2Act Mini World Summary ({summary['episodes_present']}/{summary['episodes_expected']} episodes, "
        f"mode={summary['mode_filter']}, complete={summary['complete']}, seal_verified={summary['seal_verified']})",
        f"Protocol: policy_kind={summary['policy_kind']} budget_version={summary['budget_version']} "
        f"scoring_version={summary['scoring_version']} (+M1 {M1_SCORING_VERSION})",
        f"Overall Score (0.25 Mind + 0.25 Act + 0.50 Mind2Act): {_fmt(ov['weighted_score'])} | "
        f"Weighted SR: {_fmt(ov['weighted_sr'])}%",
        f"Unweighted Score (8 tasks equal): {_fmt(ov['unweighted_score'])} | "
        f"Unweighted SR: {_fmt(ov['unweighted_sr'])}%",
    ]
    for family in ("mind", "act", "coupling"):
        fam = ov["families"][family]
        label = "Mind2Act" if family == "coupling" else family.capitalize()
        lines.append(f"  {label:8s}: Score {_fmt(fam['score'])} | SR {_fmt(fam['sr'])}%")
    for mode, row in summary["by_mode"].items():
        lines.append(f"  Mode={mode:6s}: Score {_fmt(row['weighted_score'])} | SR {_fmt(row['weighted_sr'])}%")
    for diff, row in summary["by_difficulty"].items():
        lines.append(f"  Tier={diff:6s}: Score {_fmt(row['weighted_score'])} | SR {_fmt(row['weighted_sr'])}%")
    return "\n".join(lines)


def add_arguments(parser: argparse.Namespace | argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", required=True, help="Directory containing episode outputs")
    parser.add_argument("--mode", choices=("all", "clean", "random"), default="all",
                        help="Summarize all 720 episodes (default) or a single 360-episode mode")
    parser.add_argument("--output", help="Optional path to write the full JSON summary receipt")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    parser.add_argument("--no-seal-check", action="store_true",
                        help="Skip strict worker-seal.json verification (diagnostic only)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Allow partial episode coverage and mark complete=false")


def run_cli(args: argparse.Namespace) -> int:
    summary = summarize_results(
        Path(args.results_root),
        mode_filter=args.mode,
        strict_seal=not args.no_seal_check,
        allow_incomplete=args.allow_incomplete,
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_table(summary))
    return 0
