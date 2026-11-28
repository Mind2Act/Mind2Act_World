from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import pytest

from mind2act_eval.budgets import E2E_BUDGET_VERSION, task_policy_limits
from mind2act_eval.contracts import ConsumerKind, InstanceSpec, Result, Termination, to_json
from mind2act_eval.internal import _episode_assets, read_spec
from mind2act_eval.mini_layouts import (
    CASES_PER_CELL,
    CATALOGUE_SHA256,
    MINI_DIFFICULTIES,
    MINI_MODES,
    MINI_TASKS,
    TOTAL_EPISODES,
    catalogue,
)
from mind2act_eval.mini_summary import _aggregate_subset, expected_scoring_version
from mind2act_eval.public_cli import main as public_cli_main
from mind2act_eval.public_layouts import catalogue as public_catalogue, validate as validate_public
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.sealing import seal_worker_outputs
from mind2act_eval.spec_identity import spec_digest
from mind2act_eval.tasks.M1.scorer import RULE_VERSION as M1_SCORING_VERSION
from mind2act_eval.tasks.PR1.scene import validate as validate_pr1
from mind2act_eval.variations import base_spec_digest

REAL_PR1_CLEAN = {
    "asset_version": "assets-final-20261029+laneA-replay-overlay-nvidia-mdl-source-relocation",
    "assets": {
        "font_sans": "files/content/ae/ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280/ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280.ttf",
        "font_sans_bold": "files/content/5c/5c1247acef7f2b8522a31742c76d6adcb5569bacc0be7ceaa4dc39dd252ce895/5c1247acef7f2b8522a31742c76d6adcb5569bacc0be7ceaa4dc39dd252ce895.ttf",
        "portable-mdl-0": "sdk/mdl/core/Base",
        "portable-mdl-1": "sdk/mdl/core/mdl",
        "portable-mdl-2": "sdk/mdl/rtx",
        "station": "layers/PR1/medium/station/station-relocation/0000.usda",
        "x5": "shared/robots/x5/ARX.usd",
    },
    "clock": {"control_hz": 30, "physics_hz": 120, "rgb_hz": 30},
    "config": {
        "cameras": {
            "head": {"path": "/World/HeadCamera", "resolution": [640, 480]},
            "left_wrist": {"path": "/World/RobotLeft/Asset/camera/PhysCoWristCamera", "resolution": [640, 480]},
            "right_wrist": {"path": "/World/RobotRight/Asset/camera/PhysCoWristCamera", "resolution": [640, 480]},
        },
        "fixture_y_offset_m": -0.05,
        "initial_ohm": 10,
        "instruction": (
            "Adjust the resistance between the left terminal and wiper to 28.5 ohm, "
            "then release the slider and keep that reading for 0.5 seconds."
        ),
        "layout_split": "mini",
        "material_mdl_search_asset_ids": ["portable-mdl-0", "portable-mdl-1", "portable-mdl-2"],
        "robot_rest": [
            0.0,
            1.1344640137963142,
            1.7453292519943295,
            -1.1344640137963142,
            0.0,
            0.0,
            0.044,
            0.044,
        ],
        "target_ohm": 28.5,
    },
    "config_version": "pr1-measured-stability-20260930-v2",
    "difficulty": "medium",
    "instance_id": "PR1_medium_04b867fb0d33c0b3",
    "protocol_version": "v1",
    "seed": 5190064999731,
    "source_id": "pr1-original-source-uniform-run925de9",
    "task_id": "PR1",
}

REAL_PR1_RANDOM = {
    **REAL_PR1_CLEAN,
    "variation": {
        "asset_manifest_sha256": "22d522417bad544e9cae2cdadd81ae4788a0fb4324543e0ced3799fcb1841263",
        "base_spec_sha256": "305c7e5ad56e1685c54448be6a735e1efc62dbed0a47e206c7b7a39c2238c687",
        "layout_id": "PR1_medium_04b867fb0d33c0b3",
        "layout_sha256": "6c3be6b56f30795ddb952de540ee86f6f9c01a8ae943b8cb7fbd177a4b8e4005",
        "mode": "random",
    },
}


def _write_sealed_attempt(
    attempt_dir: Path,
    spec_dict: dict,
    *,
    score: float,
    policy_kind: str = "e2e",
    budget_version: str = E2E_BUDGET_VERSION,
    scoring_version: str | None = None,
) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    spec = InstanceSpec.from_dict(spec_dict)
    limits, _ = task_policy_limits(spec, policy_kind, budget_version)
    (attempt_dir / "spec.json").write_text(json.dumps(spec_dict, indent=2, sort_keys=True) + "\n")
    (attempt_dir / "evidence.json").write_text(
        json.dumps({"task_id": spec.task_id, "instance_id": spec.instance_id, "tick": 120, "values": {}}) + "\n"
    )
    (attempt_dir / "score.json").write_text(json.dumps({"value": score, "components": {}}) + "\n")
    (attempt_dir / "videos").mkdir(exist_ok=True)
    (attempt_dir / "videos" / "head.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (attempt_dir / "camera-receipt.json").write_text(
        json.dumps(
            {
                "schema": "mind2act_eval.native_camera_receipt.v1",
                "status": "verified",
                "initial_frame_encoded": True,
                "cameras": {"head": {"valid": True}},
                "recording_cameras": ["head"],
            }
        )
        + "\n"
    )
    (attempt_dir / "shutdown-receipt.json").write_text(
        json.dumps(
            {
                "schema": "mind2act_eval.shutdown_receipt.v1",
                "status": "os_exit_prepared",
                "teardown": "os_exit",
                "phases": [
                    {"name": "recorder", "status": "complete"},
                    {"name": "app.close", "status": "skipped"},
                ],
            }
        )
        + "\n"
    )
    task_metadata = {}
    artifacts = {
        "native_camera_receipt": "camera-receipt.json",
        "native_shutdown_receipt": "shutdown-receipt.json",
        "video_head": "videos/head.mp4",
    }
    if spec.variation.mode == "random":
        (attempt_dir / "variation-receipt.json").write_text(json.dumps({"mode": "random"}) + "\n")
        artifacts["variation_receipt"] = "variation-receipt.json"
        task_metadata["variation"] = {
            **to_json(spec.variation),
            "applied": True,
            "authored": True,
            "receipt": "variation-receipt.json",
        }
    calls_kwargs = (
        {"calls_used": 4, "calls_limit": 150, "decision_retry_limit": 12}
        if policy_kind in ("agent", "coding_agent")
        else {}
    )
    result = Result(
        task_id=spec.task_id,
        instance_id=spec.instance_id,
        score=score,
        termination=Termination.COMPLETED if score == 100.0 else Termination.PHYSICS_BUDGET,
        infra_valid=True,
        consumer_kind=ConsumerKind.MODEL,
        model_tested=True,
        ticks=120,
        artifacts=artifacts,
        score_components={"task_success": score == 100.0},
        task_metadata=task_metadata,
        policy_kind=policy_kind,
        budget_version=budget_version,
        budget_limits=limits,
        scoring_version=scoring_version or expected_scoring_version(spec.task_id),
        **calls_kwargs,
    )
    (attempt_dir / "result.json").write_text(json.dumps(to_json(result), indent=2, sort_keys=True) + "\n")
    for fname, stage, extra in (
        ("run-episode-returned.json", "run_episode_returned", {"infra_valid": True}),
        ("worker-returned.json", "worker_returned", {"exit_code": 0}),
        ("cli-returned.json", "cli_returned", {"exit_code": 0}),
    ):
        (attempt_dir / fname).write_text(
            json.dumps({"schema": "mind2act_eval.worker_lifecycle.v1", "stage": stage, "pid": 4242, **extra}) + "\n"
        )
    seal_worker_outputs(attempt_dir, expected_consumer=ConsumerKind.MODEL)
    return attempt_dir


def test_mini_catalogue_structure_and_pairing():
    records = catalogue()
    assert MINI_TASKS == ("M1", "M2", "DR1", "PR1", "CP02", "CP05", "CP06", "CP10")
    assert MINI_DIFFICULTIES == ("easy", "medium", "hard")
    assert len(records) == TOTAL_EPISODES == 720
    assert len(CATALOGUE_SHA256) == 64

    cell_counts = Counter()
    for (mode, instance_id), row in records.items():
        assert mode in MINI_MODES
        assert row["task_id"] in MINI_TASKS
        assert row["difficulty"] in MINI_DIFFICULTIES
        assert row["instance_id"] == instance_id
        cell_counts[(mode, row["task_id"], row["difficulty"])] += 1
        if mode == "clean":
            assert row["spec_sha256"] == row["base_spec_sha256"]
            assert row["variation_manifest"] is None
        else:
            clean_partner = records[("clean", instance_id)]
            assert row["base_spec_sha256"] == clean_partner["spec_sha256"]
            assert row["seed"] == clean_partner["seed"]
            assert row["variation_manifest"] == f"variation-{row['task_id']}.json"
            assert row["layout_id"] == instance_id

    assert len(cell_counts) == len(MINI_MODES) * len(MINI_TASKS) * len(MINI_DIFFICULTIES) == 48
    assert set(cell_counts.values()) == {CASES_PER_CELL}


def test_real_mini_specs_validate_and_reject_tampering(tmp_path):
    clean_path = tmp_path / "pr1_clean.json"
    random_path = tmp_path / "pr1_random.json"
    clean_path.write_text(json.dumps(REAL_PR1_CLEAN))
    random_path.write_text(json.dumps(REAL_PR1_RANDOM))

    clean_spec = read_spec(clean_path)
    random_spec = read_spec(random_path)
    assert validate_public(clean_spec)["mode"] == "clean"
    assert validate_public(random_spec)["mode"] == "random"
    validate_pr1(clean_spec)
    validate_pr1(random_spec)

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "variation-PR1.json").write_text("{}")
    for rel in random_spec.assets.values():
        target = asset_root / rel
        if rel.startswith("sdk/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"ok")
    episode_assets = _episode_assets(
        random_spec, SimpleNamespace(asset_root=str(asset_root), variation_manifest=None)
    )
    assert episode_assets.resolve("variation_manifest") == (asset_root / "variation-PR1.json").resolve()
    with pytest.raises(ValueError, match="variation manifest differs"):
        _episode_assets(
            random_spec, SimpleNamespace(asset_root=str(asset_root), variation_manifest="variation-M1.json")
        )

    tampered = json.loads(json.dumps(REAL_PR1_CLEAN))
    tampered["config"]["target_ohm"] = 29.0
    with pytest.raises(ValueError, match="mini layout differs"):
        validate_public(InstanceSpec.from_dict(tampered))

    forged = json.loads(json.dumps(REAL_PR1_CLEAN))
    forged["config"]["test_source"] = {"split": "test"}
    with pytest.raises(ValueError, match="private source provenance"):
        validate_public(InstanceSpec.from_dict(forged))

    non_mini_random = json.loads(json.dumps(REAL_PR1_RANDOM))
    non_mini_random["config"].pop("layout_split")
    with pytest.raises(ValueError, match="clean train/validation layouts"):
        validate_public(InstanceSpec.from_dict(non_mini_random))


def test_public_development_preserved_and_equal_1_1_1_tier_weighting():
    assert list(public_catalogue().values())
    cell_means = {}
    diff_vals = {"easy": 60.0, "medium": 30.0, "hard": 0.0}
    fam_offsets = {"M1": 10.0, "M2": 10.0, "DR1": 20.0, "PR1": 20.0, "CP02": 0.0, "CP05": 0.0, "CP06": 0.0, "CP10": 0.0}
    for mode in MINI_MODES:
        for t in MINI_TASKS:
            for d in MINI_DIFFICULTIES:
                cell_means[(mode, t, d)] = (diff_vals[d] + fam_offsets[t], 50.0)
    agg = _aggregate_subset(cell_means, modes=MINI_MODES, difficulties=MINI_DIFFICULTIES)
    # Equal 1:1:1 across easy(60), medium(30), hard(0) -> base 30.0 + family offset
    assert agg["families"]["mind"]["score"] == pytest.approx(40.0)
    assert agg["families"]["act"]["score"] == pytest.approx(50.0)
    assert agg["families"]["coupling"]["score"] == pytest.approx(30.0)
    # 0.25 * 40 + 0.25 * 50 + 0.50 * 30 == 37.5, and under 2+2+4 tasks unweighted_score is identical
    assert agg["weighted_score"] == pytest.approx(37.5)
    assert agg["unweighted_score"] == pytest.approx(37.5)


def test_summarize_mini_cli_seal_mode_filter_and_m1_scoring_version(tmp_path, monkeypatch):
    results_root = tmp_path / "campaign"
    _write_sealed_attempt(results_root / "episode-a001-attempt-1", REAL_PR1_CLEAN, score=100.0)
    _write_sealed_attempt(results_root / "episode-a002-attempt-1", REAL_PR1_RANDOM, score=20.0)

    # Without --allow-incomplete, partial 2/720 campaign must fail
    assert public_cli_main(["summarize-mini", "--results-root", str(results_root)]) == 2

    out_json = tmp_path / "summary.json"
    assert (
        public_cli_main(
            [
                "summarize-mini",
                "--results-root",
                str(results_root),
                "--allow-incomplete",
                "--output",
                str(out_json),
            ]
        )
        == 0
    )
    payload = json.loads(out_json.read_text())
    assert payload["seal_verified"] is True
    assert payload["complete"] is False
    assert payload["episodes_present"] == 2
    assert payload["cells"]["clean/PR1/medium"]["score"] == 100.0
    assert payload["cells"]["random/PR1/medium"]["score"] == 20.0

    # Unsealed failed random attempt must NOT break --mode clean, and must be skipped under --allow-incomplete
    failed_dir = results_root / "episode-a003-attempt-1"
    failed_dir.mkdir()
    random_other = next(
        r for (m, _), r in catalogue().items() if m == "random" and r["task_id"] == "DR1"
    )
    dr1_spec = {
        "task_id": "DR1",
        "instance_id": random_other["instance_id"],
        "difficulty": random_other["difficulty"],
        "source_id": "synthetic-dr1",
        "config_version": "v1",
        "asset_version": "v1",
        "protocol_version": "v1",
        "seed": random_other["seed"],
        "clock": {"physics_hz": 120, "control_hz": 30, "rgb_hz": 30},
        "config": {"layout_split": "mini"},
        "assets": {"station": "s.usda", "x5": "x.usd"},
        "variation": {
            "mode": "random",
            "layout_id": random_other["layout_id"],
            "layout_sha256": random_other["layout_sha256"],
            "base_spec_sha256": "0" * 64,
            "asset_manifest_sha256": random_other["asset_manifest_sha256"],
        },
    }
    dr1_base = base_spec_digest(InstanceSpec.from_dict(dr1_spec))
    dr1_spec["variation"]["base_spec_sha256"] = dr1_base
    dr1_spec_obj = InstanceSpec.from_dict(dr1_spec)

    m1_row = next(r for (m, _), r in catalogue().items() if m == "clean" and r["task_id"] == "M1")
    m1_spec = {
        "task_id": "M1",
        "instance_id": m1_row["instance_id"],
        "difficulty": m1_row["difficulty"],
        "source_id": "synthetic-m1",
        "config_version": "v1",
        "asset_version": "v1",
        "protocol_version": "v1",
        "seed": m1_row["seed"],
        "clock": {"physics_hz": 120, "control_hz": 30, "rgb_hz": 30},
        "config": {"layout_split": "mini"},
        "assets": {"station": "s.usda", "x5": "x.usd"},
    }
    m1_spec_obj = InstanceSpec.from_dict(m1_spec)

    patched_cat = dict(catalogue())
    patched_cat[("random", dr1_spec["instance_id"])] = MappingProxyType(
        {
            **dict(random_other),
            "base_spec_sha256": dr1_base,
            "spec_sha256": spec_digest(dr1_spec_obj),
        }
    )
    patched_cat[("clean", m1_spec["instance_id"])] = MappingProxyType(
        {
            **dict(m1_row),
            "base_spec_sha256": base_spec_digest(m1_spec_obj),
            "spec_sha256": spec_digest(m1_spec_obj),
        }
    )
    monkeypatch.setattr("mind2act_eval.mini_layouts.catalogue", lambda: MappingProxyType(patched_cat))
    monkeypatch.setattr("mind2act_eval.mini_summary.catalogue", lambda: MappingProxyType(patched_cat))

    (failed_dir / "spec.json").write_text(json.dumps(dr1_spec) + "\n")
    failed_res = Result(
        task_id="DR1",
        instance_id=dr1_spec["instance_id"],
        score=None,
        termination=Termination.WALL_TIMEOUT,
        infra_valid=False,
        consumer_kind=ConsumerKind.MODEL,
        model_tested=True,
        ticks=10,
        policy_kind="e2e",
        budget_version=E2E_BUDGET_VERSION,
        scoring_version=SCORING_VERSION,
    )
    (failed_dir / "result.json").write_text(json.dumps(to_json(failed_res)) + "\n")

    # M1 sealed episode using M1's authoritative RULE_VERSION must succeed
    _write_sealed_attempt(
        results_root / "episode-a004-attempt-1",
        m1_spec,
        score=100.0,
        scoring_version=M1_SCORING_VERSION,
    )

    clean_out = tmp_path / "clean_summary.json"
    assert (
        public_cli_main(
            [
                "summarize-mini",
                "--results-root",
                str(results_root),
                "--mode",
                "clean",
                "--allow-incomplete",
                "--output",
                str(clean_out),
            ]
        )
        == 0
    )
    clean_payload = json.loads(clean_out.read_text())
    assert clean_payload["episodes_present"] == 2
    assert clean_payload["cells"][f"clean/M1/{m1_spec['difficulty']}"]["score"] == 100.0

    all_out = tmp_path / "all_summary.json"
    assert (
        public_cli_main(
            [
                "summarize-mini",
                "--results-root",
                str(results_root),
                "--allow-incomplete",
                "--output",
                str(all_out),
            ]
        )
        == 0
    )
    all_payload = json.loads(all_out.read_text())
    assert all_payload["failed_attempts_skipped"] == [f"random/{dr1_spec['instance_id']}"]

    # Tampering score.json or result.json must fail even with --no-seal-check or strict seal
    score_file = results_root / "episode-a001-attempt-1" / "score.json"
    score_file.write_text(json.dumps({"value": 0.0, "components": {}}) + "\n")
    assert (
        public_cli_main(
            [
                "summarize-mini",
                "--results-root",
                str(results_root),
                "--no-seal-check",
                "--allow-incomplete",
            ]
        )
        == 2
    )
