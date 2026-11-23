"""Audit and repair missing authored root-layer metersPerUnit/upAxis on USD overlays.

Portable utility (stdlib + OpenUSD `pxr` only) for Mind2Act USD archive maintenance.
Never copies OpenUSD's 0.01 fallback: only propagates explicitly authored stage
metadata from the layer stack into the root layer, then reopens in a fresh stage to
verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

STAGE_FIELDS = (
    "metersPerUnit",
    "upAxis",
    "timeCodesPerSecond",
    "framesPerSecond",
    "startTimeCode",
    "endTimeCode",
    "defaultPrim",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_usd_root(path: Path) -> dict:
    from pxr import Usd, UsdGeom

    resolved = Path(path).resolve(strict=True)
    stage = Usd.Stage.Open(str(resolved))
    if stage is None:
        raise ValueError(f"failed to open USD stage: {resolved}")
    root = stage.GetRootLayer()
    authored_units = bool(root.pseudoRoot.HasInfo("metersPerUnit"))
    authored_axis = bool(root.pseudoRoot.HasInfo("upAxis"))
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "root_has_meters_per_unit": authored_units,
        "root_has_up_axis": authored_axis,
        "effective_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "effective_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
    }


def repair_usd_root_units(
    path: Path,
    *,
    expected_meters_per_unit: float = 1.0,
    expected_up_axis: str = "Z",
    dry_run: bool = False,
) -> dict:
    """Propagate explicitly authored stage metadata from sublayers to a USD root layer."""
    from pxr import Sdf, Usd

    resolved = Path(path).resolve(strict=True)
    before_sha = file_sha256(resolved)
    stage = Usd.Stage.Open(str(resolved))
    if stage is None:
        raise ValueError(f"failed to open USD stage: {resolved}")

    root = stage.GetRootLayer()
    layers = stage.GetLayerStack(includeSessionLayers=False)
    inherited = {}

    authored = {}
    for field in STAGE_FIELDS:
        for layer in layers:
            if layer.pseudoRoot.HasInfo(field):
                value = layer.pseudoRoot.GetInfo(field)
                authored[field] = value
                if layer != root:
                    inherited[field] = {"value": value, "source_layer": layer.identifier}
                break

    expected = {"metersPerUnit": expected_meters_per_unit, "upAxis": expected_up_axis}
    if any(field not in authored for field in expected):
        raise ValueError(f"overlay source graph lacks explicit metersPerUnit/upAxis: {resolved}")
    if any(authored[field] != value for field, value in expected.items()):
        raise ValueError(
            f"stage units/axis mismatch ({authored['metersPerUnit']}, {authored['upAxis']}) vs "
            f"expected ({expected_meters_per_unit}, {expected_up_axis}): {resolved}"
        )

    status = "already_authored"
    if inherited:
        status = "would_repair"
        if not dry_run:
            for field in inherited:
                root.pseudoRoot.SetInfo(field, authored[field])
            if not root.Save():
                raise OSError(f"failed to save USD root layer: {resolved}")
            reopened = Sdf.Layer.OpenAsAnonymous(str(resolved))
            if any(not reopened.pseudoRoot.HasInfo(field) or reopened.pseudoRoot.GetInfo(field) != value
                   for field, value in expected.items()):
                raise ValueError(f"saved root layer has unexpected stage units/axis: {resolved}")
            status = "repaired"

    return {
        "path": str(resolved),
        "status": status,
        "before_sha256": before_sha,
        "after_sha256": file_sha256(resolved),
        "meters_per_unit": authored["metersPerUnit"],
        "up_axis": authored["upAxis"],
        "inherited": inherited,
    }


def discover_usd_files(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            for ext in ("*.usda", "*.usd", "*.usdc"):
                found.extend(sorted(p.rglob(ext)))
        else:
            raise FileNotFoundError(p)
    # Deterministic de-duplicated order
    seen: set[Path] = set()
    ordered: list[Path] = []
    for item in found:
        key = item.resolve()
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path, help="USD files or directories to inspect/repair")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Verify authored root units and up axis")
    mode.add_argument("--dry-run", action="store_true", help="Report planned repairs without modifying files")
    mode.add_argument("--apply", action="store_true", help="Repair missing root-layer stage metadata in place")
    parser.add_argument("--expected-meters-per-unit", type=float, default=1.0)
    parser.add_argument("--expected-up-axis", default="Z")
    parser.add_argument("--receipt", type=Path, default=None, help="Optional JSON receipt output path")
    args = parser.parse_args(argv)

    files = discover_usd_files(args.targets)
    if args.check_only:
        rows = [inspect_usd_root(f) for f in files]
        missing = [
            row for row in rows
            if not row["root_has_meters_per_unit"] or not row["root_has_up_axis"]
            or row["effective_meters_per_unit"] != args.expected_meters_per_unit
            or row["effective_up_axis"] != args.expected_up_axis
        ]
        report = {
            "schema": "mind2act.usd_stage_units_audit.v1",
            "mode": "check_only",
            "total": len(rows),
            "missing_or_mismatched": len(missing),
            "rows": rows,
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.receipt is not None:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(payload)
        print(payload, end="")
        return 1 if missing else 0

    rows = [
        repair_usd_root_units(
            f,
            expected_meters_per_unit=args.expected_meters_per_unit,
            expected_up_axis=args.expected_up_axis,
            dry_run=args.dry_run,
        )
        for f in files
    ]
    report = {
        "schema": "mind2act.usd_stage_units_repair.v1",
        "mode": "dry_run" if args.dry_run else "apply",
        "total": len(rows),
        "repaired": sum(1 for r in rows if r["status"] == "repaired"),
        "would_repair": sum(1 for r in rows if r["status"] == "would_repair"),
        "already_authored": sum(1 for r in rows if r["status"] == "already_authored"),
        "rows": rows,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
