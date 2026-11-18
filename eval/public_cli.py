"""Public command line for running a model episode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


_MODES = {
    "e2e": ("e2e", "mind2act_eval.adapters.xpolicylab:create", "Run an end-to-end policy through XPolicyLab"),
    "agent": ("agent", "mind2act_eval.adapters.m2a_agent:create", "Run the Mind2Act tool agent"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mind2act-eval", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a model episode", allow_abbrev=False)
    run.add_argument("--spec", required=True, help="Instance spec JSON path")
    run.add_argument("--asset-root", help="Root directory for the instance assets")
    mode = run.add_mutually_exclusive_group(required=True)
    for name, (_, _, description) in _MODES.items():
        mode.add_argument(f"--{name}", dest="mode", action="store_const", const=name, help=description)
    run.add_argument("--gpu", type=int, help="Physical GPU index")
    run.add_argument("--output-root", help="Output directory")
    run.add_argument("--video-hz", type=int, help="Video frames per second")
    run.add_argument("--record-wrist-cameras", action="store_true",
                     help="Record both wrist cameras")
    args = parser.parse_args(argv)
    config_path = Path.cwd() / "config" / f"{args.mode}.json"
    try:
        raw_config = config_path.read_bytes()
        config = json.loads(raw_config)
    except (OSError, ValueError) as exc:
        run.error(f"could not load {config_path}: {exc}")
    if not isinstance(config, dict):
        run.error(f"{config_path} must contain a JSON object")
    if args.mode == "agent":
        for field, factory in (
            ("client_factory", "mind2act_eval.adapters.roboprobe_client:create"),
            ("planner_factory", "mind2act_eval.adapters.roboprobe_planner:create"),
        ):
            if config.get(field) != factory:
                run.error(f"{config_path}: {field} must be {factory}")

    budget_mode, adapter, _ = _MODES[args.mode]
    arguments = ["run", f"--{budget_mode}", f"--spec={args.spec}", f"--adapter={adapter}",
                 f"--adapter-config={config_path}", f"--adapter-config-sha256={hashlib.sha256(raw_config).hexdigest()}"]
    for field in ("asset_root", "gpu", "output_root", "video_hz"):
        value = getattr(args, field)
        if value is not None:
            arguments.append(f"--{field.replace('_', '-')}={value}")
    if args.record_wrist_cameras:
        arguments.append("--record-wrist-cameras")

    from mind2act_eval.internal import main as internal_main
    return internal_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
