"""Internal model, diagnostics and replay entry point."""
from __future__ import annotations

import argparse
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from mind2act_eval.budgets import add_model_mode_arguments, configure_model_mode
from mind2act_eval.contracts import BuildContext, ConsumerKind, InstanceSpec, Termination
from mind2act_eval.release_contract import path_check_session
from mind2act_eval.runner import EpisodeLimits, run_episode, write_json

_WORKER_LIFECYCLE_OUTPUT = None
_WORKER_EXPECTED_CONSUMER = None
_WORKER_FAILURE_EXIT_READY = False


def _write_worker_stage(output, stage, **details):
    try:
        write_json(Path(output) / (stage.replace("_", "-") + ".json"), {
            "schema": "mind2act_eval.worker_lifecycle.v1",
            "stage": stage,
            "pid": os.getpid(),
            **details,
        })
    except Exception as exc:
        print(f"worker lifecycle marker {stage} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def read_spec(path):
    spec = InstanceSpec.from_dict(json.loads(Path(path).read_text()))
    from mind2act_eval.public_layouts import validate
    validate(spec)
    return spec




def _episode_assets(spec, args):
    from mind2act_eval.sim import LocalAssets
    entries = dict(spec.assets)
    manifest = getattr(args, "variation_manifest", None)
    if spec.variation.mode == "random":
        if not manifest or Path(manifest).is_absolute():
            raise ValueError("random requires --variation-manifest relative to --asset-root")
        if "variation_manifest" in entries:
            raise ValueError("variation_manifest is reserved for the pinned deployment overlay")
        entries["variation_manifest"] = manifest
    elif manifest:
        raise ValueError("clean cannot request a variation manifest")
    return LocalAssets(Path(args.asset_root), entries, spec=spec)


def _material_search_asset_ids(spec):
    from mind2act_eval.public_layouts import validate
    validate(spec)
    if 'material_mdl_search_asset_ids' in spec.config:
        return spec.config['material_mdl_search_asset_ids']
    return {
        'P2': ('mdl0', 'mdl1', 'mdl2'),
        'CP06': ('portable-mdl-0', 'portable-mdl-1', 'portable-mdl-2'),
        'CP08': ('portable-mdl-0', 'portable-mdl-1', 'portable-mdl-2'),
        'CP09': ('portable-mdl-0', 'portable-mdl-1', 'portable-mdl-2'),
    }.get(spec.task_id, ())


def _check_expected_digest(raw, expected, label):
    """Optional caller-requested file check; absent expectations do no hashing."""
    if expected is not None:
        if raw is None or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(label + ' differs from the explicitly requested SHA256')


def _worker(args):
    global _WORKER_FAILURE_EXIT_READY
    _WORKER_FAILURE_EXIT_READY = False
    from mind2act_eval.runtime_release import guard_input_paths, validate_release_inputs, resolve_case_arguments
    resolve_case_arguments(args)
    guard_input_paths(args)
    validate_release_inputs(read_spec(args.spec), args)
    if getattr(args, 'policy_kind', None) is not None or getattr(args, 'budget_version', None) is not None:
        EpisodeLimits(args.physics_ticks, args.wall_s, getattr(args, 'policy_kind', None),
                      getattr(args, 'budget_version', None))
    if getattr(args, 'policy_kind', None) and (args.h5 or getattr(args, 'consumer', None)):
        raise ValueError('policy kind is reserved for scored model consumers')
    if any(getattr(args, key, None) is not None for key in ('consumer', 'probe_config', 'probe_config_sha256')):
        raise ValueError("diagnostic consumers are not part of the public runtime")
    from mind2act_eval.sim import LocalAssets, IsaacSimulator
    from mind2act_eval.tasks.catalog import load_task
    spec = read_spec(args.spec)
    output = Path(args.output)
    assets = _episode_assets(spec, args)
    adapter = None
    sim = None
    run_episode_started = False
    try:
        if args.backend == "demo":
            if spec.task_id != "demo":
                raise ValueError("CPU demo backend is restricted to the public demo task")
            from mind2act_eval.sim.demo import DemoSimulator, DemoRobot
            sim, robot = DemoSimulator(), DemoRobot()
        else:
            from mind2act_eval.robot import X5Robot
            x5_path = assets.resolve("x5")
            station_path = assets.resolve("station") if "station" in spec.assets else None
            material_receipt = None
            material_ids = _material_search_asset_ids(spec)
            if material_ids:
                from mind2act_eval.sim.material_closure import configure_mdl_search
                material_receipt = configure_mdl_search(assets, material_ids)
            if spec.variation.mode == "random":
                from mind2act_eval.variations import configure_material_search
                variation_materials = configure_material_search(spec, assets,
                    spec.config.get('material_mdl_search_asset_ids', ()))
                if variation_materials is not None:
                    material_receipt = variation_materials
            sim = IsaacSimulator(spec.clock, output, physical_gpu=args.gpu, cameras=spec.config.get("cameras"),
                                 ffmpeg=spec.config.get("ffmpeg", "ffmpeg"), video_hz=getattr(args, "video_hz", 6),
                                 record_wrist_cameras=getattr(args, "record_wrist_cameras", False),
                                 observe_wrist_cameras=(not args.h5
                                     and args.adapter == "mind2act_eval.adapters.xpolicylab:create"))
            if material_receipt is not None:
                from mind2act_eval.sim.material_closure import verify_native_mdl_search
                write_json(output / 'material-resolver.json', verify_native_mdl_search(material_receipt))
            robot = X5Robot(rgb_reader=sim.rgb, usd_path=x5_path, rest=spec.config.get("robot_rest"),
                            physics_hz=spec.clock.physics_hz)
            if station_path is not None:
                from mind2act_eval.sim.station_deployment import prepare_open_station
                sim.open_stage(prepare_open_station(spec, assets, output))
        sim.video_hz = getattr(args, "video_hz", 6)
        task = load_task(spec.task_id)
        if args.h5:
            from mind2act_eval.replay import H5Replay, ReplayMetadata
            metadata_bytes = Path(args.replay_metadata).read_bytes() if args.replay_metadata else None
            _check_expected_digest(metadata_bytes, args.replay_metadata_sha256, "replay metadata")
            metadata = ReplayMetadata(**json.loads(metadata_bytes)) if metadata_bytes is not None else None
            adapter = H5Replay(Path(args.h5), metadata, expected_sha256=args.h5_sha256)
            adapter.assert_spec(spec)
            write_json(output / "replay.json", adapter.receipt())
            consumer = ConsumerKind.EXPERT_REPLAY
        else:
            module, name = args.adapter.split(":")
            adapter_config_bytes = Path(args.adapter_config).read_bytes() if args.adapter_config else None
            _check_expected_digest(adapter_config_bytes, getattr(args, "adapter_config_sha256", None), "adapter config")
            config = json.loads(adapter_config_bytes) if adapter_config_bytes is not None else {}
            adapter = getattr(import_module(module), name)(config)
            consumer = ConsumerKind.DIAGNOSTIC if args.backend == "demo" else ConsumerKind.MODEL
        prepared_results = []
        try:
            context = BuildContext(sim, robot, assets, output)
            limits = EpisodeLimits(args.physics_ticks, args.wall_s,
                                   getattr(args, 'policy_kind', None), getattr(args, 'budget_version', None))
            run_episode_started = True
            result = run_episode(spec, task, context, adapter,
                                 consumer_kind=consumer,
                                 on_worker_exit_prepared=prepared_results.append if args.backend == "isaac" else None,
                                 limits=limits,
                                 **({'budget_source': args._development_budget}
                                    if getattr(args, '_development_budget', None) else {}))
        except BaseException as setup_error:
            if args.backend == 'isaac' and not run_episode_started:
                from mind2act_eval.runtime_diagnostics import checkpoint_prerun_failure
                checkpoint_prerun_failure(output, setup_error, phase='worker_setup', source_file=__file__, spec=spec)
            raise
        termination = getattr(result, "termination", None)
        _write_worker_stage(output, "run_episode_returned",
                            infra_valid=getattr(result, "infra_valid", None),
                            termination=getattr(termination, "value", termination),
                            ticks=getattr(result, "ticks", None))
        _WORKER_FAILURE_EXIT_READY = (args.backend == "isaac" and result.infra_valid is False
                                      and result.score is None and len(prepared_results) == 1
                                      and prepared_results[0] is result)
        return 0 if result.infra_valid else 1
    except BaseException as worker_error:
        if args.backend == 'isaac' and not run_episode_started:
            from mind2act_eval.runtime_diagnostics import checkpoint_prerun_failure
            checkpoint_prerun_failure(output, worker_error, phase='worker_setup', source_file=__file__, spec=spec)
        cleanup_errors = []
        for resource in (adapter, sim):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(f"{type(resource).__name__}.close: {type(cleanup_error).__name__}: {cleanup_error}")
        if cleanup_errors:
            print("worker cleanup errors: " + "; ".join(cleanup_errors), file=sys.stderr)
        if isinstance(worker_error, SystemExit) and worker_error.code in (None, 0):
            raise RuntimeError("worker exited with SystemExit(0) before completing its result") from worker_error
        raise


def _job(args):
    from mind2act_eval.batch import EpisodeJob
    from mind2act_eval.runtime_release import guard_input_paths, validate_release_inputs, resolve_case_arguments
    resolve_case_arguments(args)
    guard_input_paths(args)
    spec_bytes = Path(args.spec).read_bytes()
    spec = InstanceSpec.from_dict(json.loads(spec_bytes))
    from mind2act_eval.sim.video import video_interval
    video_interval(spec.clock.physics_hz, getattr(args, "video_hz", 6))
    validate_release_inputs(spec, args)
    EpisodeLimits(args.physics_ticks, args.wall_s, getattr(args, 'policy_kind', None),
                  getattr(args, 'budget_version', None))
    if getattr(args, 'policy_kind', None) and (args.h5 or getattr(args, 'consumer', None)):
        raise ValueError('policy kind is reserved for scored model consumers')
    if any(getattr(args, key, None) is not None for key in ('consumer', 'probe_config', 'probe_config_sha256')):
        raise ValueError("diagnostic consumers are not part of the public runtime")
    if getattr(args, 'adapter_config_sha256', None) is not None:
        raw = Path(args.adapter_config).read_bytes() if args.adapter_config else None
        _check_expected_digest(raw, args.adapter_config_sha256, 'adapter config')
    if args.backend == "isaac" and not args.h5 and args.adapter == "mind2act_eval.adapters.demo:create":
        raise ValueError("Isaac model runs require an explicit model adapter")
    if args.backend == "isaac" and args.gpu is None:
        raise ValueError("Isaac episodes require an explicit idle GPU index")
    if spec.variation.mode == "random":
        from mind2act_eval.variations import resolve
        resolve(spec, _episode_assets(spec, args))
    elif getattr(args, "variation_manifest", None):
        raise ValueError("clean cannot request a variation manifest")
    if args.h5:
        from mind2act_eval.replay import H5Replay, ReplayMetadata
        metadata_bytes = Path(args.replay_metadata).read_bytes() if args.replay_metadata else None
        _check_expected_digest(metadata_bytes, args.replay_metadata_sha256, "replay metadata")
        metadata = ReplayMetadata(**json.loads(metadata_bytes)) if metadata_bytes is not None else None
        replay = H5Replay(Path(args.h5), metadata, expected_sha256=args.h5_sha256)
        try:
            replay.assert_spec(spec)
            if spec.clock != replay.metadata.clock:
                raise ValueError("set instance clock to the recorded replay clock explicitly")
        finally:
            replay.close()
    consumer = ConsumerKind.EXPERT_REPLAY if args.h5 else (ConsumerKind.DIAGNOSTIC if args.backend == "demo" else ConsumerKind.MODEL)
    return EpisodeJob(spec, _worker_argv(args), consumer, args.wall_s, args.retries, args.gpu,
                      policy_kind=getattr(args, 'policy_kind', None),
                      budget_version=getattr(args, 'budget_version', None),
                      physics_ticks=args.physics_ticks if getattr(args, 'policy_kind', None) else None)


def _worker_argv(args):
    """One worker command for admitted model runs and trusted replay."""
    argv = [sys.executable, "-m", "mind2act_eval.internal", "_worker", "--spec", str(Path(args.spec).resolve()),
            "--asset-root", str(Path(args.asset_root).resolve()), "--backend", args.backend,
            "--adapter", args.adapter, "--wall-s", str(args.wall_s), "--physics-ticks", str(args.physics_ticks),
            "--retries", str(args.retries)]
    for field in ('policy_kind', 'budget_version'):
        value = getattr(args, field, None)
        if value is not None:
            argv.extend(['--' + field.replace('_', '-'), value])
    argv.extend(["--video-hz", str(getattr(args, "video_hz", 6))])
    if getattr(args, "record_wrist_cameras", False):
        argv.append("--record-wrist-cameras")
    if args.gpu is not None:
        argv.extend(["--gpu", str(args.gpu)])
    if getattr(args, "variation_manifest", None):
        argv.extend(["--variation-manifest", args.variation_manifest])
    for flag in ("h5", "replay_metadata", "adapter_config"):
        value = getattr(args, flag)
        if value:
            argv.extend(["--" + flag.replace("_", "-"), str(Path(value).resolve())])
    if args.h5_sha256:
        argv.extend(["--h5-sha256", args.h5_sha256])
    if args.replay_metadata_sha256:
        argv.extend(["--replay-metadata-sha256", args.replay_metadata_sha256])
    adapter_config_sha = getattr(args, "adapter_config_sha256", None)
    if adapter_config_sha:
        argv.extend(["--adapter-config-sha256", adapter_config_sha])
    return tuple(argv)


def _job_defaults():
    return {
        "spec": None, "spec_sha256": None, "case_id": None, "asset_root": None, "backend": None,
        "adapter": "mind2act_eval.adapters.demo:create", "adapter_config": None,
        "adapter_config_sha256": None, "h5": None, "replay_metadata": None,
        "h5_sha256": None, "replay_metadata_sha256": None, "physics_ticks": None,
        "wall_s": None, "video_hz": 6, "retries": 0, "gpu": None,
        "variation_manifest": None,
        "policy_kind": None, "budget_version": None,
    }




def main(argv=None):
    global _WORKER_LIFECYCLE_OUTPUT, _WORKER_EXPECTED_CONSUMER, _WORKER_FAILURE_EXIT_READY
    _WORKER_LIFECYCLE_OUTPUT = None
    _WORKER_EXPECTED_CONSUMER = None
    _WORKER_FAILURE_EXIT_READY = False
    parser = argparse.ArgumentParser(prog="mind2act-eval-internal")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("spec", nargs="+")
    for name in ("run", "replay", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--spec")
        command.add_argument("--case-id", help="Exact test or replay case ID in the selected release")
        command.add_argument("--asset-root", default=".")
        command.add_argument("--variation-manifest", help="Pinned variation asset manifest, relative to asset root")
        command.add_argument("--output-root", default="results")
        command.add_argument("--backend", choices=("isaac", "demo"), default="isaac")
        command.add_argument("--adapter", default="mind2act_eval.adapters.demo:create")
        command.add_argument("--adapter-config")
        command.add_argument("--adapter-config-sha256")
        add_model_mode_arguments(command)
        command.add_argument("--gpu", type=int)
        command.add_argument("--h5", required=name == "replay")
        command.add_argument("--replay-metadata")
        command.add_argument("--h5-sha256")
        command.add_argument("--replay-metadata-sha256")
        command.add_argument("--physics-ticks", type=int)
        command.add_argument("--wall-s", type=float)
        command.add_argument("--video-hz", type=int, default=6, help="Recording Hz; independent of the instance RGB/control clock")
        command.add_argument("--record-wrist-cameras", action="store_true",
                             help="Record both X5 wrist cameras without changing policy observations")
        command.add_argument("--retries", type=int)
        if name == "_worker":
            command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        _WORKER_LIFECYCLE_OUTPUT = Path(args.output)
    try:
        if args.command == "validate":
            for path in args.spec:
                spec = read_spec(path)
                print(f"valid {spec.task_id}/{spec.instance_id}/{spec.difficulty}")
            return 0
        # Diagnostic callers can run a short private episode without a model
        # protocol. The separate public parser requires an explicit mode.
        if args.model_mode or args.policy_kind or args.budget_version:
            configure_model_mode(args)
        if args.command == "_worker":
            exit_code = _worker(args)
            _write_worker_stage(args.output, "worker_returned", exit_code=exit_code)
            return exit_code
        from mind2act_eval.batch import run_attempt
        if args.backend == "isaac" and not args.h5 and args.adapter == "mind2act_eval.adapters.demo:create":
            raise ValueError("Isaac model runs require an explicit model adapter")
        job = _job(args)
        output = Path(args.output_root).resolve()
        run_id = uuid.uuid4().hex
        retryable = {Termination.KILLED, Termination.RESOURCE_LIMIT, Termination.WALL_TIMEOUT}
        for attempt in range(job.retries + 1):
            directory = output / f'episode-{run_id}-attempt-{attempt + 1}'
            result = run_attempt(job, directory)
            print(directory)
            receipt = json.loads((directory / 'batch-exit.json').read_text())
            if not receipt.get('worker_exit_verified'):
                raise RuntimeError('worker exit unknown; retain output and do not retry')
            if result.infra_valid or result.termination not in retryable:
                break
        return 0 if result.infra_valid else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _run_cli_entrypoint(argv=None):
    exit_code = main(argv)
    if _WORKER_LIFECYCLE_OUTPUT is not None:
        _write_worker_stage(_WORKER_LIFECYCLE_OUTPUT, "cli_returned", exit_code=exit_code)
        if exit_code == 0:
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                from mind2act_eval.sealing import seal_worker_outputs
                if _WORKER_EXPECTED_CONSUMER is None:
                    seal_worker_outputs(_WORKER_LIFECYCLE_OUTPUT)
                else:
                    seal_worker_outputs(_WORKER_LIFECYCLE_OUTPUT, expected_consumer=_WORKER_EXPECTED_CONSUMER)
            except Exception as exc:
                print(f"worker output seal failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            os._exit(0)
        elif _WORKER_FAILURE_EXIT_READY:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                from mind2act_eval.sealing import flush_failed_worker_outputs
                flush_failed_worker_outputs(_WORKER_LIFECYCLE_OUTPUT, exit_code=exit_code)
            except Exception as exc:
                print(f"failed worker output flush failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return exit_code
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_run_cli_entrypoint())
