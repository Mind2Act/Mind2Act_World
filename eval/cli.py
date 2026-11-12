"""Public CLI: validate/run/replay/batch. Each episode starts a fresh interpreter."""
from __future__ import annotations

import argparse
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import sys

from mind2act_eval.contracts import BuildContext, ConsumerKind, InstanceSpec
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
    return InstanceSpec.from_dict(json.loads(Path(path).read_text()))


def _probe_inputs(args, spec):
    from mind2act_eval.scene_probe import ProbeInputs
    fields = (getattr(args, "probe_config", None), getattr(args, "probe_config_sha256", None))
    if (args.backend != "isaac" or args.h5 or args.replay_metadata or args.adapter_config
            or args.adapter != "mind2act_eval.adapters.demo:create" or args.physics_ticks != 600
            or args.retries != 0 or not all(fields) or type(args.gpu) is not int or args.gpu < 0
            or args.h5_sha256 or args.replay_metadata_sha256 or args.adapter_config_sha256):
        raise ValueError("scene_probe requires Isaac, explicit GPU, 600 ticks, no retries/H5/model adapter, pinned probe config")
    return ProbeInputs.read(*fields, spec)


def _gpu_selection_kwargs(args):
    path = getattr(args, "gpu_allocation", None)
    expected = getattr(args, "gpu_allocation_sha256", None)
    if path is None and expected is None:
        return {}  # Preserve the original call signature and single-machine defaults.
    if not path or not expected:
        raise ValueError("--gpu-allocation and --gpu-allocation-sha256 must be supplied together")
    if args.backend != "isaac":
        raise ValueError("GPU allocation mapping requires the Isaac backend")
    from mind2act_eval.gpu_allocation import load_gpu_selection
    selection = load_gpu_selection(Path(path), expected, physical_gpu=args.gpu)
    return {"gpu_selection": selection}


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


def _worker(args):
    global _WORKER_FAILURE_EXIT_READY
    _WORKER_FAILURE_EXIT_READY = False
    from mind2act_eval.runtime_release import guard_input_paths, validate_release_inputs, resolve_case_arguments
    resolve_case_arguments(args)
    guard_input_paths(args)
    validate_release_inputs(read_spec(args.spec), args, require_code_identity=args.backend == "isaac")
    if getattr(args, 'policy_kind', None) is not None or getattr(args, 'budget_version', None) is not None:
        EpisodeLimits(args.physics_ticks, args.wall_s, getattr(args, 'policy_kind', None),
                      getattr(args, 'budget_version', None))
    if getattr(args, 'policy_kind', None) and (args.h5 or getattr(args, 'consumer', None)):
        raise ValueError('policy kind is reserved for scored model consumers')
    gpu_kwargs = _gpu_selection_kwargs(args)
    if getattr(args, "consumer", None) not in (None, "scene_probe"):
        raise ValueError("unsupported explicit consumer")
    if getattr(args, "consumer", None) == "scene_probe":
        _probe_inputs(args, read_spec(args.spec))
        from mind2act_eval.worker import run_scene_probe
        result = run_scene_probe(args, **gpu_kwargs)
        _write_worker_stage(args.output, "scene_probe_returned", infra_valid=result.infra_valid,
                            ticks=result.ticks)
        return 0 if result.infra_valid else 1
    if getattr(args, "probe_config", None) or getattr(args, "probe_config_sha256", None):
        raise ValueError("probe configuration requires explicit --consumer scene_probe")
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
            if 'material_mdl_search_asset_ids' in spec.config:
                from mind2act_eval.sim.material_closure import configure_mdl_search
                material_receipt = configure_mdl_search(assets, spec.config['material_mdl_search_asset_ids'])
            if spec.variation.mode == "random":
                from mind2act_eval.variations import configure_material_search
                variation_materials = configure_material_search(spec, assets,
                    spec.config.get('material_mdl_search_asset_ids', ()))
                if variation_materials is not None:
                    material_receipt = variation_materials
            sim = IsaacSimulator(spec.clock, output, physical_gpu=args.gpu, cameras=spec.config.get("cameras"),
                                 ffmpeg=spec.config.get("ffmpeg", "ffmpeg"), video_hz=getattr(args, "video_hz", 6), **gpu_kwargs)
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
            if (metadata_bytes is not None and args.replay_metadata_sha256
                    and hashlib.sha256(metadata_bytes).hexdigest() != args.replay_metadata_sha256):
                raise ValueError("replay metadata digest mismatch")
            metadata = ReplayMetadata(**json.loads(metadata_bytes)) if metadata_bytes is not None else None
            adapter = H5Replay(Path(args.h5), metadata, expected_sha256=args.h5_sha256)
            adapter.assert_spec(spec)
            write_json(output / "replay.json", adapter.receipt())
            consumer = ConsumerKind.EXPERT_REPLAY
        else:
            module, name = args.adapter.split(":")
            adapter_config_sha = getattr(args, "adapter_config_sha256", None)
            adapter_config_bytes = Path(args.adapter_config).read_bytes() if args.adapter_config else None
            if adapter_config_sha:
                if adapter_config_bytes is None:
                    raise ValueError("adapter config digest supplied without adapter config")
                actual = hashlib.sha256(adapter_config_bytes).hexdigest()
                if actual != adapter_config_sha:
                    raise ValueError("adapter config digest mismatch in worker")
            config = json.loads(adapter_config_bytes) if adapter_config_bytes is not None else {}
            adapter = getattr(import_module(module), name)(config)
            consumer = ConsumerKind.DIAGNOSTIC if args.backend == "demo" else ConsumerKind.MODEL
        prepared_results = []
        setup_error_active = False
        try:
            context = BuildContext(sim, robot, assets, output)
            limits = EpisodeLimits(args.physics_ticks, args.wall_s,
                                   getattr(args, 'policy_kind', None), getattr(args, 'budget_version', None))
            run_episode_started = True
            result = run_episode(spec, task, context, adapter,
                                 consumer_kind=consumer,
                                 on_worker_exit_prepared=prepared_results.append if args.backend == "isaac" else None,
                                 limits=limits)
        except BaseException as setup_error:
            if args.backend == 'isaac' and not run_episode_started:
                setup_error_active = True
                from mind2act_eval.runtime_diagnostics import checkpoint_prerun_failure
                checkpoint_prerun_failure(output, setup_error, phase='worker_setup', source_file=__file__, spec=spec)
            raise
        finally:
            try:
                from mind2act_eval.sim.source_assets import write_source_projection_receipt
                write_source_projection_receipt(assets, output)
            except BaseException as receipt_error:
                if not setup_error_active:
                    raise
                # Keep the in-flight setup exception primary. Receipt failures
                # after runner entry retain their existing propagation behavior.
                try:
                    print(f'secondary source receipt error: {type(receipt_error).__name__}: {receipt_error}', file=sys.stderr)
                    sys.stderr.flush()
                except BaseException:
                    pass
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
    from mind2act_eval.batch import EpisodeJob, NativeExitDebugConfig
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
    if getattr(args, "consumer", None) not in (None, "scene_probe"):
        raise ValueError("unsupported explicit consumer")
    _gpu_selection_kwargs(args)  # Explicit mappings are admitted before child launch.
    probe = getattr(args, "consumer", None) == "scene_probe"
    probe_fields = (getattr(args, "probe_config", None), getattr(args, "probe_config_sha256", None))
    if probe:
        _probe_inputs(args, spec)
    elif any(probe_fields):
        raise ValueError("probe configuration requires explicit --consumer scene_probe")
    expected_spec_sha = getattr(args, "spec_sha256", None)
    if expected_spec_sha and hashlib.sha256(spec_bytes).hexdigest() != expected_spec_sha:
        raise ValueError("batch instance spec digest mismatch")
    adapter_config_sha = getattr(args, "adapter_config_sha256", None)
    adapter_config_bytes = Path(args.adapter_config).read_bytes() if args.adapter_config else None
    if adapter_config_sha:
        if (adapter_config_bytes is None
                or hashlib.sha256(adapter_config_bytes).hexdigest() != adapter_config_sha):
            raise ValueError("batch adapter config digest mismatch")
    if not probe and args.backend == "isaac" and not args.h5 and args.adapter == "mind2act_eval.adapters.demo:create":
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
        metadata_sha = hashlib.sha256(metadata_bytes).hexdigest() if metadata_bytes is not None else None
        if args.replay_metadata_sha256 and metadata_sha != args.replay_metadata_sha256:
            raise ValueError("replay metadata digest mismatch")
        metadata = ReplayMetadata(**json.loads(metadata_bytes)) if metadata_bytes is not None else None
        replay = H5Replay(Path(args.h5), metadata, expected_sha256=args.h5_sha256)
        try:
            args.h5_sha256 = replay.h5_sha256
            if args.replay_metadata:
                args.replay_metadata_sha256 = metadata_sha
            replay.assert_spec(spec)
            if spec.source_id != replay.metadata.source_id:
                raise ValueError("replay source differs from instance")
            if spec.clock != replay.metadata.clock:
                raise ValueError("set instance clock to the recorded replay clock explicitly")
        finally:
            replay.close()
    consumer = ConsumerKind.SCENE_PROBE if probe else (ConsumerKind.EXPERT_REPLAY if args.h5 else (ConsumerKind.DIAGNOSTIC if args.backend == "demo" else ConsumerKind.MODEL))
    debug_path = getattr(args, "native_exit_debug_config", None)
    native_exit_debug = NativeExitDebugConfig.read(debug_path) if debug_path else None
    return EpisodeJob(spec, _worker_argv(args), consumer, args.wall_s, args.retries, args.gpu,
                      native_exit_debug, policy_kind=getattr(args, 'policy_kind', None),
                      budget_version=getattr(args, 'budget_version', None),
                      physics_ticks=args.physics_ticks if getattr(args, 'policy_kind', None) else None)


def _worker_argv(args):
    """One standard worker command, including the trusted scene-probe path."""
    probe = getattr(args, "consumer", None) == "scene_probe"
    probe_fields = (getattr(args, "probe_config", None), getattr(args, "probe_config_sha256", None))
    argv = [sys.executable, "-m", "mind2act_eval.cli", "_worker", "--spec", str(Path(args.spec).resolve()),
            "--asset-root", str(Path(args.asset_root).resolve()), "--backend", args.backend,
            "--adapter", args.adapter, "--wall-s", str(args.wall_s), "--physics-ticks", str(args.physics_ticks),
            "--retries", str(args.retries)]
    for field in ('policy_kind', 'budget_version'):
        value = getattr(args, field, None)
        if value is not None:
            argv.extend(['--' + field.replace('_', '-'), value])
    argv.extend(["--video-hz", str(getattr(args, "video_hz", 6))])
    if args.gpu is not None:
        argv.extend(["--gpu", str(args.gpu)])
    if getattr(args, "variation_manifest", None):
        argv.extend(["--variation-manifest", args.variation_manifest])
    if getattr(args, "gpu_allocation", None) is not None:
        argv.extend(["--gpu-allocation", str(Path(args.gpu_allocation).resolve()),
                     "--gpu-allocation-sha256", args.gpu_allocation_sha256])
    if probe:
        argv.extend(["--consumer", "scene_probe", "--probe-config", str(Path(probe_fields[0]).resolve()),
                     "--probe-config-sha256", probe_fields[1]])
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
        "wall_s": None, "video_hz": 6, "retries": 0, "gpu": None, "native_exit_debug_config": None,
        "consumer": None, "probe_config": None, "probe_config_sha256": None,
        "gpu_allocation": None, "gpu_allocation_sha256": None,
        "variation_manifest": None,
        "policy_kind": None, "budget_version": None,
    }


@path_check_session()
def _batch_jobs(entries, defaults):
    """Admit explicit episode inputs before launching workers."""
    jobs = []
    for entry in entries:
        values = defaults.copy()
        allowed = {"spec", "spec_sha256", "case_id", "asset_root", "backend", "adapter", "adapter_config",
                   "adapter_config_sha256", "h5", "replay_metadata", "physics_ticks", "wall_s", "video_hz",
                   "retries", "gpu", "h5_sha256", "replay_metadata_sha256", "native_exit_debug_config",
                   "episode_key"}
        allowed.update({"consumer", "probe_config", "probe_config_sha256"})
        allowed.update({"gpu_allocation", "gpu_allocation_sha256"})
        allowed.add("variation_manifest")
        allowed.update({'policy_kind', 'budget_version'})
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("invalid batch manifest options")
        values.update(entry)
        jobs.append(_job(argparse.Namespace(**values)))
    return jobs


def main(argv=None):
    global _WORKER_LIFECYCLE_OUTPUT, _WORKER_EXPECTED_CONSUMER, _WORKER_FAILURE_EXIT_READY
    _WORKER_LIFECYCLE_OUTPUT = None
    _WORKER_EXPECTED_CONSUMER = None
    _WORKER_FAILURE_EXIT_READY = False
    parser = argparse.ArgumentParser(prog="mind2act-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("spec", nargs="+")
    for name in ("run", "replay", "_worker", "batch"):
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
        command.add_argument("--consumer", choices=("scene_probe",), help="Explicit unscored native scene diagnostic")
        command.add_argument('--policy-kind', choices=('e2e', 'coding_agent'))
        command.add_argument('--budget-version')
        command.add_argument("--probe-config")
        command.add_argument("--probe-config-sha256")
        command.add_argument("--gpu", type=int)
        command.add_argument("--gpu-allocation")
        command.add_argument("--gpu-allocation-sha256")
        command.add_argument("--h5", required=name == "replay")
        command.add_argument("--replay-metadata")
        command.add_argument("--h5-sha256")
        command.add_argument("--replay-metadata-sha256")
        command.add_argument("--physics-ticks", type=int, default=12000)
        command.add_argument("--wall-s", type=float, default=3600)
        command.add_argument("--video-hz", type=int, default=6, help="Recording Hz; independent of the instance RGB/control clock")
        command.add_argument("--retries", type=int, default=1)
        if name in ("run", "replay", "batch"):
            command.add_argument("--native-exit-debug-config")
        if name == "_worker":
            command.add_argument("--output", required=True)
        if name == "batch":
            command.add_argument("--max-parallel", type=int)
            command.add_argument("--resource-config", help="Explicit single-node RSS/parallel config JSON; do not combine with --max-parallel overrides")
            command.add_argument("--manifest", help="JSON list of explicit CLI option dictionaries")
            command.add_argument("--gpus", help="Comma-separated physical GPUs; one episode per GPU")
    args = parser.parse_args(argv)
    if args.command == "_worker":
        _WORKER_LIFECYCLE_OUTPUT = Path(args.output)
        if args.consumer == "scene_probe":
            _WORKER_EXPECTED_CONSUMER = ConsumerKind.SCENE_PROBE
    try:
        if args.command == "validate":
            for path in args.spec:
                spec = read_spec(path)
                print(f"valid {spec.task_id}/{spec.instance_id}/{spec.difficulty}")
            return 0
        if args.command == "_worker":
            exit_code = _worker(args)
            _write_worker_stage(args.output, "worker_returned", exit_code=exit_code)
            return exit_code
        from mind2act_eval.batch import run_batch
        if args.backend == "isaac" and not args.h5 and args.adapter == "mind2act_eval.adapters.demo:create" and args.command != "batch" and args.consumer != "scene_probe":
            raise ValueError("Isaac model runs require an explicit model adapter")
        from mind2act_eval.batch_resources import BatchResourceConfig
        resource_path = getattr(args, "resource_config", None)
        resource_config = BatchResourceConfig.read(resource_path) if resource_path else None
        if args.command == "batch":
            if not args.manifest:
                raise ValueError("batch requires --manifest")
            entries = json.loads(Path(args.manifest).read_text())
            if args.gpus:
                gpus = [int(value) for value in args.gpus.split(",")]
                if not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus):
                    raise ValueError("--gpus requires distinct nonnegative GPU indices")
                if args.max_parallel is not None or resource_path or args.gpu is not None:
                    raise ValueError("--gpus owns GPU assignment and parallelism")
                if any(entry.get("gpu") is not None for entry in entries):
                    raise ValueError("--gpus cannot override manifest GPU assignments")
                # Keep each task on one GPU; each GPU executes its jobs serially.
                assignments = {}
                for entry in entries:
                    spec_path = entry.get("spec")
                    if spec_path:
                        key = read_spec(spec_path).task_id
                    else:
                        from mind2act_eval.release_contract import load_release
                        release = load_release(entry.get("asset_root", args.asset_root))
                        key = release.resolve_case(entry["case_id"])["task_id"]
                    if key not in assignments:
                        assignments[key] = gpus[len(assignments) % len(gpus)]
                    entry["gpu"] = assignments[key]
                args.max_parallel = len(gpus)
            jobs = _batch_jobs(entries, vars(args))
        else:
            jobs = [_job(args)]
        if not jobs:
            raise ValueError("batch manifest must contain at least one episode")
        native = any(job.spec.task_id != "demo" for job in jobs)
        campaign, summary = run_batch(jobs, Path(args.output_root), require_shared=False,
                                      max_parallel=getattr(args, "max_parallel", None) or 1,
                                      start_interval_s=60. if native else 0., resource_config=resource_config)
        print(campaign)
        final = {}
        for attempt in summary:
            result = attempt["result"]
            final[(result.task_id, result.instance_id)] = result
        return 0 if all(result.infra_valid for result in final.values()) else 1
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
