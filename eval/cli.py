"""Public CLI: validate/run/replay/batch. Each episode starts a fresh interpreter."""
from __future__ import annotations

import argparse
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import stat
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
    spec = InstanceSpec.from_dict(json.loads(Path(path).read_text()))
    from mind2act_eval.public_layouts import validate
    validate(spec)
    return spec




def _gpu_selection_kwargs(args):
    path = getattr(args, "gpu_allocation", None)
    expected = getattr(args, "gpu_allocation_sha256", None)
    if path is None and expected is None:
        return {}  # Preserve the original call signature and L20 defaults.
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
    _gpu_selection_kwargs(args)  # Explicit mappings are admitted before child launch.
    expected_spec_sha = getattr(args, "spec_sha256", None)
    if expected_spec_sha and hashlib.sha256(spec_bytes).hexdigest() != expected_spec_sha:
        raise ValueError("batch instance spec digest mismatch")
    adapter_config_sha = getattr(args, "adapter_config_sha256", None)
    adapter_config_bytes = Path(args.adapter_config).read_bytes() if args.adapter_config else None
    if adapter_config_sha:
        if (adapter_config_bytes is None
                or hashlib.sha256(adapter_config_bytes).hexdigest() != adapter_config_sha):
            raise ValueError("batch adapter config digest mismatch")
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
    consumer = ConsumerKind.EXPERT_REPLAY if args.h5 else (ConsumerKind.DIAGNOSTIC if args.backend == "demo" else ConsumerKind.MODEL)
    return EpisodeJob(spec, _worker_argv(args), consumer, args.wall_s, args.retries, args.gpu,
                      policy_kind=getattr(args, 'policy_kind', None),
                      budget_version=getattr(args, 'budget_version', None),
                      physics_ticks=args.physics_ticks if getattr(args, 'policy_kind', None) else None)


def _worker_argv(args):
    """One worker command for admitted model runs and trusted replay."""
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
        "gpu_allocation": None, "gpu_allocation_sha256": None,
        "variation_manifest": None,
        "policy_kind": None, "budget_version": None,
    }


def _queue_job(entry, gpu):
    values = _job_defaults()
    values.update(entry["options"])
    values["gpu"] = gpu
    # The queue owns its single infrastructure retry outside run_attempt.
    values["retries"] = 0
    job = _job(argparse.Namespace(**values))
    actual_key = {"task_id": job.spec.task_id, "instance_id": job.spec.instance_id,
                  "difficulty": job.spec.difficulty}
    if job.spec.variation.mode == "random":
        from mind2act_eval.variations import digest
        actual_key["variation_sha256"] = digest(job.spec.variation)
    if actual_key != entry["episode_key"]:
        raise ValueError("queue episode_key does not match its pinned spec")
    return job


@path_check_session()
def _batch_jobs(entries, defaults, *, distributed=False):
    """The same CPU input admission for the CLI and quarantine repair."""
    jobs = []
    for entry in entries:
        values = defaults.copy()
        allowed = {"spec", "spec_sha256", "case_id", "asset_root", "backend", "adapter", "adapter_config",
                   "adapter_config_sha256", "h5", "replay_metadata", "physics_ticks", "wall_s", "video_hz",
                   "retries", "gpu", "h5_sha256", "replay_metadata_sha256",
                   "episode_key"}
        allowed.update({"gpu_allocation", "gpu_allocation_sha256"})
        allowed.add("variation_manifest")
        allowed.update({'policy_kind', 'budget_version'})
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError("invalid batch manifest options")
        if distributed:
            required = {"spec", "spec_sha256", "episode_key", "asset_root", "backend", "adapter",
                "physics_ticks", "wall_s", "retries", "gpu"}
            if not required <= set(entry):
                raise ValueError("distributed campaign job is missing pinned execution options")
            if entry.get("h5") and not entry.get("h5_sha256"):
                raise ValueError("distributed expert replay jobs require a pinned h5_sha256")
            if entry.get("adapter_config") and not entry.get("adapter_config_sha256"):
                raise ValueError("distributed model jobs require a pinned adapter_config_sha256")
            if entry.get("replay_metadata") and not entry.get("replay_metadata_sha256"):
                raise ValueError("distributed replay metadata requires a pinned digest")
        values.update(entry)
        jobs.append(_job(argparse.Namespace(**values)))
    if distributed and any(type(job.gpu) is not int or job.gpu < 0 for job in jobs):
        raise ValueError("distributed campaign jobs must assign each episode a nonnegative GPU")
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
    queue = commands.add_parser("queue")
    queue.add_argument("--manifest", required=True, help="Pinned shared dynamic queue JSON")
    queue.add_argument("--sha256", required=True, help="Expected SHA256 of the queue manifest")
    queue.add_argument("--publish-job", help="Prepared new generation JSON; updates queue state and exits")
    queue.add_argument("--dispatch-control", help="PRO rolling intake CAS request JSON; requires frozen controller identity")
    queue.add_argument("--register-catalog", action="store_true", help="Append this immutable PRO rolling catalog without changing existing cases or active profiles")
    queue.add_argument("--catalog-revision", type=int, help="Expected central catalog revision for registration")
    queue.add_argument("--evidence", help="Required evidence for publishing a new generation")
    queue.add_argument("--review-job")
    queue.add_argument("--decision", choices=("pass", "fail"))
    queue.add_argument("--reviewer")
    queue.add_argument("--correct-review", help="JSON audit correction bound to an exact generation, attempt and prior review")
    queue.add_argument("--quarantine-job", help="JSON request to fence an exact queued-zero generation as blocked")
    queue.add_argument("--quarantine-completion", help="JSON external completion audit; retains the blocked quarantine")
    queue.add_argument("--quarantine-gpu-amendment", help="Pinned never-launched external campaign GPU-omission repair JSON")
    queue.add_argument("--initialize", action="store_true", help="Explicit first creation of a new queue only; never recovery")
    queue.add_argument("--adopt-legacy", action="store_true", help="Guard exact reviewed legacy state during full maintenance; does not restore state")
    queue.add_argument("--expected-state-sha256")
    queue.add_argument("--restore-checkpoint", help="Explicit exact checkpoint recovery plan; stopped owners and all maintenance gates required")
    queue.add_argument("--recovery-plan-sha256")
    queue.add_argument("--check-restore", action="store_true", help="Validate recovery prerequisites without replacing state")
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
        command.add_argument('--policy-kind', choices=('e2e', 'agent', 'coding_agent'),
                             help='agent: multimodal agents acting through tools; coding_agent is a legacy alias')
        command.add_argument('--budget-version')
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
        if name == "_worker":
            command.add_argument("--output", required=True)
        if name == "batch":
            command.add_argument("--max-parallel", type=int)
            command.add_argument("--resource-config", help="Explicit single-node RSS/parallel config JSON; do not combine with --max-parallel overrides")
            command.add_argument("--manifest", help="JSON list of explicit CLI option dictionaries")
            command.add_argument("--campaign-manifest", help="Immutable multi-node static partition manifest")
            command.add_argument("--campaign-sha256", help="Expected SHA256 of --campaign-manifest")
    args = parser.parse_args(argv)
    if args.command == "_worker":
        _WORKER_LIFECYCLE_OUTPUT = Path(args.output)
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
        if args.command == "queue":
            from mind2act_eval.shared_queue import load_queue, run_queue_driver
            if any(value is not None and not value for value in (args.quarantine_job, args.quarantine_completion, args.quarantine_gpu_amendment)):
                raise ValueError('quarantine request path must be nonempty')
            queue_spec = load_queue(args.manifest, expected_sha256=args.sha256)
            if sum(bool(value) for value in (args.register_catalog, args.dispatch_control, args.publish_job, args.review_job, args.correct_review, args.initialize, args.adopt_legacy, args.restore_checkpoint, args.quarantine_job, args.quarantine_completion, args.quarantine_gpu_amendment)) > 1:
                raise ValueError("initialization, publish, review and correction are separate queue transactions")
            if args.catalog_revision is not None and not args.register_catalog:
                raise ValueError('--catalog-revision requires --register-catalog')
            if args.register_catalog:
                from mind2act_eval.shared_queue import QueueStore
                from mind2act_eval.pro_rolling_queue import append_catalog
                print(json.dumps(append_catalog(QueueStore(queue_spec), expected_revision=args.catalog_revision,
                                               evidence=args.evidence), sort_keys=True))
                return 0
            if args.dispatch_control:
                from mind2act_eval.shared_queue import QueueStore
                from mind2act_eval.pro_rolling_queue import set_control
                request = json.loads(Path(args.dispatch_control).read_text())
                print(json.dumps(set_control(QueueStore(queue_spec), **request), sort_keys=True))
                return 0
            if (args.recovery_plan_sha256 or args.check_restore) and not args.restore_checkpoint:
                raise ValueError("recovery options require --restore-checkpoint")
            if args.restore_checkpoint:
                from mind2act_eval.queue_checkpoint_recovery import restore_checkpoint
                receipt = restore_checkpoint(queue_spec, args.restore_checkpoint,
                    expected_plan_sha256=args.recovery_plan_sha256, check_only=args.check_restore)
                print(json.dumps(receipt, sort_keys=True))
                return 0
            if args.expected_state_sha256 and not args.adopt_legacy:
                raise ValueError("--expected-state-sha256 requires --adopt-legacy")
            if args.adopt_legacy:
                from mind2act_eval.shared_queue import QueueStore
                QueueStore(queue_spec).adopt_legacy(expected_state_sha256=args.expected_state_sha256,
                                                    evidence=args.evidence)
                return 0
            if args.initialize:
                from mind2act_eval.shared_queue import QueueStore
                QueueStore(queue_spec).initialize(evidence=args.evidence)
                return 0
            if args.quarantine_job or args.quarantine_completion or args.quarantine_gpu_amendment:
                from mind2act_eval.shared_queue import QueueStore
                path = args.quarantine_job or args.quarantine_completion or args.quarantine_gpu_amendment
                fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
                with os.fdopen(fd) as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError('quarantine request must be a regular file')
                    request = json.load(stream)
                store = QueueStore(queue_spec)
                operation = (store.amend_quarantine_external_batch if args.quarantine_gpu_amendment else
                             store.quarantine_queued if args.quarantine_job else store.record_quarantine_completion)
                print(json.dumps(operation(**request), sort_keys=True))
                return 0
            if args.correct_review:
                from mind2act_eval.shared_queue import QueueStore
                request = json.loads(Path(args.correct_review).read_text())
                QueueStore(queue_spec).correct_review(**request)
                return 0
            if args.review_job:
                from mind2act_eval.shared_queue import QueueStore
                QueueStore(queue_spec).review(args.review_job, decision=args.decision,
                                              reviewer=args.reviewer, evidence=args.evidence)
                return 0
            if args.publish_job:
                from mind2act_eval.shared_queue import QueueStore
                QueueStore(queue_spec).publish_generation(json.loads(Path(args.publish_job).read_text()), evidence=args.evidence)
                return 0
            exit_code = run_queue_driver(queue_spec, job_factory=_queue_job)
            print(json.dumps({"queue_id": queue_spec.queue_id,
                              "hostname": queue_spec.node.hostname,
                              "state_file": str(queue_spec.state_root / f"queue-{queue_spec.queue_id}.json"),
                              "exit_code": exit_code}, sort_keys=True), flush=True)
            return exit_code
        from mind2act_eval.batch import run_batch
        if args.backend == "isaac" and not args.h5 and args.adapter == "mind2act_eval.adapters.demo:create" and args.command != "batch":
            raise ValueError("Isaac model runs require an explicit model adapter")
        distributed_partition = None
        resource_path = None
        resource_config = None
        if args.command == "batch":
            resource_path = getattr(args, "resource_config", None)
            if args.campaign_manifest:
                if args.manifest or resource_path or args.max_parallel is not None or args.campaign_sha256 is None:
                    raise ValueError("distributed campaign requires its SHA and cannot be combined with manifest/resource/parallel overrides")
                from mind2act_eval.distributed_campaign import load_partition
                distributed_partition = load_partition(
                    args.campaign_manifest, expected_sha256=args.campaign_sha256)
                entries = json.loads(distributed_partition.manifest_bytes)
                resource_config = distributed_partition.resource_config
            else:
                if not args.manifest or args.campaign_sha256:
                    raise ValueError("batch requires --manifest, or --campaign-manifest with --campaign-sha256")
                manifest_path = Path(args.manifest)
                entries = json.loads(manifest_path.read_text())
            jobs = _batch_jobs(entries, vars(args) if distributed_partition is None else _job_defaults(),
                               distributed=distributed_partition is not None)
        else:
            jobs = [_job(args)]
        if not jobs:
            raise ValueError("batch manifest must contain at least one episode")
        require_shared = any(job.spec.task_id != "demo" or job.gpu is not None for job in jobs)
        if distributed_partition is None:
            from mind2act_eval.batch_resources import BatchResourceConfig
            resource_config = BatchResourceConfig.read(resource_path) if resource_path else None
        max_parallel_override = getattr(args, "max_parallel", None)
        max_parallel = max_parallel_override if max_parallel_override is not None else 1
        if distributed_partition is not None:
            require_shared = True
            if any(type(job.gpu) is not int or job.gpu < 0 for job in jobs):
                raise ValueError("distributed campaign jobs must assign each episode a nonnegative GPU")
        campaign, summary = run_batch(jobs, Path(args.output_root), require_shared=require_shared,
                                      max_parallel=max_parallel,
                                      start_interval_s=60. if require_shared else 0., resource_config=resource_config,
                                      distributed_partition=distributed_partition)
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
