"""Single episode execution. Production entrypoint runs this in a fresh child."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

from mind2act_eval.contracts import (
    Action, Health, Observation, PolicyStop, Result, ConsumerKind, Termination, to_json,
)
from .completion import completion_kind, completed
from mind2act_eval.budgets import AGENT_POLICY_KINDS


@dataclass(frozen=True)
class EpisodeLimits:
    physics_ticks: int = 12000
    wall_s: float = 3600
    policy_kind: str | None = None
    budget_version: str | None = None

    def __post_init__(self):
        from mind2act_eval.budgets import E2E_BUDGET_VERSION, AGENT_BUDGET_VERSION
        versions = {'e2e': E2E_BUDGET_VERSION, 'agent': AGENT_BUDGET_VERSION,
                    'coding_agent': AGENT_BUDGET_VERSION}
        if ((self.policy_kind is None) != (self.budget_version is None)
                or self.policy_kind is not None and versions.get(self.policy_kind) != self.budget_version):
            raise ValueError('explicit matching policy kind and budget version required')
        from math import isfinite
        if type(self.physics_ticks) is not int or self.physics_ticks <= 0 or not isfinite(self.wall_s) or self.wall_s <= 0:
            raise ValueError("episode budgets must be positive")


def write_json(path: Path, value):
    """Atomic per-attempt artifact; caller must allocate a unique attempt dir."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_timing_breakdown(path: Path, value):
    """Remove only this optional write's exclusively created temporary file."""
    payload = json.dumps(to_json(value), indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    stream = temporary.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(payload)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _first_frame(output_dir, rgb):
    """Persist already prepared pixels without rendering or advancing physics."""
    if "head" not in rgb:
        return None
    import numpy as np
    from PIL import Image
    path = Path(output_dir) / "first-frame" / "head.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb["head"])).save(path)
    return "first-frame/head.png"


def interpolate_action(current, following, fraction):
    """Sample a control interval at its physics-tick start, without extrapolation."""
    if following is None:
        return current
    def blend(a, b):
        return a + (b - a) * fraction
    return Action(tuple(blend(a, b) for a, b in zip(current.left_arm, following.left_arm)),
                  tuple(blend(a, b) for a, b in zip(current.right_arm, following.right_arm)),
                  blend(current.left_gripper, following.left_gripper),
                  blend(current.right_gripper, following.right_gripper))


def run_episode(spec, task, ctx, adapter, *, consumer_kind=ConsumerKind.MODEL,
                limits=EpisodeLimits(), on_worker_exit_prepared=None, budget_source=None):
    # Only an isolated CLI worker supplies this callback and guarantees a
    # process exit after the runner and its outer source receipt finish.
    if on_worker_exit_prepared is not None and not callable(on_worker_exit_prepared):
        raise TypeError("worker exit preparation callback must be callable")
    consumer_kind = ConsumerKind(consumer_kind)
    start = time.monotonic()
    episode_start = None
    timings = {"build_s": 0.0, "adapter_s": 0.0, "reset_s": 0.0, "inference_s": 0.0,
               "observation_s": 0.0, "action_s": 0.0, "score_s": 0.0}
    from mind2act_eval.timing import PhaseTimings
    phases = PhaseTimings()
    tick = 0
    scene = None
    evidence = None
    result = None
    model_tested = False
    policy_metadata = {}
    score_components = {}
    task_metadata = {}
    first_frame_artifact = None
    asset_failure = None
    robot_receipt = None
    variation_session = None
    replay_coverage = None
    replay_rows_returned = 0
    replay_rows_dispatched = 0
    replay_physics_ticks_executed = 0
    replay_eof = False
    replay_last_hold_ticks = 0
    last_target = None
    task_first_complete_tick = None
    completion_mode = completion_kind(spec.task_id)
    completion_receipt = dict(kind=completion_mode, checks=0, first_ready_tick=None,
        result_boundary='model_action_chunk_end', event_boundary='physics_tick',
        legacy_violation_done_used=False)
    replay_success_terminal = False
    expert_replay = consumer_kind == ConsumerKind.EXPERT_REPLAY
    ticks_per_action = spec.clock.ticks_per_action
    action_execution = {"schema": "mind2act_eval.action_execution.v1", "method": "linear_same_chunk",
                        "sampling": "interval_start_plus_subtick_over_ticks_per_action",
                        "ticks_per_action": ticks_per_action, "interpolated_ticks": 0, "held_ticks": 0}
    from mind2act_eval.budgets import DecisionBudget, CallBudgetExhausted
    def record_calls(receipt):
        write_json(ctx.output_dir / 'call-budget.json', dict(
            schema='mind2act_eval.call_budget.v1', task_id=spec.task_id,
            instance_id=spec.instance_id, budget_version=limits.budget_version, **receipt))
    calls = DecisionBudget(recorder=record_calls) if limits.policy_kind in AGENT_POLICY_KINDS else None
    budget_receipt = dict(policy_kind=limits.policy_kind, budget_version=limits.budget_version,
                          physics_ticks=limits.physics_ticks, wall_s=limits.wall_s, retries=0)
    if budget_source is not None:
        budget_receipt.update(budget_source)
    def check_task_completion():
        ready = completed(scene, spec.task_id)
        completion_receipt['checks'] += 1
        if ready and completion_receipt['first_ready_tick'] is None:
            completion_receipt['first_ready_tick'] = tick
        return ready
    try:
        bind_public = getattr(adapter, 'bind_public_task', None)
        if callable(bind_public):
            # Never forward the private spec: only declared public Goal inputs.
            parameters = {}
            if spec.task_id in ('PR1', 'PR2'):
                parameters['target'] = spec.config['target_ohm' if spec.task_id=='PR1' else 'target_volt']
            elif spec.task_id == 'CP07':
                from mind2act_eval.adapters.agent_input.cp07 import project_parameters
                parameters = project_parameters(spec, ctx.assets)
            bind_public(spec.task_id, spec.difficulty, parameters)
        if limits.policy_kind is not None:
            if consumer_kind != ConsumerKind.MODEL:
                raise ValueError('versioned policy budgets require a model consumer')
            write_json(ctx.output_dir / 'run-budget.json', budget_receipt)
        if calls is not None:
            record_calls(calls.receipt())
            bind = getattr(adapter, 'bind_decision_budget', None)
            if not callable(bind):
                raise ValueError('agent adapter must bind framework decision accounting')
            bind(calls)
        from mind2act_eval.sim.video import DEFAULT_VIDEO_HZ, video_interval
        video_hz = getattr(ctx.sim, "video_hz", DEFAULT_VIDEO_HZ)
        video_ticks = video_interval(spec.clock.physics_hz, video_hz)
        before = time.monotonic()
        if spec.variation.mode == "random":
            from dataclasses import replace
            from mind2act_eval.variations import VariationSession, resolve
            variation_session = VariationSession(resolve(spec, ctx.assets))
            ctx = replace(ctx, variation_session=variation_session)
        for service in (ctx.robot, ctx.sim):
            bind_task = getattr(service, "bind_task_instance", None)
            if bind_task is not None:
                bind_task(spec, ctx.output_dir)
        scene = task.build(spec, ctx)
        replay_success_terminal = getattr(scene, "terminate_replay_on_success", False)
        if type(replay_success_terminal) is not bool:
            raise TypeError("scene terminate_replay_on_success must be boolean")
        stop_on_success = completion_mode is not None and (
            consumer_kind == ConsumerKind.MODEL or expert_replay and replay_success_terminal)
        initialize = getattr(ctx.robot, "initialize", None)
        if initialize is not None:
            initialize(ctx.world)
        if variation_session is not None:
            write_json(ctx.output_dir / "variation-receipt.json",
                       variation_session.verify(native=True, stage=ctx.sim.stage))
        ready = getattr(ctx.sim, "ready", None)
        if ready is not None:
            ready()
        if variation_session is not None:
            write_json(ctx.output_dir / "variation-receipt.json",
                       variation_session.verify_visuals(stage=ctx.sim.stage))
        timings["build_s"] = time.monotonic() - before
        write_json(ctx.output_dir / "ready.json", {"ready": True, "pid": __import__("os").getpid()})
        # Build/reset and render-only first-frame preparation precede episode time.
        episode_start = time.monotonic()
        timings["pre_episode_s"] = episode_start - start
        before = time.monotonic()
        try:
            adapter.reset(spec.seed)
        finally:
            elapsed = time.monotonic() - before
            timings["reset_s"] += elapsed
            timings["adapter_s"] += elapsed
        validate_clock = getattr(adapter, "validate_clock", None)
        if validate_clock is not None:
            validate_clock(spec.clock)
        validate_limits = getattr(adapter, "validate_limits", None)
        if callable(validate_limits):
            validate_limits(limits)
        # Prompt physics and model input both start at tick zero.
        termination = Termination.PHYSICS_BUDGET
        while tick < limits.physics_ticks:
            if time.monotonic() - episode_start >= limits.wall_s:
                raise TimeoutError("episode wall budget exhausted")
            with phases.measure('scene.health'):
                health = scene.health()
            if not isinstance(health, Health):
                raise TypeError("scene.health() must return Health, never None/dict")
            if not health.ok:
                raise RuntimeError("native scene health: " + health.reason)
            # No initial success shortcut: the first model input is always tick0.
            # Legacy is_complete may mean failure; it never gates model control.
            before = time.monotonic()
            try:
                prepare_observation = getattr(ctx.sim, "prepare_observation", None)
                if callable(prepare_observation):
                    # Unspecified consumers retain fresh images on every request.
                    # Tick zero also supplies the shared first-frame artifact.
                    requires_rgb = getattr(adapter, "requires_rgb", True)
                    if type(requires_rgb) is not bool:
                        raise TypeError("adapter.requires_rgb must be boolean")
                    prepare_observation(tick == 0 or requires_rgb)
                observation = scene.observe(tick)
            finally:
                timings["observation_s"] += time.monotonic() - before
            if not isinstance(observation, Observation):
                raise TypeError("scene.observe() must return Observation")
            if observation.tick != tick or abs(observation.sim_time_s - tick / spec.clock.physics_hz) > 1e-7:
                raise ValueError("observation clock mismatch")
            if observation.protocol_version != spec.protocol_version:
                raise ValueError("observation protocol version mismatch")
            if tick == 0:
                first_frame_artifact = _first_frame(ctx.output_dir, observation.rgb)
            before = time.monotonic()
            try:
                action = adapter.act(observation)
                if not isinstance(action, Action):
                    raise TypeError("adapter.act() must return one Action")
                if expert_replay:
                    replay_rows_returned += 1
                model_tested = consumer_kind == ConsumerKind.MODEL
                peek = getattr(adapter, "peek_next_action", None)
                try:
                    following_action = peek() if callable(peek) else None
                except StopIteration:
                    raise RuntimeError("peek_next_action must return None at chunk/trajectory end") from None
                if following_action is not None and not isinstance(following_action, Action):
                    raise TypeError("adapter.peek_next_action() must return Action or None")
            except PolicyStop as stop:
                if consumer_kind != ConsumerKind.MODEL:
                    raise RuntimeError("policy stop is restricted to model consumers") from None
                model_tested = True
                termination = Termination.CALL_BUDGET if isinstance(stop, CallBudgetExhausted) else Termination.POLICY_STOP
                break
            except StopIteration:
                if not expert_replay:
                    raise RuntimeError("only expert replay may exhaust a trajectory") from None
                replay_eof = True
                termination = Termination.TRAJECTORY_EXHAUSTED
                break
            finally:
                elapsed = time.monotonic() - before
                timings["inference_s"] += elapsed
                timings["adapter_s"] += elapsed
            before = time.monotonic()
            action_hold_ticks = 0
            # Reaching the frozen physics budget is a normal replay endpoint,
            # including a partial final control hold. Coverage records it below.
            requested_ticks = getattr(adapter, 'action_physics_ticks', ticks_per_action)
            if type(requested_ticks) is not int or not 1 <= requested_ticks <= ticks_per_action:
                raise ValueError('adapter action duration must be within one control row')
            ticks_this_action = min(requested_ticks, limits.physics_ticks - tick)
            for subtick in range(ticks_this_action):
                tick += 1
                before_step = getattr(scene, "before_step", None)
                if before_step is not None:
                    with phases.measure('scene.before_step'):
                        before_step(tick)
                with phases.measure('action.interpolate'):
                    target = interpolate_action(action, following_action, subtick / ticks_per_action)
                with phases.measure('robot.apply'):
                    ctx.robot.apply(target)
                last_target = target
                action_execution["held_ticks" if following_action is None else "interpolated_ticks"] += 1
                if expert_replay and action_hold_ticks == 0:
                    replay_rows_dispatched += 1
                with phases.measure('sim.step'):
                    ctx.sim.step(render=tick % video_ticks == 0)
                action_hold_ticks += 1
                if expert_replay:
                    replay_physics_ticks_executed += 1
                    replay_last_hold_ticks = action_hold_ticks
                after_step = getattr(scene, "after_step", None)
                if after_step is not None:
                    with phases.measure('scene.after_step'):
                        after_step(tick)
                requires_rgb = getattr(scene, "requires_rgb", None)
                if callable(requires_rgb) and requires_rgb(tick):
                    ctx.sim.prepare_observation(True)
                with phases.measure('scene.evidence'):
                    evidence = scene.evidence(tick)
                with phases.measure('evidence.validate'):
                    if evidence.task_id != spec.task_id or evidence.instance_id != spec.instance_id or evidence.tick != tick:
                        raise ValueError("evidence instance/clock mismatch")
                capture = getattr(ctx.sim, "capture", None)
                if capture is not None and tick % video_ticks == 0:
                    with phases.measure('sim.capture'):
                        capture()
                with phases.measure('scene.completion'):
                    complete = (check_task_completion() if stop_on_success and completion_mode == 'events'
                                else False)
                    if not stop_on_success:
                        # Expert replay retains its historical coverage receipt;
                        # this diagnostic flag never truncates the trajectory.
                        complete = scene.is_complete()
                if complete and task_first_complete_tick is None:
                    task_first_complete_tick = tick
                if complete and stop_on_success:
                    termination = Termination.COMPLETED
                    break
            confirm_action = getattr(adapter, "confirm_action_executed", None)
            if callable(confirm_action):
                confirm_action(action_hold_ticks)
            # A model action may contain many control rows. Result tasks are
            # checked only after its last row, never in the middle of a motion.
            if stop_on_success and completion_mode == 'result' and following_action is None:
                with phases.measure('scene.completion'):
                    if check_task_completion():
                        task_first_complete_tick = tick
                        termination = Termination.COMPLETED
            timings["action_s"] += time.monotonic() - before
            if termination == Termination.COMPLETED:
                break
        if time.monotonic() - episode_start >= limits.wall_s:
            raise TimeoutError("episode wall budget exhausted")
        # T0 is the last executed actor physics row, with no post-actor steps.
        with phases.measure('scene.health'):
            health = scene.health()
        if not isinstance(health, Health) or not health.ok:
            raise RuntimeError("invalid final native health")
        if evidence is None:
            with phases.measure('scene.evidence'):
                evidence = scene.evidence(tick)
        with phases.measure('evidence.validate'):
            if evidence.task_id != spec.task_id or evidence.instance_id != spec.instance_id or evidence.tick != tick:
                raise ValueError("evidence instance/clock mismatch")
        before = time.monotonic()
        score = task.scorer(evidence)
        timings["score_s"] = time.monotonic() - before
        score_components = score.components
        write_json(ctx.output_dir / "evidence.json", evidence)
        write_json(ctx.output_dir / "score.json", score)
        result = Result(spec.task_id, spec.instance_id, score.value, termination, True,
                        consumer_kind, model_tested, tick, timings)
    except SystemExit as exc:
        result = Result(spec.task_id, spec.instance_id, None, Termination.ERROR,
                        False, consumer_kind, model_tested, tick, timings,
                        error=f"SystemExit({exc.code!r}) during episode")
    except Exception as exc:
        if getattr(exc, 'invalid_reason', None) == 'asset_unresolved':
            asset_failure = dict(invalid_reason='asset_unresolved', missing_paths=list(exc.missing_paths))
        result = Result(spec.task_id, spec.instance_id, None,
                        Termination.WALL_TIMEOUT if isinstance(exc, TimeoutError) else Termination.ERROR,
                        False, consumer_kind, model_tested, tick, timings,
                        error=f"{type(exc).__name__}: {exc}")
    finally:
        timings["episode_s"] = time.monotonic() - episode_start if episode_start is not None else 0.
        cleanup_errors = []
        if variation_session is not None:
            try:
                receipt = dict(variation_session.receipt or {})
                receipt.update(plan_sha256=variation_session.plan.sha256,
                               status="verified" if (receipt.get("post_initialization", {}).get("status") == "verified" and
                                   receipt.get("post_initialization_visual", {}).get("status") == "verified") else "failed",
                               error=result.error if result is not None else None)
                write_json(ctx.output_dir / "variation-receipt.json", receipt)
            except Exception as exc:
                cleanup_errors.append(f"variation receipt: {type(exc).__name__}: {exc}")
        try:
            write_json(ctx.output_dir / "action-execution.json", action_execution)
            write_json(ctx.output_dir / "task-completion.json", completion_receipt)
        except Exception as exc:
            cleanup_errors.append(f"action execution receipt: {type(exc).__name__}: {exc}")
        try:
            native_receipt = getattr(ctx.robot, "native_receipt", None)
            if native_receipt is not None:
                from collections.abc import Mapping
                receipt = native_receipt()
                if not isinstance(receipt, Mapping):
                    raise TypeError("robot native_receipt must return a JSON mapping")
                robot_receipt = {
                    "task_id": spec.task_id,
                    "instance_id": spec.instance_id,
                    **dict(receipt),
                }
                write_json(ctx.output_dir / "robot-receipt.json", robot_receipt)
        except Exception as exc:
            cleanup_errors.append(f"robot receipt: {type(exc).__name__}: {exc}")
        try:
            result_metadata = getattr(adapter, "result_metadata", None)
            if result_metadata is not None:
                from mind2act_eval.contracts import freeze
                from collections.abc import Mapping
                metadata = result_metadata()
                if not isinstance(metadata, Mapping):
                    raise TypeError("adapter result_metadata must return a JSON mapping")
                policy_metadata = freeze(metadata)
        except Exception as exc:
            cleanup_errors.append(f"adapter metadata: {type(exc).__name__}: {exc}")
        try:
            task_result_metadata = getattr(scene, "result_metadata", None)
            if task_result_metadata is not None:
                from mind2act_eval.contracts import freeze
                from collections.abc import Mapping
                metadata = task_result_metadata()
                if not isinstance(metadata, Mapping):
                    raise TypeError("scene result_metadata must return a JSON mapping")
                task_metadata = freeze(metadata)
        except Exception as exc:
            cleanup_errors.append(f"task metadata: {type(exc).__name__}: {exc}")
        try:
            coverage_factory = getattr(adapter, "coverage_receipt", None)
            if coverage_factory is not None:
                from collections.abc import Mapping
                raw_coverage = coverage_factory()
                if not isinstance(raw_coverage, Mapping):
                    raise TypeError("adapter coverage_receipt must return a JSON mapping")
                allowed = {"rows_total", "adapter_returned_rows", "adapter_eof"}
                if set(raw_coverage) != allowed:
                    raise ValueError("adapter replay coverage fields do not match the allowlist")
                total_rows = raw_coverage["rows_total"]
                returned_rows = raw_coverage["adapter_returned_rows"]
                adapter_eof = raw_coverage["adapter_eof"]
                if type(total_rows) is not int or type(returned_rows) is not int or type(adapter_eof) is not bool:
                    raise TypeError("adapter replay coverage has invalid field types")
                if returned_rows != replay_rows_returned:
                    raise ValueError("adapter replay row count disagrees with runner action returns")
                expected_ticks = total_rows * ticks_per_action
                executed_ticks = replay_physics_ticks_executed
                clock_coverage_complete = (
                    adapter_eof and replay_eof and total_rows == returned_rows == replay_rows_dispatched
                    and executed_ticks == expected_ticks and replay_last_hold_ticks == ticks_per_action
                )
                replay_coverage = {
                    "schema": "mind2act_eval.replay_coverage.v1",
                    "rows_total": total_rows,
                    "adapter_returned_rows": returned_rows,
                    "dispatched_rows": replay_rows_dispatched,
                    "adapter_eof": adapter_eof and replay_eof,
                    "ticks_per_action": ticks_per_action,
                    "expected_physics_ticks": expected_ticks,
                    "executed_physics_ticks": executed_ticks,
                    "last_hold_expected_ticks": ticks_per_action if replay_rows_dispatched else 0,
                    "last_hold_executed_ticks": replay_last_hold_ticks,
                    "task_first_complete_tick": task_first_complete_tick,
                    "clock_coverage_complete": clock_coverage_complete,
                    "truncated": not clock_coverage_complete,
                }
                if expert_replay and replay_success_terminal:
                    replay_coverage["stopped_on_native_success"] = (
                        termination == Termination.COMPLETED and task_first_complete_tick is not None)
                write_json(ctx.output_dir / "replay-coverage.json", replay_coverage)
        except Exception as exc:
            cleanup_errors.append(f"replay coverage: {type(exc).__name__}: {exc}")
        try:
            write_json(ctx.output_dir / "preclose-trace.json", {
                "schema": "mind2act_eval.preclose_trace.v1",
                "stage": "awaiting_cleanup",
                "task_id": spec.task_id,
                "instance_id": spec.instance_id,
                "tick": tick,
                "result": result,
                "robot_receipt": robot_receipt,
                "sim_artifacts": dict(getattr(ctx.sim, "artifacts", {})),
            })
        except Exception as exc:
            cleanup_errors.append(f"pre-close trace: {type(exc).__name__}: {exc}")
        try:
            write_json(ctx.output_dir / "cleanup-started.json", {
                "schema": "mind2act_eval.cleanup_started.v1",
                "task_id": spec.task_id,
                "instance_id": spec.instance_id,
                "tick": tick,
            })
        except Exception as exc:
            cleanup_errors.append(f"cleanup marker: {type(exc).__name__}: {exc}")
        arm_cleanup_diagnostics = getattr(ctx.sim, "arm_cleanup_diagnostics", None)
        if callable(arm_cleanup_diagnostics):
            try:
                trace_info = arm_cleanup_diagnostics()
                write_json(ctx.output_dir / "cleanup-trace-armed.json", {
                    "schema": "mind2act_eval.cleanup_trace_armed.v1",
                    "pid": __import__("os").getpid(),
                    "timeout_s": 90,
                    "repeat": True,
                    "trace": "cleanup-traceback.log",
                    "signal": trace_info.get("signal_name") if isinstance(trace_info, dict) else None,
                })
            except Exception as exc:
                # Missing diagnostic output must not change episode validity.
                try:
                    write_json(ctx.output_dir / "cleanup-trace-arming-error.json", {
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                except Exception:
                    pass
        for resource in (adapter, scene):
            if resource is not None:
                try:
                    resource.close()
                except SystemExit as exc:
                    cleanup_errors.append(f"SystemExit from {type(resource).__name__}.close (code={exc.code!r})")
                except Exception as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if ctx.sim is not None:
            try:
                prepare_for_os_exit = getattr(ctx.sim, "prepare_for_os_exit", None)
                if (result is not None and (result.infra_valid or on_worker_exit_prepared is not None)
                        and callable(prepare_for_os_exit)):
                    prepare_for_os_exit()
                else:
                    ctx.sim.close()
            except SystemExit as exc:
                cleanup_errors.append(f"SystemExit from {type(ctx.sim).__name__}.close (code={exc.code!r})")
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if cleanup_errors:
            errors = ([result.error] if result is not None and result.error else []) + cleanup_errors
            result = Result(spec.task_id, spec.instance_id, None, Termination.ERROR, False,
                            consumer_kind, model_tested, tick, timings,
                            error="cleanup: " + "; ".join(errors))
    timings["total_s"] = time.monotonic() - start
    # Reconstruct to freeze the final timings after cleanup.
    from dataclasses import replace
    timings.update(getattr(ctx.sim, "timings", {}))
    timings["overhead_s"] = max(0., timings["total_s"] - sum(timings[key] for key in ("build_s", "adapter_s", "observation_s", "action_s", "score_s")))
    artifacts = dict(getattr(ctx.sim, "artifacts", {}))
    if first_frame_artifact is not None:
        artifacts["first_frame_head"] = first_frame_artifact
    if spec.variation.mode == "random":
        from mind2act_eval.contracts import freeze
        from mind2act_eval.variations import file_digest
        receipt_path = Path(ctx.output_dir) / "variation-receipt.json"
        has_receipt = variation_session is not None and receipt_path.is_file()
        if has_receipt:
            artifacts["variation_receipt"] = "variation-receipt.json"
        receipt = (variation_session.receipt or {}) if variation_session is not None else {}
        task_metadata = freeze({**dict(task_metadata), "variation": {
            **to_json(spec.variation),
            "plan_sha256": variation_session.plan.sha256 if variation_session is not None else None,
            "receipt": "variation-receipt.json" if has_receipt else None,
            "receipt_sha256": file_digest(receipt_path) if has_receipt else None,
            "authored": receipt.get("mode") == "random",
            "applied": (receipt.get("post_initialization", {}).get("status") == "verified" and
                        receipt.get("post_initialization_visual", {}).get("status") == "verified"),
            "post_initialization": receipt.get("post_initialization"),
            "post_initialization_visual": receipt.get("post_initialization_visual"),
        }})
        if task_metadata["variation"]["applied"] is not True:
            result = replace(result, score=None, infra_valid=False,
                termination=Termination.ERROR if result.infra_valid else result.termination,
                error="; ".join(filter(None, (result.error,
                    "variation_not_applied: random requires task_metadata.variation.applied=true"))))
    if (Path(ctx.output_dir) / "initial-environment-hold.json").exists():
        artifacts["initial_environment_hold"] = "initial-environment-hold.json"
    if (Path(ctx.output_dir) / "observation-closeout.json").exists():
        artifacts["observation_closeout"] = "observation-closeout.json"
    if (Path(ctx.output_dir) / "action-execution.json").exists():
        artifacts["action_execution"] = "action-execution.json"
    if (Path(ctx.output_dir) / "task-completion.json").exists():
        artifacts["task_completion"] = "task-completion.json"
    if (Path(ctx.output_dir) / "cleanup-started.json").exists():
        artifacts["cleanup_started"] = "cleanup-started.json"
    if (Path(ctx.output_dir) / "cleanup-trace-armed.json").exists():
        artifacts["cleanup_traceback"] = "cleanup-traceback.log"
    if robot_receipt is not None:
        artifacts["native_robot_receipt"] = "robot-receipt.json"
    if (Path(ctx.output_dir) / "preclose-trace.json").exists():
        artifacts["preclose_trace"] = "preclose-trace.json"
    if replay_coverage is not None and (Path(ctx.output_dir) / "replay-coverage.json").exists():
        artifacts["replay_coverage"] = "replay-coverage.json"
    if asset_failure is not None:
        task_metadata = {**task_metadata, **asset_failure}
        artifacts['asset_resolution_report'] = 'asset-resolution-report.json'
    result = replace(result, policy_kind=limits.policy_kind, budget_version=limits.budget_version,
                     budget_limits=budget_receipt if limits.policy_kind else {},
                     **(calls.receipt() if calls is not None else {}))
    if limits.policy_kind:
        write_json(ctx.output_dir / 'run-budget.json', dict(budget_receipt,
                   **(calls.receipt() if calls is not None else {})))
        artifacts['run_budget'] = 'run-budget.json'
    if calls is not None:
        artifacts['call_budget'] = 'call-budget.json'
    result = replace(result, timings=timings, policy_metadata=policy_metadata,
                     scoring_version=score_components.get('scoring_version', result.scoring_version),
                     score_components=score_components, task_metadata=task_metadata,
                     artifacts=artifacts, video_metadata={
                         "video_hz": getattr(ctx.sim, "video_hz", 6),
                         "frame_counts": dict(getattr(getattr(ctx.sim, "recorder", None), "frame_counts", {})),
                         "sampling": "tick_zero_then_physics_tick_multiples",
                     })
    # Aggregate once after the original lifecycle. Hard-killed workers may have
    # no final breakdown, just as they may have no final legacy timings.
    try:
        backend = getattr(ctx.sim, '_phase_timings', None)
        video = getattr(getattr(ctx.sim, 'recorder', None), '_phase_timings', None)
        _write_timing_breakdown(ctx.output_dir / 'timing-breakdown.json', dict(
            schema='mind2act_eval.timing_breakdown.v1', clock='monotonic',
            nested_not_additive=True, runner=phases.snapshot(),
            sim=backend.snapshot() if backend is not None else {},
            video=video.snapshot() if video is not None else {}))
        result = replace(result, artifacts={**dict(result.artifacts),
                         'timing_breakdown': 'timing-breakdown.json'})
    except Exception:
        pass  # Optional diagnostics must not change episode validity.
    write_json(ctx.output_dir / "spec.json", spec)
    write_json(ctx.output_dir / "result.json", result)
    if (on_worker_exit_prepared is not None and not cleanup_errors
            and getattr(ctx.sim, "os_exit_prepared", False) is True):
        on_worker_exit_prepared(result)
    return result
