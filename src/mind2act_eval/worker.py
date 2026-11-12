"""Restricted native diagnostic branch; normal episodes remain in the CLI runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import time
import zlib

from .contracts import BuildContext, ConsumerKind, Health, Result, Termination, to_json
from .runner import write_json
from .scene_probe import (PROJECTION, RECEIPT_SCHEMA, ProbeInputs, SceneProbeReceipt,
                          sha256, spec_digest, stability, validate_clock_delta)


def world_clock(world):
    return {"tick": getattr(world, "current_time_step_index", None), "time_s": getattr(world, "current_time", None)}


def write_png(path, rgb):
    """Lossless head image using only stdlib; no resize or screen capture."""
    import numpy as np
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
        raise ValueError("head image must be nonempty uint8 RGB")
    height, width = pixels.shape[:2]
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    raw = b"".join(b"\0" + pixels[row].tobytes() for row in range(height))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    with Path(path).open("xb") as stream:
        stream.write(data)


def run_probe_loop(spec, task, ctx, inputs, *, wall_s, native=None):
    """Run a diagnostic without an adapter, scorer, action command or terminal gate."""
    if native is None:
        from . import scene_probe_native as native
    start = time.monotonic()
    source_clean = inputs.is_source_clean
    failure_metadata = {}
    scene, world, samples = None, None, []
    entry, exit_clock = {}, {}
    tick, error, modes, clutter, appearance = 0, None, {}, [], []
    before_head = after_head = None
    checks = {}
    source_initialization_exit_clock = {}
    cleanup_errors = []
    try:
        # Same actual frozen source spec binds both shared services before any
        # task build/World access (not a replacement diagnostic spec).
        for service in (ctx.sim, ctx.robot):
            bind = getattr(service, "bind_task_instance", None)
            if callable(bind):
                bind(spec, ctx.output_dir)
        # Preserve source initialization without random clutter participating in
        # its own settling loop (PR4 can legitimately consume thousands of ticks).
        scene = task.build(spec, ctx)
        world = ctx.world
        ctx.robot.initialize(world)  # idempotent; retains native rest targets
        source_initialization_exit_clock = world_clock(world)
        modes["before_projection"] = native.physics_modes(ctx.sim.stage)
        if not source_clean:
            clutter = native.spawn_clutter(ctx.sim.stage, inputs, inputs.catalog["asset_root"])
            appearance = native.apply_appearance(ctx.sim.stage, inputs, inputs.catalog["asset_root"])
        modes["after_projection"] = native.physics_modes(ctx.sim.stage)
        if modes["before_projection"] != modes["after_projection"]:
            raise ValueError("projection changed protected task/robot physical modes")
        # Existing render-only warmup processes USD edits without consuming any
        # physics tick. Native actor/view registration must succeed; never reset
        # the source World or freeze/repair geometry to make it succeed.
        ctx.sim.ready()
        monitor = (None if source_clean else native.NativeMonitor(
            ctx, clutter, floor_targets=inputs.catalog["floor_contact_targets"]))
        validate_clock_delta(source_initialization_exit_clock, world_clock(world), 0)
        write_json(ctx.output_dir / "ready.json", {"ready": True, "pid": os.getpid(), "consumer_kind": "scene_probe"})
        entry = world_clock(world)
        validate_clock_delta(entry, entry, 0)
        write_png(ctx.output_dir / "before-head.png", ctx.sim.rgb()["head"])
        before_head = "before-head.png"
        samples.append({} if monitor is None else monitor.sample())
        with (ctx.output_dir / "scene-probe-samples.jsonl").open("x") as stream:
            stream.write(json.dumps({"tick": 0, "clock": entry, "bodies": samples[-1]}, allow_nan=False)+"\n")
            for step in range(1, 601):
                if time.monotonic() - start >= wall_s:
                    raise TimeoutError("scene probe wall budget exhausted")
                health = scene.health()
                if not isinstance(health, Health) or not health.ok:
                    raise RuntimeError("native scene unhealthy during probe: " + str(getattr(health, "reason", health)))
                # No robot.apply: its initialized native targets persist. Task
                # hooks still run; is_complete is deliberately never consulted.
                scene.before_step(step)
                previous = world_clock(world)
                ctx.sim.step(render=False)
                tick = step
                current = world_clock(world)
                validate_clock_delta(previous, current, 1)
                scene.after_step(step)
                samples.append({} if monitor is None else monitor.sample())
                stream.write(json.dumps({"tick": step, "clock": current, "bodies": samples[-1]}, allow_nan=False)+"\n")
            stream.flush()
        exit_clock = world_clock(world)
        validate_clock_delta(entry, exit_clock, 600)
        # Actual live sensor capture at the post-settle clock, zero physics delta.
        ctx.sim.capture()
        write_png(ctx.output_dir / "after-head.png", ctx.sim.rgb()["head"])
        after_head = "after-head.png"
        validate_clock_delta(exit_clock, world_clock(world), 0)
        if source_clean:
            checks = {"source_clean_preview": True, "projection_applied": False,
                      "requires_image_review": True}
        else:
            checks = stability(samples, support_z=inputs.catalog["support_z_m"])
            checks["layout_verdict"] = "requires_image_and_task_interference_review"
        modes["after_600_steps"] = native.physics_modes(ctx.sim.stage)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if getattr(exc, "invalid_reason", None) == "asset_unresolved":
            failure_metadata = {"invalid_reason": "asset_unresolved", "missing_paths": list(exc.missing_paths)}
        if world is not None:
            exit_clock = world_clock(world)
    finally:
        write_json(ctx.output_dir / "cleanup-started.json", {"consumer_kind": "scene_probe", "tick": tick})
        if scene is not None:
            try:
                scene.close()
            except BaseException as exc:
                cleanup_errors.append(f"scene.close: {type(exc).__name__}: {exc}")
        try:
            ctx.sim.prepare_for_os_exit()
        except BaseException as exc:
            cleanup_errors.append(f"prepare_for_os_exit: {type(exc).__name__}: {exc}")
    if cleanup_errors:
        error = "; ".join(([error] if error else []) + cleanup_errors)
    valid = error is None
    receipt = SceneProbeReceipt(RECEIPT_SCHEMA, spec.task_id, spec.instance_id, inputs.config.get("logical_slot"),
        inputs.config["projection"], valid, tick, entry, exit_clock, spec_digest(spec), inputs.config_sha256,
        inputs.config.get("layout_sha256"), before_head, after_head, checks,
        {"generation_seed": inputs.config.get("generation_seed"), "seed_provenance": inputs.config.get("seed_provenance"),
         "source_seed": spec.seed, "source_id": spec.source_id, "source_assets": to_json(spec.assets),
         "catalog_sha256": inputs.config.get("catalog_sha256"), "catalog_source": (inputs.catalog or {}).get("source_provenance"),
         "physics_modes": modes, "clutter": clutter, "appearance": appearance,
         "source_initialization_exit_clock": source_initialization_exit_clock,
         "projection_phase": ("unchanged_source_initialization; no_projection" if source_clean else
                              "after_original_task_initialization; zero_physics_delta_before_entry"),
         "robot_policy": "native_initialization_targets_retained; no action commands",
         "clutter_physics_policy": ("not_applied_source_clean" if source_clean else
                                   "fixed_source_clutter_spawn_defaults; no task/robot/collider edits"),
         "code_sha256": {p.relative_to(Path(__file__).parent).as_posix(): sha256(p)
                         for p in sorted(Path(__file__).parent.rglob("*.py"))}}, error)
    write_json(ctx.output_dir / "scene-probe-receipt.json", receipt)
    artifacts = {"scene_probe_receipt": "scene-probe-receipt.json", **dict(ctx.sim.artifacts)}
    for key, filename in (("before_head", before_head), ("after_head", after_head)):
        if filename:
            artifacts[key] = filename
    if (ctx.output_dir / "scene-probe-samples.jsonl").is_file():
        artifacts["scene_probe_samples"] = "scene-probe-samples.jsonl"
    result = Result(spec.task_id, spec.instance_id, None, Termination.COMPLETED if valid else Termination.ERROR,
                    valid, ConsumerKind.SCENE_PROBE, False, ticks=tick,
                    timings={"wall_s": time.monotonic()-start}, error=error, artifacts=artifacts,
                    task_metadata={"diagnostic_projection": inputs.config["projection"], "excluded_from_benchmark": True, **failure_metadata},
                    video_metadata={"video_hz": getattr(ctx.sim, "video_hz", 6),
                        "frame_counts": dict(getattr(getattr(ctx.sim, "recorder", None), "frame_counts", {}))})
    write_json(ctx.output_dir / "result.json", result)
    return result


def run_scene_probe(args, *, gpu_selection=None):
    from .cli import _gpu_selection_kwargs, read_spec
    from .robot import X5Robot
    from .sim import IsaacSimulator, LocalAssets
    from .tasks.catalog import load_task
    spec = read_spec(args.spec)
    gpu_kwargs = _gpu_selection_kwargs(args) if gpu_selection is None else {"gpu_selection": gpu_selection}
    inputs = ProbeInputs.read(args.probe_config, args.probe_config_sha256, spec)
    output = Path(args.output)
    # Preserve exact data inputs independently of archive-provided receipts.
    archives = [(args.probe_config, args.probe_config_sha256, "probe-config.json")]
    if not inputs.is_source_clean:
        archives.extend([(inputs.config["layout"], inputs.config["layout_sha256"], "layout.json"),
                         (inputs.config["catalog"], inputs.config["catalog_sha256"], "probe-catalog.json")])
    for source, expected, name in archives:
        raw = Path(source).read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("probe input changed before archival")
        with (output / name).open("xb") as stream:
            stream.write(raw)
    assets = LocalAssets(Path(args.asset_root), spec.assets, spec=spec)
    sim = None
    try:
        material_receipt = None
        if 'material_mdl_search_asset_ids' in spec.config:
            from .sim.material_closure import configure_mdl_search
            material_receipt = configure_mdl_search(assets, spec.config['material_mdl_search_asset_ids'])
        sim = IsaacSimulator(spec.clock, output, physical_gpu=args.gpu, cameras=spec.config.get("cameras"),
                             ffmpeg=spec.config.get("ffmpeg", "ffmpeg"), video_hz=getattr(args, "video_hz", 6), **gpu_kwargs)
        if material_receipt is not None:
            from .sim.material_closure import verify_native_mdl_search
            write_json(output / 'material-resolver.json', verify_native_mdl_search(material_receipt))
        if not inputs.is_source_clean or "station" in spec.assets:
            from .sim.station_deployment import prepare_open_station
            sim.open_stage(prepare_open_station(spec, assets, output))
        robot = X5Robot(rgb_reader=sim.rgb, usd_path=assets.resolve("x5"), rest=spec.config.get("robot_rest"),
                        physics_hz=spec.clock.physics_hz)
        return run_probe_loop(spec, load_task(spec.task_id), BuildContext(sim, robot, assets, output), inputs,
                              wall_s=args.wall_s)
    except BaseException:
        if sim is not None:
            sim.close()
        raise
