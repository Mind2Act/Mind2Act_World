"""Asset resolution and lazy Isaac lifecycle. No Kit imports at module scope."""
from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
import json
from math import isfinite
from pathlib import Path
import os
import sys
import time
from mind2act_eval.timing import phases_for
from mind2act_eval.startup_paths import (GATE_SCHEMA, read_gate_state, startup_paths,
                                      startup_configuration)

CAMERA_WARMUP_RENDER_LIMIT = 8
CAPTURE_GRAPH_WARMUP_STEPS = 2
CAPTURE_GRAPH_WARMUP_SUBFRAMES = 8
CAPTURE_SUBFRAMES = 2
CAPTURE_GRAPH_BASELINE_SHA256 = "a5d6a772aa26f7f4d5172dbc382c771997d443a5e4837eb72528a6fe7725149f"


class LocalAssets:
    def __init__(self, root: Path, entries, *, spec=None):
        self.root = Path(root).resolve(strict=True)
        self.entries = dict(entries)
        if spec is not None:
            from mind2act_eval.public_layouts import validate
            validate(spec)
        self.spec = spec

    def resolve(self, asset_id: str) -> Path:
        path = (self.root / self.entries[asset_id]).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("asset path escapes configured root")
        return path


def _write_gate_state(lock, path, state):
    payload = json.dumps({"schema": GATE_SCHEMA, "state": state}) + "\n"
    lock.seek(0)
    lock.truncate()
    lock.write(payload)
    lock.flush()
    os.fsync(lock.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _gate_wait(seconds, *, clock=time, cancel_event=None):
    if cancel_event is None:
        clock.sleep(seconds)
        return
    end = clock.monotonic() + seconds
    while clock.monotonic() < end:
        if cancel_event.is_set():
            raise InterruptedError('startup cancelled during cooldown')
        clock.sleep(min(.1, end - clock.monotonic()))


def recover_startup_state(lock, path, interval_s, *, inherited_cooldown=False, clock=time, cancel_event=None):
    """Recover this exact gate under its EX lock; never transfer monotonic timestamps."""
    lock.seek(0)
    raw = lock.read()
    state = read_gate_state(raw)
    if state in {'active', 'cooling'} or state is None and inherited_cooldown:
        _gate_wait(interval_s, clock=clock, cancel_event=cancel_event)
        _write_gate_state(lock, path, 'idle')


@contextmanager
def startup_gate(path: Path, *, shared=False, minimum_interval_s=0.0, cancel_event=None,
                 inherited_cooldown=False, clock=time, domain_check=None, track_state=False):
    """Lock a launcher seam, optionally enforcing a crash-safe Kit start interval.

    Shared mode is for the legacy batch compatibility guard. A Kit gate is
    exclusive and durably tracks active/cooling/idle so an owner crash forces a
    full cooldown on the next launcher. State tracking can also be retained
    explicitly when the configured cooldown is zero.
    """
    if type(shared) is not bool:
        raise TypeError("shared must be bool")
    if type(track_state) is not bool:
        raise TypeError("track_state must be bool")
    if (type(minimum_interval_s) not in (int, float) or not isfinite(minimum_interval_s)
            or minimum_interval_s < 0):
        raise ValueError("minimum_interval_s must be finite and nonnegative")
    if shared and (minimum_interval_s or track_state):
        raise ValueError("shared gates cannot enforce a start interval or state tracking")
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if cancel_event is None:
            fcntl.flock(lock, operation)
        else:
            while True:
                if cancel_event.is_set():
                    raise InterruptedError("launcher cancelled before lock acquisition")
                try:
                    fcntl.flock(lock, operation | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    clock.sleep(.1)
        if domain_check is not None:
            domain_check()  # Under the lock, before any persistent state write.
        started = None
        armed = False
        transferred = False
        try:
            if minimum_interval_s or track_state:
                recover_startup_state(lock, path, minimum_interval_s, inherited_cooldown=inherited_cooldown,
                                      clock=clock, cancel_event=cancel_event)
                _write_gate_state(lock, path, "active")
                armed = True
            def mark_started():
                nonlocal started
                started = clock.monotonic()
            def transfer_to_child():
                nonlocal transferred
                transferred = True
            mark_started.lock_fd = lock.fileno()
            mark_started.transfer_to_child = transfer_to_child
            yield mark_started
        finally:
            try:
                if armed and not transferred:
                    _write_gate_state(lock, path, "cooling")
                    elapsed = 0.0 if started is None else clock.monotonic() - started
                    remaining = minimum_interval_s - elapsed
                    try:
                        if remaining > 0:
                            _gate_wait(remaining, clock=clock, cancel_event=cancel_event)
                    except InterruptedError:
                        # Retain cooling for the next owner, but never mask the
                        # supervisor's result/exception during owned cleanup.
                        pass
                    else:
                        _write_gate_state(lock, path, "idle")
            finally:
                # Inherited descriptors share one open-file description: an
                # explicit UN would also unlock the live child's lease.
                if not transferred:
                    fcntl.flock(lock, fcntl.LOCK_UN)



@contextmanager
def kit_startup_gate(paths):
    """Bound concurrent initialization and stagger starts within one node."""
    slots, interval = startup_configuration()
    if slots == 1:
        with startup_gate(paths.kit, minimum_interval_s=interval) as mark:
            yield mark
        return
    import fcntl
    # Shared ownership excludes any old/default serial initializer. Each slot
    # remains held through ready(), not just SimulationApp construction.
    with startup_gate(paths.kit, shared=True):
        lease = None
        try:
            while lease is None:
                for index in range(slots):
                    candidate = (paths.root / f'kit-init-{index}.lock').open('a+')
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                    else:
                        lease = candidate
                        break
                if lease is None:
                    time.sleep(.1)
            # This small gate is released before construction; it must not
            # serialize the full initialization lifetime across all slots.
            with startup_gate(paths.root / 'kit-start.lock', minimum_interval_s=interval) as mark:
                mark()
            yield lambda: None
        finally:
            if lease is not None:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()


def recording_cameras(cameras, *, include_wrists=False):
    """Extra recording views do not extend the policy observation contract."""
    configured = dict(cameras or {'head': {'path': '/World/HeadCamera', 'resolution': [640, 480]}})
    observations = tuple(configured)
    if include_wrists:
        for side, root in (('left', 'RobotLeft'), ('right', 'RobotRight')):
            configured.setdefault(side + '_wrist', {
                'path': f'/World/{root}/Asset/camera/PhysCoWristCamera',
                'resolution': [640, 480],
            })
    return configured, observations


def _require_isaac_extension_dependencies():
    for module in ("psutil", "requests", "scipy", "toml", "torch"):
        try:
            available = find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            raise RuntimeError(
                f"Isaac extension dependency {module!r} is missing; install the 'isaac' runtime extra"
            )


class IsaacSimulator:
    """One app/world per child process. Call ready() after build/reset completes."""
    def __init__(self, clock, output_dir: Path, *, physical_gpu, headless=True, cameras=None, ffmpeg="ffmpeg", video_hz=6, record_wrist_cameras=False):
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise ValueError("explicit nonnegative physical GPU index required")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
            raise ValueError("CUDA visibility must match the assigned physical GPU")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        self._gate = None
        self._ready = False
        self._contacts = None
        self._cameras = {}
        self._camera_receipt = None
        self._initial_rgb = None
        self._sampled_rgb = None
        self._observation_rgb_required = True
        self._rendered_since_step = False
        self._active_task_spec = None
        self._physics_profile = None
        self._physics_profile_receipt = None
        self._world_creation_started = False
        self._world_configuration_error = None
        self._camera_config, self._observation_camera_names = recording_cameras(
            cameras, include_wrists=record_wrist_cameras)
        self.timings = {"physics_s": 0., "render_s": 0., "video_s": 0.}
        from mind2act_eval.sim.video import RGBRecorder, video_interval
        # Runtime output configuration is deliberately outside InstanceSpec.clock.
        video_interval(clock.physics_hz, video_hz)
        self.video_hz = video_hz
        self.recorder = RGBRecorder(output_dir, video_hz, ffmpeg=ffmpeg)
        self.app = None
        try:
            _require_isaac_extension_dependencies()
            paths = startup_paths()
            from isaacsim import SimulationApp
            runtime_dir = output_dir / "kit"
            from mind2act_eval.episode_storage import worker_cache
            shared_cache = worker_cache()
            runtime_paths = {
                "user_config": runtime_dir / "user.config.json",
                "cache": runtime_dir / "cache",
                "data": runtime_dir / "data",
                "logs": runtime_dir / "logs",
            }
            for path in runtime_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == "":
                    path.mkdir(parents=True, exist_ok=True)
            extra_args = [
                "--/app/settings/persistent=false",
                "--/app/userConfigPath=" + str(runtime_paths["user_config"]),
                "--/app/cachePath=" + str(runtime_paths["cache"]),
                "--/app/dataPath=" + str(runtime_paths["data"]),
                "--/app/logPath=" + str(runtime_paths["logs"]),
                "--/log/file=" + str(runtime_dir / "kit.log"),
            ]
            if shared_cache:
                # These are the actual Kit 107 GPU Foundation settings. Keep
                # mutable app/cooked-mesh databases private so shared caches do
                # not need an exclusive lock over a full episode's lifetime.
                extra_args.append("--/rtx/shaderDb/shaderCachePath=" + str(shared_cache / "shaders"))
                extra_args.append("--/rtx/shaderDb/driverShaderCachePath=" + str(shared_cache / "driver-shaders"))
                # Registry downloads are reusable; app data, settings, logs,
                # documents and credentials remain per-attempt.
                extra_args.append("--/app/extensions/registryCache=" + str(Path(shared_cache) / "extensions"))
                extra_args.append("--/rtx-transient/resourcemanager/localTextureCachePath=" + str(shared_cache / "textures"))
            original_argv = sys.argv[:]
            try:
                sys.argv.extend(("--portable-root", str(runtime_dir)))
                gate = kit_startup_gate(paths)
                mark_started = gate.__enter__()
                self._gate = gate
                active_gpu = physical_gpu
                physics_gpu = 0
                # Start encoders before Kit grows the process image.
                self.recorder.prepare({name: tuple(config["resolution"])
                                       for name, config in self._camera_config.items()})
                self.app = SimulationApp({"headless": headless, "active_gpu": active_gpu,
                                          "physics_gpu": physics_gpu, "multi_gpu": False,
                                          # Kit's native fast-shutdown path may terminate the
                                          # process before the worker can persist final artifacts.
                                          "fast_shutdown": False,
                                          "extra_args": extra_args})
                mark_started()
                if shared_cache:
                    # Keep the startup GPU table, then stop duplicating verbose
                    # per-frame Info into kit.log. Warnings/errors continue on
                    # stdout in the retained worker.log; Kit errors also remain
                    # in kit.log. Scoring/media/evidence paths are unaffected.
                    import carb.settings
                    settings = carb.settings.get_settings()
                    expected_cache_settings = {
                        '/app/cachePath': str(runtime_paths['cache']),
                        '/app/dataPath': str(runtime_paths['data']),
                        '/app/extensions/registryCache': str(shared_cache / 'extensions'),
                        '/rtx/shaderDb/shaderCachePath': str(shared_cache / 'shaders'),
                        '/rtx/shaderDb/driverShaderCachePath': str(shared_cache / 'driver-shaders'),
                        '/rtx-transient/resourcemanager/localTextureCachePath': str(shared_cache / 'textures'),
                    }
                    applied_cache_settings = {key: settings.get(key) for key in expected_cache_settings}
                    if applied_cache_settings != expected_cache_settings:
                        raise RuntimeError('Kit cache setting readback differs from the configured storage paths')
                    settings.set('/log/fileLogLevel', 'Error')
                    from mind2act_eval.runner import write_json
                    write_json(output_dir / 'node-cache.json', dict(
                        schema='mind2act_eval.node_cache.v1', cache=str(shared_cache),
                        concurrency='shared inherited active-user FD; existing startup gate; no whole-episode exclusive lock',
                        settings=applied_cache_settings,
                        private_paths=['kit/cache', 'kit/data', 'kit/user.config.json', 'kit/logs', 'cache/ov/client'],
                        kit_file_log='startup Info, then Error; stdout Warning retained in worker.log'))
            finally:
                sys.argv[:] = original_argv
            self._clock = clock
            self._world = None
        except BaseException as startup_error:
            from mind2act_eval.runtime_diagnostics import checkpoint_prerun_failure
            checkpoint_prerun_failure(output_dir, startup_error, phase='constructor', source_file=__file__)
            try:
                self.close()
            except BaseException as cleanup_error:
                try:
                    print(f'constructor cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}', file=sys.stderr)
                    sys.stderr.flush()
                except BaseException:
                    pass
            raise

    @property
    def stage(self):
        from isaacsim.core.utils.stage import get_current_stage
        return get_current_stage()

    def open_stage(self, path: Path):
        if self._world is not None:
            raise RuntimeError("open stage before first ctx.world access")
        from isaacsim.core.utils.stage import open_stage
        if not open_stage(str(path)):
            raise RuntimeError("failed to open prepared stage")

    @property
    def active_task_spec(self):
        """The same frozen instance bound to robot/sim by the shared runner."""
        return getattr(self, "_active_task_spec", None)

    def bind_task_instance(self, spec, output_dir):
        """Select source capabilities before the first lazy World construction."""
        from mind2act_eval.sim.physics_profile import profile_for_instance
        if Path(output_dir).resolve() != self._output_dir:
            raise ValueError("sim task output must match its owned output directory")
        active = self.active_task_spec
        if active is not None:
            if active is not spec:
                raise ValueError("sim active task binding cannot be replaced")
            return
        if self._world is not None or getattr(self, "_world_creation_started", False):
            raise RuntimeError("bind the actual task before the first World construction")
        profile = profile_for_instance(spec)
        if profile is not None and self._clock != spec.clock:
            raise ValueError("PR4 simulator clock binding mismatch")
        self._active_task_spec, self._physics_profile = spec, profile
        if profile is not None:
            self._physics_profile_receipt = profile.receipt()
            self._write_physics_profile_receipt()

    def _write_physics_profile_receipt(self):
        receipt = getattr(self, "_physics_profile_receipt", None)
        if receipt is not None:
            path = self._output_dir / "physics-profile-receipt.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)

    def _verify_physics_profile(self, phase):
        receipt = getattr(self, "_physics_profile_receipt", None)
        if receipt is not None:
            from mind2act_eval.sim.physics_profile import verify
            try:
                verify(self.world, receipt, phase)
            except Exception as exc:
                receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                self._world_configuration_error = exc
                raise
            finally:
                self._write_physics_profile_receipt()

    @property
    def world(self):
        if getattr(self, "_world_configuration_error", None) is not None:
            raise RuntimeError("physics profile initialization previously failed") from self._world_configuration_error
        if self._world is None:
            from isaacsim.core.api import World
            self._world_creation_started = True
            try:
                self._world = World(physics_dt=1 / self._clock.physics_hz,
                                    rendering_dt=1 / self._clock.rgb_hz, stage_units_in_meters=1.0)
                from mind2act_eval.sim.asset_resolution import WorldAssetGate
                self._asset_resolution_gate = WorldAssetGate(self._world, self.stage, self._output_dir)
                if getattr(self, "_physics_profile", None) is not None:
                    from mind2act_eval.sim.physics_profile import configure
                    configure(self._world, self._physics_profile_receipt)
                    self._write_physics_profile_receipt()
            except Exception as exc:
                self._world_configuration_error = exc
                receipt = getattr(self, "_physics_profile_receipt", None)
                if receipt is not None:
                    receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    self._write_physics_profile_receipt()
                raise
        return self._world

    @property
    def contacts(self):
        if self._contacts is None:
            from mind2act_eval.sim.contacts import IsaacContacts
            self._contacts = IsaacContacts(self.stage)
        return self._contacts

    @property
    def artifacts(self):
        artifacts = dict(self.recorder.artifacts)
        if self._camera_receipt is not None:
            artifacts["native_camera_receipt"] = "camera-receipt.json"
        if getattr(self, "_physics_profile_receipt", None) is not None:
            artifacts["native_physics_profile_receipt"] = "physics-profile-receipt.json"
        if getattr(self, "_output_dir", None) is not None:
            artifacts["native_shutdown_receipt"] = "shutdown-receipt.json"
            if (self._output_dir / "runtime-receipt.json").is_file():
                artifacts["native_runtime_receipt"] = "runtime-receipt.json"
            if (self._output_dir / "asset-resolution-report.json").is_file():
                artifacts['asset_resolution_report'] = 'asset-resolution-report.json'
        return artifacts

    def _sample_cameras(self):
        import numpy as np
        frames, observations = {}, {}
        for name, camera in self._cameras.items():
            resolution = self._camera_config[name]["resolution"]
            expected_shape = [resolution[1], resolution[0]]
            try:
                with phases_for(self).measure('get_rgba.' + name):
                    pixels = np.asarray(camera.get_rgba())
            except Exception as exc:
                observations[name] = {"expected_shape_hw": expected_shape,
                                      "actual_shape": None, "valid": False,
                                      "read_error": f"{type(exc).__name__}: {exc}"}
                continue
            with phases_for(self).measure('rgba.validate.' + name):
                shape = list(pixels.shape)
                reasons = []
                if (pixels.ndim != 3 or pixels.shape[0] != expected_shape[0]
                        or pixels.shape[1] != expected_shape[1] or pixels.shape[2] != 4):
                    reasons.append("shape_mismatch")
                if pixels.dtype != np.dtype(np.uint8):
                    reasons.append("dtype_uint8_required")
                numeric = np.issubdtype(pixels.dtype, np.number)
                finite = bool(np.isfinite(pixels).all()) if numeric else False
                if not finite:
                    reasons.append("non_finite_values" if numeric else "numeric_dtype_required")
                finite_values = pixels[np.isfinite(pixels)] if numeric else np.asarray([], dtype=float)
                observed_min = float(finite_values.min()) if finite_values.size else None
                observed_max = float(finite_values.max()) if finite_values.size else None
                in_range = (not numeric or not finite_values.size
                            or (observed_min >= 0 and observed_max <= 255))
                if not in_range:
                    reasons.append("value_range_out_of_bounds")
                valid = not reasons and pixels.size > 0
                observations[name] = {"expected_shape_hw": expected_shape,
                                      "actual_shape": shape, "dtype": str(pixels.dtype),
                                      "finite": finite,
                                      "value_range": [observed_min, observed_max],
                                      "valid": bool(valid), "invalid_reasons": reasons}
            if valid:
                with phases_for(self).measure('rgb.copy.' + name):
                    frames[name] = pixels[..., :3].astype(np.uint8, copy=True)
        return frames, observations

    def _write_camera_receipt(self, status, renders, cameras):
        import json
        renderer_warmup = dict(getattr(self, "_renderer_warmup", {}))
        before_step = renderer_warmup.get("world_clock_before", {}).get("physics_step_index")
        after_step = renderer_warmup.get("world_clock_after", {}).get("physics_step_index")
        observed_steps = after_step - before_step if type(before_step) is int and type(after_step) is int else 0
        self._camera_receipt = {"schema": "mind2act_eval.native_camera_receipt.v1",
                                "status": status,
                                "render_only_warmup_renders": renders,
                                "physics_steps_during_warmup": observed_steps,
                                "warmup_frames_encoded": 0,
                                "initial_frame_encoded": status == "verified",
                                "initial_pose_publication": getattr(self, "_initial_pose_publication", None),
                                "renderer_warmup": renderer_warmup,
                                "bindings": self._camera_bindings(),
                                "cameras": cameras}
        path = self._output_dir / "camera-receipt.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._camera_receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def _camera_bindings(self):
        """Read actual sensor/render-product identity and transforms for audit."""
        rows = {}
        for name, camera in getattr(self, "_cameras", {}).items():
            try:
                path = str(camera.prim_path)
                product = str(camera.get_render_product_path())
                targets = [str(p) for p in self.stage.GetPrimAtPath(product).GetRelationship("camera").GetTargets()]
                position, orientation = camera.get_world_pose()
                from pxr import UsdGeom
                prim = self.stage.GetPrimAtPath(path)
                matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                position = [float(x) for x in position]
                orientation = [float(x) for x in orientation]
                matrix = [[float(x) for x in row] for row in matrix]
                if not all(isfinite(x) for values in (position, orientation, *matrix) for x in values):
                    raise ValueError("nonfinite camera pose or USD world matrix")
                rows[name] = {"status": "observed", "sensor_prim": path,
                    "requested_prim": self._camera_config[name]["path"],
                    "render_product": product, "render_camera_targets": targets,
                    "render_camera_matches_sensor": targets == [path],
                    "parent_prim": str(prim.GetParent().GetPath()),
                    "reset_xform_stack": UsdGeom.Xformable(prim).GetResetXformStack(),
                    "world_position_m": position,
                    "world_orientation_wxyz": orientation,
                    "usd_world_matrix": matrix}
            except Exception as exc:
                rows[name] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        return rows

    def rgb(self):
        if not getattr(self, "_observation_rgb_required", True):
            return {}
        initial = getattr(self, "_initial_rgb", None)
        if initial is not None:
            # Tick zero must use precisely the image already sent to the
            # recorder, even if a sensor updates between ready() and observe().
            return {name: pixels.copy() for name, pixels in initial.items()
                    if name in getattr(self, "_observation_camera_names", initial)}
        sampled = getattr(self, "_sampled_rgb", None)
        if sampled is not None:
            return {name: pixels.copy() for name, pixels in sampled.items()
                    if name in getattr(self, "_observation_camera_names", sampled)}
        result, observations = self._sample_cameras()
        unavailable = [name for name, row in observations.items() if not row["valid"]]
        if unavailable:
            raise RuntimeError("native RGB unavailable: " + ", ".join(unavailable))
        if not result and self._ready:
            raise RuntimeError("native RGB sensors not initialized")
        return {name: pixels for name, pixels in result.items()
                if name in getattr(self, "_observation_camera_names", result)}

    def prepare_observation(self, requires_rgb):
        """Refresh only images a consumer will read; never advance physics."""
        if type(requires_rgb) is not bool:
            raise TypeError("requires_rgb must be boolean")
        self._observation_rgb_required = requires_rgb
        if requires_rgb:
            self._refresh_rgb()

    def _refresh_rgb(self):
        if getattr(self, "_initial_rgb", None) is not None or getattr(self, "_sampled_rgb", None) is not None:
            return
        if not getattr(self, "_rendered_since_step", False):
            before = time.monotonic()
            with phases_for(self).measure('world.render'):
                self.world.render()
            self.timings["render_s"] += time.monotonic() - before
            self._rendered_since_step = True
        self._synchronize_capture()
        frames, observations = self._sample_cameras()
        unavailable = [name for name, row in observations.items() if not row["valid"]]
        if unavailable or (not frames and self._ready):
            raise RuntimeError("native RGB unavailable: " + ", ".join(unavailable))
        self._sampled_rgb = frames

    def capture(self):
        before = time.monotonic()
        render_before = self.timings["render_s"]
        self._refresh_rgb()
        frames = getattr(self, "_initial_rgb", None)
        self.recorder.capture(frames if frames is not None else self._sampled_rgb)
        self.timings["video_s"] += time.monotonic() - before - (self.timings["render_s"] - render_before)

    def _synchronize_capture(self):
        """Schedule and finish SDG capture at the current state, as in training.

        A zero-delta warmup leaves Replicator STEPPED. World.render() alone
        does not schedule its next capture, even when RGB shape remains valid.
        """
        import omni.replicator.core as rep
        before_clock = self._runtime_clock()
        started = time.monotonic()
        try:
            with phases_for(self).measure('rep.step'):
                rep.orchestrator.step(rt_subframes=CAPTURE_SUBFRAMES, delta_time=0.,
                                      pause_timeline=False, wait_for_render=True)
            after_clock = self._runtime_clock()
            with phases_for(self).measure('capture.audit'):
                self._record_runtime("capture", before_clock, after_clock)
            if any(after_clock[key] != before_clock[key] for key in ("time_s", "physics_step_index")):
                raise RuntimeError("render-only runtime capture advanced the world clock")
        finally:
            self.timings["render_s"] += time.monotonic() - started

    def _runtime_clock(self):
        world = self.world
        playing = getattr(world, "is_playing", None)
        return {"time_s": getattr(world, "current_time", None),
                "physics_step_index": getattr(world, "current_time_step_index", None),
                "timeline_playing": bool(playing()) if callable(playing) else None}

    def _record_runtime(self, kind, before, after):
        # Bounded readback evidence; these counters do not impose task or
        # physical-qualification thresholds and contain no private H5 labels.
        if not hasattr(self, "_runtime_receipt"):
            self._runtime_receipt = {"schema": "mind2act_eval.native_runtime.v1",
                                     "physics_calls": 0, "captures_completed": 0,
                                     "physics_step_nonunit_deltas": 0,
                                     "capture_subframes": CAPTURE_SUBFRAMES}
        receipt = self._runtime_receipt
        row = {"before": before, "after": after}
        receipt.setdefault(kind + "_first", row)
        receipt[kind + "_last"] = row
        if kind == "physics":
            receipt["physics_calls"] += 1
            first, last = before["physics_step_index"], after["physics_step_index"]
            if first is not None and last is not None and last - first != 1:
                receipt["physics_step_nonunit_deltas"] += 1
        else:
            receipt["captures_completed"] += 1
            row["render_settings"] = self._render_settings_snapshot()
            row["camera_bindings"] = self._camera_bindings()
            row["camera_clocks"] = {}
            for name, camera in getattr(self, "_cameras", {}).items():
                try:
                    frame = camera.get_current_frame()
                    stamp = float(frame["rendering_time"])
                    if not isfinite(stamp):
                        raise ValueError("nonfinite camera rendering_time")
                    row["camera_clocks"][name] = {
                        "status": "observed", "time_s": stamp,
                        "source": "Camera.get_current_frame.rendering_time"}
                except Exception as exc:
                    row["camera_clocks"][name] = {"status": "unavailable",
                                                 "error": f"{type(exc).__name__}: {exc}"}

    def _write_runtime_receipt(self):
        receipt = getattr(self, "_runtime_receipt", None)
        output = getattr(self, "_output_dir", None)
        if receipt is not None and output is not None:
            path = output / "runtime-receipt.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)

    def _render_settings_snapshot(self):
        """Read the renderer's effective values without changing capture or physics."""
        import carb
        settings = carb.settings.get_settings()
        return {name: settings.get(path) for name, path in {
            "render_mode": "/rtx/rendermode",
            "anti_aliasing_op": "/rtx/post/aa/op",
            "anti_aliasing_limited_ops": "/rtx-transient/post/aa/limitedOps",
            "motion_blur_enabled": "/rtx/post/motionblur/enabled",
        }.items()}

    def _warmup_renderer(self):
        """Complete the training capture graph/material warmup at zero physics delta."""
        import omni.replicator.core as rep
        def clock_snapshot():
            return {"time_s": getattr(self.world, "current_time", None),
                    "physics_step_index": getattr(self.world, "current_time_step_index", None)}
        before_clock = clock_snapshot()
        self._renderer_warmup = {
            "method": "training_capture_graph_zero_delta", "baseline_sha256": CAPTURE_GRAPH_BASELINE_SHA256,
            "steps_requested": CAPTURE_GRAPH_WARMUP_STEPS,
            "subframes_per_step": CAPTURE_GRAPH_WARMUP_SUBFRAMES,
            "steps_completed": 0, "subframes_completed": 0,
            "delta_time": 0., "wait_for_render": True,
            "world_clock_before": before_clock, "status": "started",
            "anti_aliasing_requested": "FXAA",
            "render_settings_before": self._render_settings_snapshot(),
        }
        started = time.monotonic()
        try:
            rep.orchestrator.set_capture_on_play(False)
            for index in range(CAPTURE_GRAPH_WARMUP_STEPS):
                with phases_for(self).measure('rep.step'):
                    rep.orchestrator.step(rt_subframes=CAPTURE_GRAPH_WARMUP_SUBFRAMES, delta_time=0.0,
                                          pause_timeline=False, wait_for_render=True)
                self._renderer_warmup["steps_completed"] += 1
                self._renderer_warmup["subframes_completed"] += CAPTURE_GRAPH_WARMUP_SUBFRAMES
                if index == 0:
                    # Match NativeRGBRecorder a5d6a772: configure FXAA after graph
                    # setup and before its completion render. The public API also
                    # disables AA limitedOps, preventing a fallback to temporal AA.
                    rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
            after_clock = clock_snapshot()
            self._renderer_warmup["world_clock_after"] = after_clock
            if after_clock != before_clock:
                raise RuntimeError("render-only capture warmup advanced the world clock")
            self._renderer_warmup["status"] = "completed"
        except Exception as exc:
            self._renderer_warmup.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            elapsed = time.monotonic() - started
            self._renderer_warmup["world_clock_after"] = clock_snapshot()
            self._renderer_warmup["render_settings_after"] = self._render_settings_snapshot()
            self._renderer_warmup["wall_s"] = elapsed
            self.timings["render_s"] += elapsed

    def ready(self):
        if not self._ready:
            gate = getattr(self, '_asset_resolution_gate', None)
            if gate is not None:
                gate.check(force=True)
            self._verify_physics_profile("ready")
            from isaacsim.sensors.camera import Camera
            for name, config in self._camera_config.items():
                if not self.stage.GetPrimAtPath(config["path"]):
                    raise ValueError("configured camera prim is missing: " + name)
                camera = Camera(prim_path=config["path"], name="eval_" + name,
                                resolution=tuple(config["resolution"]))
                camera.initialize()
                self._cameras[name] = camera
            try:
                self._publish_reset_transforms()
                self._warmup_renderer()
            except Exception:
                self._write_camera_receipt("unavailable", 0, {})
                raise
            observations = {}
            for render_count in range(1, CAMERA_WARMUP_RENDER_LIMIT + 1):
                before = time.monotonic()
                with phases_for(self).measure('world.render'):
                    self.world.render()
                self.timings["render_s"] += time.monotonic() - before
                frames, observations = self._sample_cameras()
                if len(frames) == len(self._camera_config) and all(
                        row["valid"] for row in observations.values()):
                    self._initial_rgb = {name: pixels.copy() for name, pixels in frames.items()}
                    self._write_camera_receipt("verified", render_count, observations)
                    before = time.monotonic()
                    self.recorder.capture(frames)
                    self.timings["video_s"] += time.monotonic() - before
                    self._release_gate()
                    return
            self._write_camera_receipt("unavailable", CAMERA_WARMUP_RENDER_LIMIT, observations)
            unavailable = [name for name, row in observations.items() if not row["valid"]]
            raise RuntimeError("native RGB unavailable after "
                               f"{CAMERA_WARMUP_RENDER_LIMIT} render-only warm-up renders: "
                               + ", ".join(unavailable))

    def _publish_reset_transforms(self):
        """Publish actual reset joints/fingers to visible links without simulation.

        The same forward-kinematics/readback seam is used by the PR6 and CP10
        builders. A joint teleport alone can leave the first render at the USD
        default pose until the first physics step updates its link transforms.
        """
        from omni.physx import get_physx_interface
        before = self._runtime_clock()
        self.world.physics_sim_view.update_articulations_kinematic()
        get_physx_interface().update_transformations(True, True, True, False)
        after = self._runtime_clock()
        self._initial_pose_publication = {
            "method": "update_articulations_kinematic_then_update_transformations",
            "world_clock_before": before, "world_clock_after": after,
            "status": "completed" if after == before else "failed",
        }
        if after != before:
            raise RuntimeError("initial pose transform publication advanced the world clock")

    def _release_gate(self):
        if not self._ready:
            self._ready = True
            gate, self._gate = self._gate, None
            if gate is not None:
                gate.__exit__(None, None, None)

    def step(self, *, render: bool):
        self._initial_rgb = None
        self._sampled_rgb = None
        self._rendered_since_step = False
        before = time.monotonic()
        before_clock = self._runtime_clock()
        self.world.step(render=False)
        self._record_runtime("physics", before_clock, self._runtime_clock())
        self.timings["physics_s"] += time.monotonic() - before
        if render:
            before = time.monotonic()
            with phases_for(self).measure('world.render'):
                self.world.render()
            self.timings["render_s"] += time.monotonic() - before
            self._rendered_since_step = True

    def arm_cleanup_diagnostics(self):
        from mind2act_eval.runtime_diagnostics import arm_cleanup_trace, cleanup_trace_signal_name
        path = arm_cleanup_trace(self._output_dir / "cleanup-traceback.log", timeout_s=90)
        return {"path": path, "signal_name": cleanup_trace_signal_name()}

    def _close_world_and_cameras(self, phase):
        """Release native scene wrappers in a scope that ends before Kit closes."""
        for name, resource in (("recorder", getattr(self, "recorder", None)),
                               ("contacts", getattr(self, "_contacts", None))):
            if resource is not None:
                phase(name, resource.close)
        world = getattr(self, "_world", None)
        stop = getattr(world, "stop", None)
        if callable(stop):
            phase("world.stop", stop)
        cameras = getattr(self, "_cameras", {})
        for name, camera in tuple(cameras.items()):
            pause = getattr(camera, "pause", None)
            if callable(pause):
                phase("camera.pause:" + name, pause)
            for subscription_name in ("_acquisition_callback", "_stage_open_callback",
                                      "_timer_reset_callback"):
                if hasattr(camera, subscription_name):
                    phase("camera.release_subscription:" + name + ":" + subscription_name,
                          lambda camera=camera, attr=subscription_name: setattr(camera, attr, None))
        # Detach annotators from every render product before destroying any
        # product. Replicator's SDG graph can be shared across cameras.
        for name, camera in tuple(cameras.items()):
            # Isaac Sim 5.1 Camera.destroy() only detaches custom annotators;
            # its separately attached ReferenceTime annotator must be detached first.
            reference_time = getattr(camera, "_fabric_time_annotator", None)
            detach_reference_time = getattr(reference_time, "detach", None)
            if callable(detach_reference_time):
                if phase("camera.detach_reference_time:" + name, detach_reference_time):
                    phase("camera.clear_reference_time:" + name,
                          lambda camera=camera: setattr(camera, "_fabric_time_annotator", None))
            custom_annotators = tuple(getattr(camera, "_custom_annotators", {}))
            detach_annotator = getattr(camera, "detach_annotator", None)
            for annotator_name in custom_annotators:
                if callable(detach_annotator):
                    phase("camera.detach_annotator:" + name + ":" + annotator_name,
                          lambda camera=camera, detach=detach_annotator, annotator=annotator_name:
                          detach(annotator))
        for name, camera in tuple(cameras.items()):
            destroy = getattr(camera, "destroy", None)
            if callable(destroy):
                phase("camera.destroy:" + name, destroy)
        cameras.clear()
        self._world = None
        self._contacts = None

    def close(self):
        self._os_exit_prepared = False
        self._shutdown(call_app_close=True)

    @property
    def os_exit_prepared(self):
        """True only after this instance completed its latest exit preparation."""
        return getattr(self, "_os_exit_prepared", False)

    def prepare_for_os_exit(self):
        """Flush episode resources while leaving Kit teardown to process exit."""
        self._os_exit_prepared = False
        self._shutdown(call_app_close=False)
        self._os_exit_prepared = True

    def _shutdown(self, *, call_app_close):
        errors = []
        phases = []

        def persist_shutdown_receipt(status="closing"):
            output_dir = getattr(self, "_output_dir", None)
            if output_dir is None:
                return
            import json
            receipt = {"schema": "mind2act_eval.shutdown_receipt.v1",
                       "status": "failed" if errors else status,
                       "teardown": "app_close" if call_app_close else "os_exit",
                       "phases": phases}
            path = output_dir / "shutdown-receipt.json"
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
            except Exception as exc:
                errors.append(f"shutdown receipt: {type(exc).__name__}: {exc}")

        def phase(name, operation):
            row = {"name": name, "status": "started"}
            phases.append(row)
            persist_shutdown_receipt()
            try:
                operation()
            except SystemExit as exc:
                import traceback
                if name == "app.close":
                    error = f"SystemExit from {type(self.app).__name__}.close (code={exc.code!r})"
                else:
                    error = f"SystemExit during {name} (code={exc.code!r})"
                row.update(status="failed", error=error, traceback=traceback.format_exc())
                errors.append(error)
                persist_shutdown_receipt()
                return False
            except Exception as exc:
                import traceback
                error = f"{type(exc).__name__}: {exc}"
                row.update(status="failed", error=error, traceback=traceback.format_exc())
                errors.append(f"{name}: {error}")
                persist_shutdown_receipt()
                return False
            else:
                row["status"] = "complete"
            persist_shutdown_receipt()
            return True

        try:
            asset_gate = getattr(self, '_asset_resolution_gate', None)
            if asset_gate is not None:
                phase('asset_resolution_notice_revoke', asset_gate.close)
                self._asset_resolution_gate = None
                del asset_gate
            if hasattr(self, "_runtime_receipt"):
                phase("runtime_receipt", self._write_runtime_receipt)
            self._close_world_and_cameras(phase)
            app = getattr(self, "app", None)
            if app is not None:
                if call_app_close:
                    phase("app.close", app.close)
                    self.app = None
                    app = None
                else:
                    phases.append({"name": "app.close", "status": "skipped",
                                   "reason": "isolated worker exits with os._exit"})
                    persist_shutdown_receipt()
            persist_shutdown_receipt("closed" if call_app_close else "os_exit_prepared")
        finally:
            self._release_gate()
        if errors:
            raise RuntimeError("; ".join(errors))
