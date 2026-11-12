import pytest
from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim.contacts import ContactLedger


@pytest.fixture
def renderer_settings(monkeypatch):
    """SDK readback seam; initial AA values deliberately differ from training."""
    import sys
    from types import SimpleNamespace
    values = {"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 3,
              "/rtx-transient/post/aa/limitedOps": True,
              "/rtx/post/motionblur/enabled": False}
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(
        settings=SimpleNamespace(get_settings=lambda: SimpleNamespace(get=values.get))))
    return values


def test_asset_resolver_rejects_symlink_escape(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    hidden = tmp_path / "private.txt"
    hidden.write_text("outside configured asset root")
    (root / "escape").symlink_to(hidden)
    assets = LocalAssets(root, {"station": "escape"})
    with pytest.raises(ValueError, match="escapes"):
        assets.resolve("station")


def test_contacts_require_both_fingers_of_one_arm_and_clear_on_contact_lost():
    contacts = ContactLedger()
    root = "/World/Tray"
    contacts.record(root + "/rim", "/World/RobotLeft/Asset/link7/collision", active=True, impulse_ns=.01, separation_m=0.)
    assert contacts.touch(root, robot_only=True)
    assert not contacts.bilateral(root)
    contacts.record(root + "/rim", "/World/RobotRight/Asset/link8/collision", active=True, impulse_ns=.01, separation_m=0.)
    assert not contacts.bilateral(root)
    contacts.record(root + "/rim", "/World/RobotLeft/Asset/link8/collision", active=True, impulse_ns=.01, separation_m=0.)
    assert contacts.bilateral(root)
    contacts.record(root + "/rim", "/World/RobotLeft/Asset/link7/collision", active=False, impulse_ns=0., separation_m=1.)
    assert not contacts.bilateral(root)


def test_recorded_video_decodes_as_actual_rgb_frames(tmp_path):
    import shutil
    import subprocess
    import numpy as np
    from mind2act_eval.sim.video import RGBRecorder
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("explicit local ffmpeg required")
    recorder = RGBRecorder(tmp_path, 30, ffmpeg=ffmpeg)
    green = np.zeros((32, 32, 3), dtype=np.uint8)
    green[..., 1] = 255
    for _ in range(4):
        recorder.capture({"head": green})
    recorder.close()
    decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(tmp_path / "head.mp4"),
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                             check=True, capture_output=True).stdout
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 32, 32, 3)
    assert len(frames) == 4
    assert frames[..., 1].mean() > 240
    assert frames[..., 0].mean() < 10
    assert recorder.artifacts == {"video_head": "head.mp4"}


def test_encoder_close_failure_does_not_skip_native_cleanup():
    from mind2act_eval.sim import IsaacSimulator
    calls = []
    class BrokenRecorder:
        def close(self):
            calls.append("recorder")
            raise RuntimeError("encoder failed")
    class Resource:
        def __init__(self, name):
            self.name = name
        def close(self):
            calls.append(self.name)
    sim = object.__new__(IsaacSimulator)
    sim.recorder = BrokenRecorder()
    sim._contacts = Resource("contacts")
    sim.app = Resource("app")
    sim._release_gate = lambda: calls.append("release_gate")
    with pytest.raises(RuntimeError, match="encoder failed"):
        sim.close()
    assert calls == ["recorder", "contacts", "app", "release_gate"]


def test_close_stops_world_and_destroys_camera_products_before_app(tmp_path):
    import json
    from mind2act_eval.sim import IsaacSimulator

    calls = []
    class Resource:
        def __init__(self, name):
            self.name = name
        def close(self):
            calls.append(self.name)
    class World:
        def stop(self):
            calls.append("world.stop")
    class Camera:
        def destroy(self):
            calls.append("camera.destroy")

    sim = object.__new__(IsaacSimulator)
    sim.recorder = Resource("recorder")
    sim._contacts = Resource("contacts")
    sim._world = World()
    sim._cameras = {"head": Camera()}
    sim.app = Resource("app")
    sim._output_dir = tmp_path
    sim._release_gate = lambda: calls.append("release_gate")

    sim.close()

    assert calls == ["recorder", "contacts", "world.stop", "camera.destroy", "app", "release_gate"]
    receipt = json.loads((tmp_path / "shutdown-receipt.json").read_text())
    assert receipt["status"] == "closed"
    assert [phase["name"] for phase in receipt["phases"]] == [
        "recorder", "contacts", "world.stop", "camera.destroy:head", "app.close"
    ]
    assert all(phase["status"] == "complete" for phase in receipt["phases"])


def test_prepare_for_os_exit_flushes_resources_without_calling_app_close(tmp_path):
    import json
    from mind2act_eval.sim import IsaacSimulator

    class App:
        def close(self):
            raise AssertionError("sealed-worker teardown must not call app.close")

    sim = object.__new__(IsaacSimulator)
    sim.recorder = sim._contacts = sim._world = None
    sim._cameras = {}
    sim.app = App()
    sim._output_dir = tmp_path
    sim._release_gate = lambda: None

    sim.prepare_for_os_exit()

    receipt = json.loads((tmp_path / "shutdown-receipt.json").read_text())
    assert receipt["status"] == "os_exit_prepared"
    assert receipt["teardown"] == "os_exit"
    assert receipt["phases"][-1]["name"] == "app.close"
    assert receipt["phases"][-1]["status"] == "skipped"
    assert sim.app is not None


def test_close_records_app_shutdown_failure_after_ordered_camera_cleanup(tmp_path):
    import json
    from mind2act_eval.sim import IsaacSimulator

    calls = []
    class App:
        def close(self):
            calls.append("app.close")
            raise AttributeError("NoneType has no attribute get_stage_state")
    class Camera:
        def destroy(self):
            calls.append("camera.destroy")

    sim = object.__new__(IsaacSimulator)
    sim.recorder = None
    sim._contacts = None
    sim._world = None
    sim._cameras = {"head": Camera()}
    sim.app = App()
    sim._output_dir = tmp_path
    sim._release_gate = lambda: calls.append("release_gate")

    with pytest.raises(RuntimeError, match="get_stage_state"):
        sim.close()

    assert calls == ["camera.destroy", "app.close", "release_gate"]
    receipt = json.loads((tmp_path / "shutdown-receipt.json").read_text())
    assert receipt["status"] == "failed"
    phase = receipt["phases"][-1]
    assert phase["name"] == "app.close"
    assert phase["status"] == "failed"
    assert phase["error"] == "AttributeError: NoneType has no attribute get_stage_state"
    assert "Traceback (most recent call last)" in phase["traceback"]
    assert "in close" in phase["traceback"]


def test_camera_cleanup_detaches_isaac_internal_annotator_and_event_subscriptions(tmp_path):
    import json
    from mind2act_eval.sim import IsaacSimulator

    calls = []
    class ReferenceTime:
        def detach(self):
            calls.append("reference_time.detach")
    class Camera:
        def __init__(self):
            self._acquisition_callback = object()
            self._stage_open_callback = object()
            self._timer_reset_callback = object()
            self._fabric_time_annotator = ReferenceTime()
        def pause(self):
            calls.append("camera.pause")
            self._acquisition_callback = None
        def destroy(self):
            assert self._acquisition_callback is None
            assert self._stage_open_callback is None
            assert self._timer_reset_callback is None
            assert self._fabric_time_annotator is None
            calls.append("camera.destroy")
    camera = Camera()

    sim = object.__new__(IsaacSimulator)
    sim.recorder = None
    sim._contacts = None
    sim._world = None
    sim._cameras = {"left_wrist": camera}
    sim.app = None
    sim._output_dir = tmp_path
    sim._release_gate = lambda: calls.append("release_gate")

    sim.close()

    assert calls == ["camera.pause", "reference_time.detach", "camera.destroy", "release_gate"]
    receipt = json.loads((tmp_path / "shutdown-receipt.json").read_text())
    assert [phase["name"] for phase in receipt["phases"]] == [
        "camera.pause:left_wrist",
        "camera.release_subscription:left_wrist:_acquisition_callback",
        "camera.release_subscription:left_wrist:_stage_open_callback",
        "camera.release_subscription:left_wrist:_timer_reset_callback",
        "camera.detach_reference_time:left_wrist",
        "camera.clear_reference_time:left_wrist",
        "camera.destroy:left_wrist",
    ]
    assert receipt["status"] == "closed"


def test_close_detaches_all_camera_annotators_before_destroying_any_render_product(tmp_path):
    from mind2act_eval.sim import IsaacSimulator

    calls = []
    products_destroyed = False
    detached = {"head": False, "left_wrist": False, "right_wrist": False}

    class Annotator:
        def __init__(self, name):
            self.name = name
        def detach(self):
            assert not products_destroyed
            detached[self.name] = True
            calls.append("detach:" + self.name)

    class Product:
        def __init__(self, name):
            self.name = name
        def destroy(self):
            nonlocal products_destroyed
            assert all(detached.values())
            calls.append("product:" + self.name)
            products_destroyed = True

    class Camera:
        def __init__(self, name):
            self._custom_annotators = {"rgb": Annotator(name)}
            self._render_product = Product(name)
        def pause(self): pass
        def detach_annotator(self, name):
            self._custom_annotators.pop(name).detach()
        def destroy(self):
            for name in list(self._custom_annotators):
                self.detach_annotator(name)
            self._render_product.destroy()

    sim = object.__new__(IsaacSimulator)
    sim.recorder = None
    sim._contacts = None
    sim._world = None
    sim._cameras = {name: Camera(name) for name in detached}
    sim.app = None
    sim._output_dir = tmp_path
    sim._release_gate = lambda: None

    sim.close()

    assert calls == [
        "detach:head", "detach:left_wrist", "detach:right_wrist",
        "product:head", "product:left_wrist", "product:right_wrist",
    ]


def test_kit_app_closes_after_camera_and_world_python_refs_are_released(tmp_path):
    import weakref
    from mind2act_eval.sim import IsaacSimulator

    events = []

    class NativeLike:
        def __del__(self):
            events.append("resource.finalized")

    class Product:
        def destroy(self): pass

    class Camera:
        def __init__(self):
            self._acquisition_callback = object()
            self._stage_open_callback = object()
            self._timer_reset_callback = object()
            self._fabric_time_annotator = NativeLike()
            self._custom_annotators = {"rgb": NativeLike()}
            self._render_product = Product()
        def pause(self): pass
        def detach_annotator(self, name):
            self._custom_annotators.pop(name)
        def destroy(self):
            self._render_product.destroy()
            self._render_product = None

    class World:
        def stop(self): pass
        def __del__(self):
            events.append("world.finalized")

    def make_sim():
        camera = Camera()
        world = World()
        references = {
            "camera": weakref.ref(camera),
            "world": weakref.ref(world),
            "reference_time": weakref.ref(camera._fabric_time_annotator),
            "rgb_annotator": weakref.ref(camera._custom_annotators["rgb"]),
            "render_product": weakref.ref(camera._render_product),
        }
        class App:
            def close(self):
                assert all(ref() is None for ref in references.values())
                events.append("app.close")
        app = App()
        app_reference = weakref.ref(app)
        sim = object.__new__(IsaacSimulator)
        sim.recorder = None
        sim._contacts = None
        sim._world = world
        sim._cameras = {"left_wrist": camera}
        sim.app = app
        sim._output_dir = tmp_path
        def release_gate():
            assert app_reference() is None
            events.append("gate.release")
        sim._release_gate = release_gate
        return sim, references, app_reference

    sim, references, app_reference = make_sim()
    sim.close()

    assert all(ref() is None for ref in references.values())
    assert app_reference() is None
    assert events[-2:] == ["app.close", "gate.release"]


def test_ready_uses_bounded_render_only_warmup_and_records_all_camera_shapes(tmp_path, monkeypatch):
    import json
    import numpy as np
    import sys
    from types import ModuleType, SimpleNamespace
    from mind2act_eval import sim as sim_module
    monkeypatch.setattr(sim_module.IsaacSimulator, "_warmup_renderer", lambda self: None)
    monkeypatch.setattr(sim_module.IsaacSimulator, "_publish_reset_transforms", lambda self: None)

    cameras = {}
    config = {
        name: {"path": "/World/" + name, "resolution": [4, 3]}
        for name in ("head", "left_wrist", "right_wrist")
    }
    class Camera:
        def __init__(self, *, prim_path, name, resolution):
            self.key = prim_path.rsplit("/", 1)[-1]
            self.reads = 0
            cameras[self.key] = self
        def initialize(self):
            pass
        def get_rgba(self):
            self.reads += 1
            if self.reads < 3:
                return np.empty((0, 0, 4), dtype=np.uint8)
            return np.full((3, 4, 4), len(self.key), dtype=np.uint8)

    sensors = ModuleType("isaacsim.sensors")
    sensors.__path__ = []
    camera_module = ModuleType("isaacsim.sensors.camera")
    camera_module.Camera = Camera
    monkeypatch.setitem(sys.modules, "isaacsim.sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaacsim.sensors.camera", camera_module)

    events = []
    class World:
        def render(self):
            events.append("render")
        def step(self, **kwargs):
            events.append("step")
            pytest.fail("camera warm-up must not advance physics")
    class Recorder:
        artifacts = {}
        def capture(self, frames):
            events.append(("capture", tuple(sorted(frames))))
            assert all(frame.shape == (3, 4, 3) for frame in frames.values())

    stage = SimpleNamespace(GetPrimAtPath=lambda path: object())
    monkeypatch.setattr(sim_module.IsaacSimulator, "stage", property(lambda self: stage))
    monkeypatch.setattr(sim_module.IsaacSimulator, "world", property(lambda self: World()))
    sim = object.__new__(sim_module.IsaacSimulator)
    sim._ready = False
    sim._camera_config = config
    sim._cameras = {}
    sim._output_dir = tmp_path
    sim._camera_receipt = None
    sim.timings = {"render_s": 0., "video_s": 0.}
    sim.recorder = Recorder()
    sim._release_gate = lambda: events.append("release_gate")

    sim.ready()

    assert events == ["render", "render", "render",
                      ("capture", ("head", "left_wrist", "right_wrist")), "release_gate"]
    assert sim.artifacts["native_camera_receipt"] == "camera-receipt.json"
    receipt = json.loads((tmp_path / "camera-receipt.json").read_text())
    assert receipt["status"] == "verified"
    assert receipt["render_only_warmup_renders"] == 3
    assert receipt["physics_steps_during_warmup"] == 0
    assert receipt["warmup_frames_encoded"] == 0
    assert receipt["initial_frame_encoded"] is True
    assert {name: value["actual_shape"] for name, value in receipt["cameras"].items()} == {
        name: [3, 4, 4] for name in config
    }


def test_ready_stops_after_bounded_warmup_without_encoding_unavailable_rgb(tmp_path, monkeypatch):
    import json
    import numpy as np
    import sys
    from types import ModuleType, SimpleNamespace
    from mind2act_eval import sim as sim_module
    monkeypatch.setattr(sim_module.IsaacSimulator, "_warmup_renderer", lambda self: None)
    monkeypatch.setattr(sim_module.IsaacSimulator, "_publish_reset_transforms", lambda self: None)

    class Camera:
        def __init__(self, **kwargs):
            pass
        def initialize(self):
            pass
        def get_rgba(self):
            return np.empty((0, 0, 4), dtype=np.uint8)
    sensors = ModuleType("isaacsim.sensors")
    sensors.__path__ = []
    camera_module = ModuleType("isaacsim.sensors.camera")
    camera_module.Camera = Camera
    monkeypatch.setitem(sys.modules, "isaacsim.sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaacsim.sensors.camera", camera_module)

    renders = []
    class World:
        def render(self):
            renders.append("render")
        def step(self, **kwargs):
            pytest.fail("camera warm-up must not advance physics")
    class Recorder:
        artifacts = {}
        def capture(self, frames):
            pytest.fail("unavailable RGB must never be encoded")

    stage = SimpleNamespace(GetPrimAtPath=lambda path: object())
    monkeypatch.setattr(sim_module.IsaacSimulator, "stage", property(lambda self: stage))
    monkeypatch.setattr(sim_module.IsaacSimulator, "world", property(lambda self: World()))
    sim = object.__new__(sim_module.IsaacSimulator)
    sim._ready = False
    sim._camera_config = {"head": {"path": "/World/Head", "resolution": [4, 3]}}
    sim._cameras = {}
    sim._output_dir = tmp_path
    sim._camera_receipt = None
    sim.timings = {"render_s": 0., "video_s": 0.}
    sim.recorder = Recorder()
    sim._release_gate = lambda: None

    with pytest.raises(RuntimeError, match="native RGB unavailable after 8 render-only warm-up renders: head"):
        sim.ready()

    assert renders == ["render"] * 8
    receipt = json.loads((tmp_path / "camera-receipt.json").read_text())
    assert receipt["status"] == "unavailable"
    assert receipt["render_only_warmup_renders"] == 8
    assert receipt["physics_steps_during_warmup"] == 0
    assert receipt["initial_frame_encoded"] is False
    assert receipt["cameras"]["head"]["actual_shape"] == [0, 0, 4]


@pytest.mark.parametrize(("kind", "expected_reason"), [
    ("nan", "non_finite_values"),
    ("inf", "non_finite_values"),
    ("wrong_dtype", "dtype_uint8_required"),
    ("out_of_range", "value_range_out_of_bounds"),
])
def test_native_rgba_validation_rejects_invalid_pixel_payloads(kind, expected_reason):
    import numpy as np
    from types import SimpleNamespace
    from mind2act_eval.sim import IsaacSimulator

    pixels = {
        "nan": np.full((2, 3, 4), np.nan, dtype=np.float32),
        "inf": np.full((2, 3, 4), np.inf, dtype=np.float32),
        "wrong_dtype": np.full((2, 3, 4), .5, dtype=np.float32),
        "out_of_range": np.full((2, 3, 4), 256, dtype=np.int16),
    }[kind]
    sim = object.__new__(IsaacSimulator)
    sim._camera_config = {"head": {"resolution": [3, 2]}}
    sim._cameras = {"head": SimpleNamespace(get_rgba=lambda: pixels)}

    frames, observations = sim._sample_cameras()

    assert frames == {}
    assert observations["head"]["valid"] is False
    assert expected_reason in observations["head"]["invalid_reasons"]


def test_native_rgba_validation_accepts_a_real_uint8_black_frame():
    import numpy as np
    from types import SimpleNamespace
    from mind2act_eval.sim import IsaacSimulator

    black = np.zeros((2, 3, 4), dtype=np.uint8)
    sim = object.__new__(IsaacSimulator)
    sim._camera_config = {"head": {"resolution": [3, 2]}}
    sim._cameras = {"head": SimpleNamespace(get_rgba=lambda: black)}

    frames, observations = sim._sample_cameras()

    assert observations["head"]["valid"] is True
    assert observations["head"]["value_range"] == [0., 0.]
    assert frames["head"].dtype == np.uint8
    assert np.count_nonzero(frames["head"]) == 0


def test_ready_completes_training_zero_delta_warmup_before_encoding_shape_valid_cold_rgb(tmp_path, monkeypatch, renderer_settings):
    import json
    import sys
    import numpy as np
    from types import ModuleType, SimpleNamespace
    from mind2act_eval import sim as sim_module
    monkeypatch.setattr(sim_module.IsaacSimulator, "_publish_reset_transforms", lambda self: None)
    events = []
    def warm_step(**kwargs):
        assert kwargs == dict(rt_subframes=8, delta_time=0.0, pause_timeline=False, wait_for_render=True)
        events.append("warm_step")
    rep = ModuleType("omni.replicator.core")
    rep.orchestrator = SimpleNamespace(set_capture_on_play=lambda enabled: events.append(("auto_capture", enabled)), step=warm_step)
    def set_realtime(*, antialiasing):
        assert events == [("auto_capture", False), "warm_step"]
        assert antialiasing == "FXAA"
        renderer_settings.update({"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 2,
                                  "/rtx-transient/post/aa/limitedOps": False})
        events.append("FXAA")
    rep.settings = SimpleNamespace(set_render_rtx_realtime=set_realtime)
    omni = ModuleType("omni"); omni.__path__ = []
    replicator = ModuleType("omni.replicator"); replicator.__path__ = []
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", rep)
    class Camera:
        def __init__(self, **kwargs): pass
        def initialize(self): pass
        def get_rgba(self):
            # Both cold and loaded images satisfy shape/dtype/range validation.
            value = 9 if events.count("warm_step") == 2 else 0
            return np.full((3, 4, 4), value, dtype=np.uint8)
    sensors = ModuleType("isaacsim.sensors"); sensors.__path__ = []
    camera_module = ModuleType("isaacsim.sensors.camera"); camera_module.Camera = Camera
    monkeypatch.setitem(sys.modules, "isaacsim.sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaacsim.sensors.camera", camera_module)
    class World:
        current_time = 3.5
        current_time_step_index = 420
        def render(self): events.append("render")
        def step(self, **kwargs): pytest.fail("warmup cannot step physics")
    class Recorder:
        artifacts = {}
        def capture(self, frames):
            assert events.count("warm_step") == 2
            assert all((frame == 9).all() for frame in frames.values())
            events.append("capture")
    world = World()
    monkeypatch.setattr(sim_module.IsaacSimulator, "stage", property(lambda self: SimpleNamespace(GetPrimAtPath=lambda path: object())))
    monkeypatch.setattr(sim_module.IsaacSimulator, "world", property(lambda self: world))
    sim = object.__new__(sim_module.IsaacSimulator)
    sim._ready = False
    sim._camera_config = {name: {"path": "/World/" + name, "resolution": [4, 3]} for name in ("head", "left_wrist", "right_wrist")}
    sim._cameras = {}; sim._output_dir = tmp_path; sim._camera_receipt = None
    sim.timings = {"render_s": 0., "video_s": 0.}
    sim.recorder = Recorder(); sim._release_gate = lambda: None
    sim.ready()
    assert events == [("auto_capture", False), "warm_step", "FXAA", "warm_step", "render", "capture"]
    receipt = json.loads((tmp_path / "camera-receipt.json").read_text())
    assert receipt["renderer_warmup"]["render_settings_before"]["anti_aliasing_op"] == 3
    assert receipt["renderer_warmup"]["render_settings_after"]["anti_aliasing_op"] == 2
    assert receipt["renderer_warmup"]["render_settings_after"]["anti_aliasing_limited_ops"] is False
    assert receipt["renderer_warmup"]["subframes_completed"] == 16
    assert receipt["renderer_warmup"]["world_clock_before"] == receipt["renderer_warmup"]["world_clock_after"]
    assert receipt["warmup_frames_encoded"] == 0


def test_zero_delta_warmup_rejects_world_clock_advance(tmp_path, monkeypatch, renderer_settings):
    import sys
    from types import ModuleType, SimpleNamespace
    from mind2act_eval.sim import IsaacSimulator
    world = SimpleNamespace(current_time=0., current_time_step_index=0)
    rep = ModuleType("omni.replicator.core")
    def step(**kwargs):
        world.current_time += 1/120
        world.current_time_step_index += 1
    rep.orchestrator = SimpleNamespace(set_capture_on_play=lambda _: None, step=step)
    rep.settings = SimpleNamespace(set_render_rtx_realtime=lambda **_: None)
    for name in ("omni", "omni.replicator"):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", rep)
    monkeypatch.setattr(IsaacSimulator, "world", property(lambda self: world))
    sim = object.__new__(IsaacSimulator)
    sim.timings = {"render_s": 0.}
    with pytest.raises(RuntimeError, match="advanced the world clock"):
        sim._warmup_renderer()
    assert sim._renderer_warmup["status"] == "failed"
    assert sim._renderer_warmup["world_clock_after"]["physics_step_index"] == 2


def test_runtime_capture_finishes_graph_at_current_physics_state(tmp_path, monkeypatch, renderer_settings):
    """A completed zero-delta warmup does not schedule later SDG captures."""
    import sys
    import numpy as np
    from types import ModuleType, SimpleNamespace
    from mind2act_eval.sim import IsaacSimulator
    state = {"physical": 0, "rendered": 0}
    class World:
        current_time = 0.
        current_time_step_index = 0
        def step(self, *, render):
            assert render is False
            state["physical"] += 1
            self.current_time_step_index += 1
            self.current_time = self.current_time_step_index / 120
        def render(self):
            # Replicator STEPPED: app updates alone do not schedule a new capture.
            pass
    world = World()
    def finish_capture(**kwargs):
        assert kwargs == dict(rt_subframes=2, delta_time=0., pause_timeline=False, wait_for_render=True)
        state["rendered"] = state["physical"]
    rep = ModuleType("omni.replicator.core")
    rep.orchestrator = SimpleNamespace(step=finish_capture)
    for name in ("omni", "omni.replicator"):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", rep)
    monkeypatch.setattr(IsaacSimulator, "world", property(lambda self: world))
    frames = []
    class Recorder:
        def capture(self, rgb): frames.append(rgb["head"].copy())
    sim = object.__new__(IsaacSimulator)
    sim.timings = {"physics_s": 0., "render_s": 0., "video_s": 0.}
    sim.recorder = Recorder()
    sim._sample_cameras = lambda: ({"head": np.full((2, 2, 3), state["rendered"], dtype=np.uint8)},
                                   {"head": {"valid": True}})
    for _ in range(2):
        for tick in range(4): sim.step(render=tick == 3)
        clock = (world.current_time, world.current_time_step_index)
        sim.capture()
        assert (world.current_time, world.current_time_step_index) == clock
    assert [int(frame[0, 0, 0]) for frame in frames] == [4, 8]
    receipt = sim._runtime_receipt
    assert receipt["physics_calls"] == 8
    assert receipt["captures_completed"] == 2
    assert receipt["capture_first"]["render_settings"]["anti_aliasing_op"] == 3
    assert receipt["capture_last"]["render_settings"]["anti_aliasing_limited_ops"] is True
    assert receipt["physics_step_nonunit_deltas"] == 0
    assert receipt["physics_first"]["before"]["physics_step_index"] == 0
    assert receipt["physics_last"]["after"]["physics_step_index"] == 8
    assert receipt["capture_last"]["before"] == receipt["capture_last"]["after"]


def test_runtime_capture_rejects_physics_advance_without_encoding(monkeypatch, renderer_settings):
    import sys
    from types import ModuleType, SimpleNamespace
    from mind2act_eval.sim import IsaacSimulator
    world = SimpleNamespace(current_time=0., current_time_step_index=0, render=lambda: None)
    def advance(**kwargs):
        world.current_time = 1/120
        world.current_time_step_index = 1
    rep = ModuleType("omni.replicator.core")
    rep.orchestrator = SimpleNamespace(step=advance)
    for name in ("omni", "omni.replicator"):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", rep)
    monkeypatch.setattr(IsaacSimulator, "world", property(lambda self: world))
    sim = object.__new__(IsaacSimulator)
    sim.timings = {"render_s": 0., "video_s": 0.}
    sim.recorder = SimpleNamespace(capture=lambda _: pytest.fail("cannot encode physics-advancing render"))
    with pytest.raises(RuntimeError, match="runtime capture advanced the world clock"):
        sim.capture()


def test_target_support_uses_directed_normal_and_does_not_count_self_or_gripper():
    contacts = ContactLedger()
    item, tray = '/World/Tray/Item', '/World/Tray'
    contacts.record(item+'/Face', tray+'/Floor', active=True, impulse_ns=0., separation_m=0.,
                    normals_on_collider0=[(0., 0., 1.)])
    assert contacts.supported(item, other=tray)
    contacts.record(item+'/Face', tray+'/Floor', active=False, impulse_ns=0., separation_m=1.)
    contacts.record(item+'/Face', item+'/Handle', active=True, impulse_ns=.01, separation_m=0.,
                    normals_on_collider0=[(0., 0., 1.)])
    assert not contacts.supported(item, other=tray)
    contacts.record('/World/RobotLeft/finger', item+'/Face', active=True, impulse_ns=.01, separation_m=0.,
                    normals_on_collider0=[(0., 0., -1.)])
    assert not contacts.supported(item)


def test_target_support_handles_reversed_colliders_and_rejects_side_or_proximity():
    contacts = ContactLedger()
    root, target = '/World/Item', '/World/Pan'
    contacts.record(target+'/Floor', root+'/Face', active=True, impulse_ns=.001, separation_m=0.,
                    normals_on_collider0=[(0., 0., -1.)])
    assert contacts.supported(root, other=target)
    contacts.record(target+'/Floor', root+'/Face', active=True, impulse_ns=.001, separation_m=0.,
                    normals_on_collider0=[(1., 0., 0.)])
    assert not contacts.supported(root, other=target)
    contacts.record(target+'/Floor', root+'/Face', active=True, impulse_ns=0., separation_m=.001,
                    normals_on_collider0=[(0., 0., -1.)])
    assert not contacts.supported(root, other=target)


def test_missing_support_normals_are_infra_evidence_not_a_model_failure():
    contacts = ContactLedger()
    contacts.record('/World/Item/Face', '/World/Pan/Floor', active=True, impulse_ns=.01, separation_m=0.)
    with pytest.raises(ValueError, match='support normal'):
        contacts.supported('/World/Item', other='/World/Pan')
