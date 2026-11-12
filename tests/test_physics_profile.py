"""CPU lifecycle/identity checks, not native FEM execution."""
from dataclasses import replace
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from mind2act_eval.contracts import ClockSpec, InstanceSpec
from mind2act_eval.sim import IsaacSimulator
from mind2act_eval.tasks.PR4.source_admission import validate_case_source


# Selected source identity only; these fixtures do not pretend to be buildable scenes.
SOURCES = (
    ("easy", 410000, "91bae0d07468717fc99bb1e8a3a45da502a808b2b648973f0ea559e9c8a070b1",
     "61832f768989aa1a7e7a0a9fa108b49622ce04e087fd54c03717f1e499e6c918",
     "26a07213df4820654fc58dc32a77396db3907470b11ea65793013480f106145f"),
    ("medium", 420000, "e8697f4bdf7d589f0eaeedb336bec736ec9b40259c88fe5f9b4d930cf7e9ce30",
     "421ee81960cd04313990a4fb3ae83672f7acea2b54aa8de7eb54656052f840e4",
     "bb832184c9b81261a465081ef6799f508623daa9234b70e755cd5c04c60542b1"),
    ("hard", 430000, "67d453d2d34081d0d8e97f233a1becd751c8ddfd6c18b81e17cfb16c5236adeb",
     "bbf2dc9a6b601d38ed50094eecea5c0421cca838d37f35fe813a796dc7476e97",
     "8da390c0d3b70f452b79d8aa55c5c15617e5e4482cef72af563942d4b4c9e7f3"),
)


def identity_spec(row=SOURCES[0]):
    difficulty, seed, source, config, scene = row
    return InstanceSpec("PR4", f"PR4-{difficulty}-train-seed{seed}", difficulty,
                        "pr4-original-time-v3", "height-study-raised20mm-seeded-translation-v1",
                        "test-assets", "v1", seed=seed, clock=ClockSpec(120, 30, 30),
                        config={"source_spec_sha256": source, "source_config_sha256": config,
                                "scene_sha256": scene})


def simulator(tmp_path):
    sim = object.__new__(IsaacSimulator)
    sim._world = None
    sim._clock = ClockSpec(120, 30, 30)
    sim._output_dir = tmp_path
    sim.recorder = SimpleNamespace(artifacts={})
    sim._camera_receipt = None
    return sim


def install_world(monkeypatch, *, ignore_gpu=False, broadphase="GPU"):
    calls = []

    class Physics:
        def __init__(self):
            self.gpu = False
            self.broadphase = "MBP"
        def enable_gpu_dynamics(self, flag):
            calls.append(("gpu", flag))
            if not ignore_gpu:
                self.gpu = flag
        def set_broadphase_type(self, value):
            calls.append(("broadphase", value))
            self.broadphase = broadphase
        def is_gpu_dynamics_enabled(self):
            calls.append("read_gpu")
            return self.gpu
        def get_broadphase_type(self):
            calls.append("read_broadphase")
            return self.broadphase
        def get_current_physics_scene_prim(self):
            return SimpleNamespace(GetPath=lambda: "/physicsScene")

    class World:
        def __init__(self, **kwargs):
            calls.append(("world", kwargs))
            self.physics = Physics()
        def get_physics_context(self):
            return self.physics
        def initialize_physics(self):
            calls.append("initialize")
            assert self.physics.gpu and self.physics.broadphase == "GPU"
        def reset(self):
            raise AssertionError('profile configuration must not reset physics')
        def step(self, **kwargs):
            raise AssertionError('profile configuration must not advance physics')

    module = ModuleType("isaacsim.core.api")
    module.World = World
    monkeypatch.setitem(sys.modules, "isaacsim.core.api", module)
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(IsaacSimulator, 'stage', property(lambda self: stage))
    return calls


@pytest.mark.parametrize("row", SOURCES)
def test_selected_source_configures_and_reads_back_before_physics_initialization(tmp_path, monkeypatch, row):
    calls = install_world(monkeypatch)
    sim, spec = simulator(tmp_path), identity_spec(row)
    sim.bind_task_instance(spec, tmp_path)
    assert sim.active_task_spec is spec
    assert not calls
    world = sim.world
    world.initialize_physics()
    assert calls == [
        ("world", {"physics_dt": 1/120, "rendering_dt": 1/30, "stage_units_in_meters": 1.0}),
        ("gpu", True), ("broadphase", "GPU"), "read_gpu", "read_broadphase", "initialize",
    ]
    receipt = json.loads((tmp_path / "physics-profile-receipt.json").read_text())
    assert receipt["status"] == "configured"
    assert receipt["readbacks"][0]["gpu_dynamics"] is True
    assert receipt["readbacks"][0]["broadphase"] == "GPU"
    assert receipt["source_spec_sha256"] == row[2]
    assert len(receipt["input_sha256"]) == 64
    assert sim.artifacts["native_physics_profile_receipt"] == "physics-profile-receipt.json"
    assert sim.world is world  # No repeated writes on ordinary World access.
    assert len(calls) == 6


@pytest.mark.parametrize("changes", [
    {"source_id": "other-source"}, {"config_version": "other-version"},
    {"instance_id": "unselected"}, {"seed": 410001}, {"difficulty": "medium"},
    {"clock": ClockSpec(240, 30, 30)}, {"protocol_version": "v2"},
])
def test_mismatched_pr4_binding_is_rejected_before_world(tmp_path, monkeypatch, changes):
    calls = install_world(monkeypatch)
    with pytest.raises(ValueError, match="PR4.*binding"):
        validate_case_source(replace(identity_spec(), **changes))
    assert calls == []


@pytest.mark.parametrize("field", ["source_spec_sha256", "source_config_sha256", "scene_sha256"])
def test_declared_source_hashes_are_pinned_not_just_hex_strings(tmp_path, field):
    spec = identity_spec()
    config = dict(spec.config, **{field: "0" * 64})
    with pytest.raises(ValueError, match="PR4.*binding"):
        validate_case_source(replace(spec, config=config))


def test_dict_or_task_id_impostor_does_not_select_profile(tmp_path):
    for invalid in ("PR4", {"task_id": "PR4"}, SimpleNamespace(task_id="PR4")):
        with pytest.raises(TypeError, match="InstanceSpec"):
            simulator(tmp_path).bind_task_instance(invalid, tmp_path)


@pytest.mark.parametrize("bound", [False, True])
def test_other_tasks_and_unbound_world_keep_original_kwargs_and_no_physics_writes(tmp_path, monkeypatch, bound):
    calls = install_world(monkeypatch)
    sim = simulator(tmp_path)
    if bound:
        sim.bind_task_instance(replace(identity_spec(), task_id="CP04"), tmp_path)
    sim.world
    assert calls == [("world", {"physics_dt": 1/120, "rendering_dt": 1/30, "stage_units_in_meters": 1.0})]
    assert not (tmp_path / "physics-profile-receipt.json").exists()


def test_rebinding_equal_copy_late_binding_and_output_redirection_are_rejected(tmp_path, monkeypatch):
    install_world(monkeypatch)
    sim, spec = simulator(tmp_path), identity_spec()
    sim.bind_task_instance(spec, tmp_path)
    sim.bind_task_instance(spec, tmp_path)  # Exactly the same frozen object is idempotent.
    with pytest.raises(ValueError, match="replaced"):
        sim.bind_task_instance(replace(spec), tmp_path)
    with pytest.raises(ValueError, match="output"):
        sim.bind_task_instance(spec, tmp_path / "other")
    with pytest.raises(AttributeError):
        sim.active_task_spec = spec
    late = simulator(tmp_path)
    late.world
    with pytest.raises(RuntimeError, match="before.*World"):
        late.bind_task_instance(spec, tmp_path)


@pytest.mark.parametrize("kwargs", [{"ignore_gpu": True}, {"broadphase": "MBP"}])
def test_readback_mismatch_fails_and_cached_world_cannot_bypass_failure(tmp_path, monkeypatch, kwargs):
    install_world(monkeypatch, **kwargs)
    sim = simulator(tmp_path)
    sim.bind_task_instance(identity_spec(), tmp_path)
    with pytest.raises(RuntimeError, match="readback"):
        sim.world
    receipt = json.loads((tmp_path / "physics-profile-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["readbacks"][0]["phase"] == "before_physics_initialization"
    with pytest.raises(RuntimeError, match="previously failed"):
        sim.world


def test_runtime_readback_detects_changes_without_resetting_them(tmp_path, monkeypatch):
    calls = install_world(monkeypatch)
    sim = simulator(tmp_path)
    sim.bind_task_instance(identity_spec(), tmp_path)
    world = sim.world
    world.physics.gpu = False
    sim._ready = False
    with pytest.raises(RuntimeError, match="readback"):
        sim.ready()  # The real hook fails before importing/creating any cameras.
    assert not world.physics.gpu
    assert calls.count(("gpu", True)) == 1
    assert json.loads((tmp_path / "physics-profile-receipt.json").read_text())["status"] == "failed"


def test_getter_failure_is_persisted_and_never_treated_as_requested_success(tmp_path, monkeypatch):
    install_world(monkeypatch)
    sim = simulator(tmp_path)
    sim.bind_task_instance(identity_spec(), tmp_path)
    world = sim.world
    def unavailable():
        raise RuntimeError("SDK getter unavailable")
    world.physics.is_gpu_dynamics_enabled = unavailable
    with pytest.raises(RuntimeError, match="SDK getter unavailable"):
        sim._verify_physics_profile("ready")
    receipt = json.loads((tmp_path / "physics-profile-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert "SDK getter unavailable" in receipt["error"]
    with pytest.raises(RuntimeError, match="previously failed"):
        sim.world


def test_simulator_clock_conflict_is_rejected_before_request(tmp_path):
    sim = simulator(tmp_path)
    sim._clock = ClockSpec(960, 30, 30)
    with pytest.raises(ValueError, match="clock binding"):
        sim.bind_task_instance(identity_spec(), tmp_path)
    assert not (tmp_path / "physics-profile-receipt.json").exists()


@pytest.mark.parametrize('row',SOURCES)
def test_profile_configuration_is_independent_of_case_identity(row):
    from mind2act_eval.sim.physics_profile import profile_for_instance
    spec=identity_spec(row);baseline=profile_for_instance(spec).receipt()
    for key,value in (('instance_id','another-case'),('seed',row[1]+3),('source_id','another-source')):
        changed=replace(spec,**{key:value})
        actual=profile_for_instance(changed).receipt()
        assert actual['configuration_sha256']==baseline['configuration_sha256']
        assert actual['requested']==baseline['requested']
        with pytest.raises(ValueError):validate_case_source(changed)


def test_physics_configuration_asset_is_hash_bound(tmp_path,monkeypatch):
    from mind2act_eval.sim import physics_profile
    (tmp_path/'pr4_physics_profiles.json').write_text('{}')
    monkeypatch.setattr(physics_profile.resources,'files',lambda _:tmp_path)
    with pytest.raises(ValueError,match='configuration asset SHA'):
        physics_profile.profile_for_instance(identity_spec())


def test_case_hash_provenance_does_not_change_physics_configuration():
    from mind2act_eval.sim.physics_profile import profile_for_instance
    spec=identity_spec();expected=profile_for_instance(spec).receipt()
    for field in ('source_spec_sha256','source_config_sha256','scene_sha256'):
        changed=replace(spec,config={**spec.config,field:'0'*64})
        actual=profile_for_instance(changed).receipt()
        assert actual['configuration_sha256']==expected['configuration_sha256']
        assert actual['requested']==expected['requested']
        with pytest.raises(ValueError):validate_case_source(changed)
