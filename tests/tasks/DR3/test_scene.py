"""CPU lifecycle and gate target checks; these do not instantiate Isaac/Kit."""
from types import SimpleNamespace
import importlib
import math
from pathlib import Path
import sys
import tempfile

import pytest

from mind2act_eval.contracts import Health
from mind2act_eval.tasks.DR3.mechanism import Gate, single_target
from mind2act_eval.tasks.DR3.scene import GateScene


@pytest.mark.parametrize('time,expected', [(0, 0), (3, 0), (3.4, .17), (3.8, .34),
                                          (10.4, .34), (10.8, .17), (11.2, 0)])
def test_single_gate_preserves_source_schedule_boundaries(time, expected):
    source = dict(PERIOD=11.2, CLOSED=3., TRANS=.8, DUR=6.6, TRAVEL=.34)
    assert single_target(time, source) == pytest.approx(expected)


def test_hard_gates_preserve_common_open_window_and_closing_ramp():
    slow = Gate('Slow', -.23, 18.4, 4.8, 12.8)
    fast = Gate('Fast', -.105, 9.2, 2., 6.4)
    assert slow.target(0) == fast.target(0) == 0
    assert slow.target(5.) == fast.target(5.) == .34
    assert slow.target(8.8) == .34
    assert fast.target(8.8) == pytest.approx(.17)
    assert slow.target(18.4) == pytest.approx(0., abs=1e-12)


class CaptureFixture:
    def __init__(self):
        self.begins = []
        self.manifest = {'tray_colliders': ['tray']}
        self.position = [0., 0., 0.]

    def begin_step(self, tick):
        self.begins.append(tick)

    def snapshot(self, contacts, tick, time):
        return dict(poses={'item': dict(position=list(self.position), quaternion_wxyz=[1., 0., 0., 0.])},
                    contacts=[], gates={'single': 0.})

    def sampling_receipt(self):
        return {'before_step_ticks': self.begins}


def make_scene(tmp_path=None):
    spec = SimpleNamespace(instance_id='one-source-condition', difficulty='easy',
                           protocol_version='v1', clock=SimpleNamespace(physics_hz=120),
                           config={'source_case': {}, 'phase_ticks': 6,
                                   'physics_budget': 1000,
                                   'condition': {'instance_id': 'one-source-condition'},
                                   'instruction': 'public instruction'})
    world = SimpleNamespace(current_time=12.)
    output_dir = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp(prefix='dr3-scene-'))
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(world=world, output_dir=output_dir)
    capture = CaptureFixture()
    contacts = SimpleNamespace(rows={}, errors=[])
    closed = []
    subscription = SimpleNamespace(unsubscribe=lambda: closed.append(True))
    return GateScene(spec, ctx, None, {}, {}, contacts, subscription, capture), closed


def test_task_tick_one_drives_selected_phase_without_stepping_the_simulator(monkeypatch, tmp_path):
    scene, _ = make_scene(tmp_path)
    calls = []
    monkeypatch.setattr('mind2act_eval.tasks.DR3.scene._drive', lambda *args: calls.append(args[-1]))
    scene.before_step(1)
    assert scene.capture.begins == [0]
    assert calls == [6 / 120]
    assert scene.ctx.world.current_time == 12.
    scene.ctx.world.current_time += 1 / 120
    scene.after_step(1)
    evidence = scene.evidence(1)
    assert evidence.values['raw_trace']['rows'] == 2
    from mind2act_eval.tasks.DR3.audit import read_manifest, read_trace
    assert [row['physics_tick'] for row in read_trace(evidence.values['raw_trace'], 1)] == [0, 1]
    assert read_manifest(evidence.values['manifest']) == scene.capture.manifest
    assert [row['physics_tick'] for row in scene.rows] == [0, 1]
    assert scene.health() == Health(True)
    assert not scene.is_complete()


def test_any_evidence_tick_contains_complete_append_only_physics_receipt(monkeypatch, tmp_path):
    scene, _ = make_scene(tmp_path)
    monkeypatch.setattr('mind2act_eval.tasks.DR3.scene._drive', lambda *args: None)
    scene.before_step(1)
    scene.ctx.world.current_time += 1 / 120
    scene.after_step(1)
    evidence = scene.evidence(1)
    assert evidence.values['raw_trace']['rows'] == evidence.tick + 1
    assert evidence.values['raw_trace']['bytes'] > 0
    assert evidence.values['raw_trace']['sha256']


def test_completion_probe_never_replays_or_stops_the_model(monkeypatch, tmp_path):
    scene, _ = make_scene(tmp_path)
    monkeypatch.setattr('mind2act_eval.tasks.DR3.events.rebuild_events',
                        lambda *args: pytest.fail('completion probe must not score'))
    assert not scene.is_complete()
    assert not scene.is_complete()


def test_missing_or_reordered_physical_samples_are_not_fabricated(tmp_path):
    scene, _ = make_scene(tmp_path)
    with pytest.raises(ValueError, match='before-step'):
        scene.before_step(2)
    with pytest.raises(ValueError, match='sampled exactly once'):
        scene.after_step(2)
    with pytest.raises(ValueError, match='before the physical sample'):
        scene.evidence(1)


def test_health_reports_real_nonfinite_state_and_contact_sensor_failure(tmp_path):
    scene, _ = make_scene(tmp_path)
    scene.rows[-1]['detector']['poses']['item']['position'][0] = math.nan
    assert scene.health() == Health(False, 'DR3 measured pose is not finite')
    scene.contacts.errors.append('invalid support normal')
    assert not scene.health().ok and 'contact measurement failed' in scene.health().reason


def test_close_releases_only_the_task_contact_subscription_once(tmp_path):
    scene, closed = make_scene(tmp_path)
    scene.close()
    scene.close()
    assert closed == [True]


def test_scene_import_does_not_start_native_runtime():
    importlib.import_module('mind2act_eval.tasks.DR3.scene')
    assert 'isaacsim' not in sys.modules
    assert 'omni.kit' not in sys.modules


def test_sleeping_contact_keeps_its_owned_support_normal_until_lost(monkeypatch):
    import pxr
    from mind2act_eval.tasks.DR3.scene import ContactSamples
    monkeypatch.setattr(pxr, 'PhysicsSchemaTools', SimpleNamespace(intToSdfPath=str), raising=False)
    monkeypatch.setitem(sys.modules, 'omni.physx.bindings._physx', SimpleNamespace(
        ContactEventType=SimpleNamespace(CONTACT_LOST=3)))
    samples = ContactSamples()
    item, tray = '/World/Task/Item/Collider', '/World/Task/Tray/Collider'
    header = SimpleNamespace(collider0=item, collider1=tray, contact_data_offset=0,
                             num_contact_data=1, type=1)
    samples.receive([header], [SimpleNamespace(impulse=[0,0,.01], normal=[0,0,1])])
    pair = tuple(sorted((item,tray)))
    header.num_contact_data = 0; header.type = 2
    samples.receive([header], [])
    assert samples.rows[pair]['active']
    assert samples.rows[pair]['item_upward_normals'] == [1.]
    assert samples.rows[pair]['impulse_ns'] == 0.
    header.type = 3
    samples.receive([header], [])
    assert not samples.rows[pair]['active']
