import json
from types import SimpleNamespace

import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from mind2act_eval.tasks.PR2.diagnostic import ContactProbe, contact_point, usd_readback  # noqa: E402
from mind2act_eval.tasks.PR2 import mechanism  # noqa: E402


def point(**over):
    return SimpleNamespace(**dict(dict(position=(0, 0, 1), normal=(0, 0, 2),
        impulse=(3, 4, .25), separation=-.0001), **over))


def test_measured_normal_force_excludes_tangential_impulse_and_uses_dt():
    item = contact_point(point(), .005)
    assert item['normal_impulse_signed_Ns'] == .25
    assert item['normal_force_magnitude_N'] == 50
    assert item['impulse_Ns'] == [3, 4, .25]
    assert contact_point(point(), .01)['normal_force_magnitude_N'] == 25
    assert contact_point(point(normal=(0, 0, -1)), .005)['normal_impulse_signed_Ns'] == -.25


@pytest.mark.parametrize('normal', [None, (0, 0, 0), (float('nan'), 0, 1)])
def test_missing_or_unusable_normal_never_becomes_force(normal):
    assert contact_point(point(normal=normal), .005)['normal_force_magnitude_N'] is None


def stage_with_knob():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    knob = UsdGeom.Xform.Define(stage, mechanism.KNOB)
    knob.AddTranslateOp().Set((0, .1, .9))
    UsdPhysics.RigidBodyAPI.Apply(knob.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdGeom.Cylinder.Define(stage, mechanism.KNOB + '/Grip')
    return stage


def test_raw_pair_tick_force_and_motion_are_private_and_read_only(tmp_path):
    stage = stage_with_knob()
    before = stage.GetRootLayer().ExportToString()
    callbacks = []
    probe = ContactProbe(stage, tmp_path / 'private', lambda: .01,
        subscribe=lambda cb: callbacks.append(cb), decode_path=str)
    probe.before_step(17)
    headers = [SimpleNamespace(collider0=mechanism.KNOB + '/Grip',
        collider1='/World/RobotLeft/Asset/link7/Tip', contact_data_offset=0, num_contact_data=1, type='PERSISTS')]
    callbacks[0](headers, [point()])
    probe.after_step(17)
    probe.close()
    row = json.loads((tmp_path / 'private/contacts.jsonl').read_text())
    assert row['tick_before_step'] == 17 and row['dt_s'] == .01
    assert row['pair'] == [headers[0].collider0, headers[0].collider1]
    assert row['points'][0]['normal_force_magnitude_N'] == 25
    motion = json.loads((tmp_path / 'private/motion.jsonl').read_text())
    assert motion['tick'] == 17 and mechanism.KNOB in motion['world_transforms']
    assert (tmp_path / 'private/before-first-step.json').exists()
    assert (tmp_path / 'private/after-last-step.json').exists()
    assert before == stage.GetRootLayer().ExportToString()


def test_empty_contact_data_retains_event_and_does_not_invent_force(tmp_path):
    probe = ContactProbe(stage_with_knob(), tmp_path, lambda: .01, subscribe=lambda cb: cb, decode_path=str)
    probe.before_step(18)
    probe.receive([SimpleNamespace(collider0=mechanism.KNOB + '/Grip',
        collider1=mechanism.ROOT + '/Shaft', contact_data_offset=0, num_contact_data=0, type='LOST')], [])
    probe.close()
    row = json.loads((tmp_path / 'contacts.jsonl').read_text())
    assert row['points'] == []
    assert row['event_type'] == 'LOST'


def test_readback_distinguishes_authored_value_from_usd_default():
    stage = stage_with_knob()
    row = usd_readback(stage)['prims'][mechanism.KNOB]
    assert row['attributes']['physics:rigidBodyEnabled']['authored'] is True
    assert row['attributes']['physics:kinematicEnabled']['authored'] is False
    assert row['attributes']['physics:kinematicEnabled']['value'] is False
