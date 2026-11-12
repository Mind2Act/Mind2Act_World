import copy
import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.PR5.evidence import Recorder, reconstruct, rotation
from mind2act_eval.tasks.PR5.mechanism.contacts import ContactReadings
from mind2act_eval.tasks.PR5.mechanism.imprint import Frame, ImprintTracker


def case():
    return dict(id='medium/train/synthetic', level='medium', split='train', index=0,
        design='single-square-head-arrow-tight-box-20261022',
        stamps=[dict(id='A', card_xy=[0., .025], holder_xy=[.02, -.2], box_wh=[.08, .08],
                     target_yaw=0., yaw_tol=2.)])


def contact(a, b, impulse):
    return dict(pair=sorted((a, b)), impulse_ns=impulse)


FACE = '/World/Task/StampA/Face'
CARD = '/World/Task/CardA'
FINGER = '/World/RobotRight/Asset/link7/col'


def measured(tmp_path, *, force=6., smear=0., yaw=0., home=True, clear=True, bad=False):
    rec = Recorder('PR5', 'synthetic', case(), trace_path=tmp_path/'raw-trace.jsonl')
    q = [math.cos(math.radians(yaw)/2), 0., 0., math.sin(math.radians(yaw)/2)]
    for tick in range(1, 16):
        x = .02 + (smear if tick > 7 else 0.)
        contacts = [contact(FACE, CARD, [0., 0., force/120]),
                    contact('/World/Task/StampA/Handle', FINGER, [0., .01, 0.])]
        if bad:
            contacts.append(contact('/World/RobotLeft/Asset/link3/col', '/World/Task/Board', [.01, 0., 0.]))
        rec.record(tick, [x, .025, .7768], q, {'Left': [0., 0., 1.], 'Right': [0., 0., 1.]}, contacts)
    for tick in range(16, 77):
        pos = [.02, -.2, .772] if home else [.02, .025, .8]
        rec.record(tick, pos, [1., 0., 0., 0.],
            {'Left': [0., 0., 1.], 'Right': [0., 0., 1. if clear else .8]},
            [dict(contact('/World/Task/StampA/Face', '/World/Task/HolderA/Base', [0., 0., .01]), support_normal_z=1.)] if home else [])
    return rec



def raw_frames(evidence):
    return [json.loads(line) for line in Path(evidence.values['raw_trace']['path']).read_bytes().splitlines()]


def replace_frames(evidence, frames, tmp_path, *, tick=None):
    values = to_json(evidence.values)
    payload = b''.join((json.dumps(frame, separators=(',', ':'))+'\n').encode() for frame in frames)
    path = (tmp_path/'altered-trace.jsonl').resolve()
    path.write_bytes(payload)
    when = evidence.tick if tick is None else tick
    values['raw_trace'] = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), frames=when)
    return NativeEvidence('PR5', 'synthetic', when, values)


def test_physical_imprint_and_home_roundtrip(tmp_path):
    rec = measured(tmp_path)
    facts = reconstruct(rec.evidence(76))
    assert facts['success'] and facts['single'] and facts['inside'] and facts['heading']
    assert facts['clean'] and facts['stamp_home']


@pytest.mark.parametrize('kwargs,failed', [
    ({'force': 4.99}, 'single'), ({'smear': .0031}, 'clean'), ({'yaw': 2.01}, 'heading'),
    ({'home': False}, 'stamp_home')])
def test_physical_failure_is_not_success(kwargs, failed, tmp_path):
    facts = measured(tmp_path, **kwargs).facts()
    assert not facts['success'] and not facts[failed]


def test_unrelated_contact_does_not_veto_valid_imprint_and_return(tmp_path):
    facts = measured(tmp_path, bad=True).facts()
    assert facts['single'] and facts['clean'] and facts['stamp_home']
    assert facts['success']


def test_contact_force_uses_sum_vector_not_largest_contact_norm():
    readings = ContactReadings()
    readings.record([FACE, CARD], [[0., 0., .03], [0., 0., .02]])
    assert readings.snapshot()[0]['impulse_ns'] == pytest.approx([0., 0., .05])
    readings.record([FACE, CARD], [[0., 0., .04], [0., 0., -.04]])
    assert readings.snapshot()[0]['impulse_ns'] == pytest.approx([0., 0., 0.])
    readings.clear()
    assert readings.snapshot() == []


def test_imprint_force_duration_tilt_and_smear_boundaries_keep_source_semantics():
    tracker = ImprintTracker()
    tracker.update(Frame(0., 5., 0., 0., 0., 10.))
    tracker.update(Frame(.1, 5., .003, 0., 5., 10.))
    marks = tracker.finish()
    assert len(marks) == 1 and not marks[0].smeared
    short = ImprintTracker()
    short.update(Frame(0., 5., 0., 0., 0., 0.))
    short.update(Frame(.099999, 5., 0., 0., 0., 0.))
    assert short.finish() == []


def test_exact_heading_tolerance_passes_and_return_orientation_is_independent(tmp_path):
    facts = measured(tmp_path, yaw=2.).facts()
    assert facts['heading'] and facts['stamp_home'] and facts['success']


def test_frame_inside_test_uses_all_rotated_face_corners():
    from mind2act_eval.tasks.PR5.mechanism.imprint import Imprint, judge
    def assess(x):
        return judge([Imprint(x, 0., 0., .1, False)], (.02, 0.), (.08, .08), .05, 0., 2., True, True)
    assert assess(.034999999)['inside']
    assert not assess(.035001)['inside']


def test_copied_imprint_and_home_functions_match_selected_source_bytes():
    import hashlib
    from importlib import resources
    root = resources.files('mind2act_eval.tasks.PR5')
    assert hashlib.sha256(root.joinpath('mechanism/imprint.py').read_bytes()).hexdigest() == '008306e7ac4839781d0206464dd5d0322f7ddf6bf2f088111a151a45e586b250'
    assert hashlib.sha256(root.joinpath('mechanism/grading.py').read_bytes()).hexdigest() == 'f0f36325f8a12ec4f11297a69177bddcc688f9bac97a026be49dc59a2f970a25'


@pytest.mark.parametrize('change', ['missing_tick', 'duplicate_tick', 'phase', 'nan', 'quaternion', 'pair'])
def test_corrupt_or_nonphysical_evidence_is_rejected(change, tmp_path):
    evidence = measured(tmp_path).evidence(76)
    data = dict(frames=raw_frames(evidence))
    if change == 'missing_tick':
        data['frames'].pop(5)
    elif change == 'duplicate_tick':
        data['frames'][4]['tick'] = 4
    elif change == 'phase':
        data['frames'][0]['phase'] = 'finished'
    elif change == 'nan':
        data['frames'][0]['position'][0] = float('nan')
    elif change == 'quaternion':
        data['frames'][0]['orientation_wxyz'] = [0., 0., 0., 0.]
    else:
        data['frames'][0]['contacts'].append(copy.deepcopy(data['frames'][0]['contacts'][0]))
    with pytest.raises(ValueError):
        reconstruct(replace_frames(evidence, data['frames'], tmp_path))


def test_rotation_matches_source_wxyz_convention():
    q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    assert rotation(q) @ np.array([1., 0., 0.]) == pytest.approx([0., 1., 0.])


def test_native_float32_poses_become_finite_json_evidence(tmp_path):
    rec = Recorder('PR5', 'synthetic', case(), trace_path=tmp_path/'raw-trace.jsonl')
    rec.record(1, np.asarray([.02, -.2, .772], dtype=np.float32),
               np.asarray([1., 0., 0., 0.], dtype=np.float32),
               {'Left': np.asarray([0., 0., 1.], dtype=np.float32),
                'Right': np.asarray([0., 0., 1.], dtype=np.float32)}, [])
    assert raw_frames(rec.evidence(1))[0]['orientation_wxyz'] == [1., 0., 0., 0.]


def test_both_arms_may_remain_low(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    for frame in frames[15:]:
        frame['tcps']['Left'][2] = .77
    facts = reconstruct(replace_frames(evidence, frames, tmp_path))
    assert facts['success']
    frames[-1]['tcps']['Right'][2] = .80
    assert reconstruct(replace_frames(evidence, frames, tmp_path))['success']


def test_final_stamp_contact_does_not_block_return(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    frames[-1]['contacts'].append(contact('/World/Task/StampA/Handle',
        '/World/RobotLeft/Asset/link7/col', [0., .01, 0.]))
    assert reconstruct(replace_frames(evidence, frames, tmp_path))['success']


def test_both_arms_can_operate_and_remain_low(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    frames[0]['contacts'].append(contact('/World/Task/StampA/Handle',
        '/World/RobotLeft/Asset/link7/col', [0., .01, 0.]))
    frames[-1]['tcps']['Left'][2] = .77
    facts = reconstruct(replace_frames(evidence, frames, tmp_path))
    assert facts['success']


def test_sleeping_holder_support_persists_until_contact_loss_without_reusing_impulse():
    readings = ContactReadings()
    pair = [FACE, '/World/Task/HolderA/Base']
    readings.record(pair, [[0., 0., .01]], support_normal_z=1.)
    readings.clear()
    assert readings.snapshot() == [dict(pair=sorted(pair), impulse_ns=[0., 0., 0.], support_normal_z=1.)]
    readings.lost(pair)
    assert readings.snapshot() == []
