import pytest

from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.catalog import load_task
from mind2act_eval.tasks.PR5.scorer import RULE_VERSION, score
from .test_evidence import measured, raw_frames, replace_frames


@pytest.mark.parametrize('options,expected', [({}, 100.0), ({'force': 4.99}, 0.0), ({'smear': 0.0031}, 0.0), ({'yaw': 3.0}, 11), ({'yaw': 180.0}, 11), ({'home': False}, 22), ({'clear': False}, 100.0), ({'bad': True}, 100.0), ({'yaw': 2.0}, 100.0)])
def test_highest_eligible_stage_from_physical_evidence(options, expected, tmp_path):
    result = score(measured(tmp_path, **options).evidence(76))
    assert result.value == expected
    assert result.components['task_success'] is (expected == 100.)
    assert result.components['rule_version'] == RULE_VERSION
    assert result.components['stage'] == expected


def test_zero_priority_outside_box_cannot_collect_heading_or_return_credit(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    values = dict(frames=raw_frames(evidence))
    for frame in values['frames'][:15]:
        frame['position'][0] += .1
    result = score(replace_frames(evidence, values['frames'], tmp_path))
    assert result.value == 0.


def test_multiple_clean_impressions_are_zero_even_after_a_correct_return(tmp_path):
    rec = measured(tmp_path)
    for frame in raw_frames(rec.evidence(76)):
        tick = rec.frame_count + 1
        rec.record(tick, frame['position'], frame['orientation_wxyz'], frame['tcps'], frame['contacts'])
    assert len(rec.facts()['imprints']) == 2
    assert score(rec.evidence(152)).value == 0.


def test_full_credit_uses_final_pose_without_half_second_wait(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    values = dict(frames=raw_frames(evidence))
    values['frames'] = values['frames'][:74]  # 59 final home samples after the physical imprint
    assert score(replace_frames(evidence, values['frames'], tmp_path, tick=74)).value == 100.


def test_wrong_return_orientation_retains_only_heading_correct_imprint_credit(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    values = dict(frames=raw_frames(evidence))
    values['frames'][-1]['orientation_wxyz'] = [0., 0., 0., 1.]
    assert score(replace_frames(evidence, values['frames'], tmp_path)).value == 22


def test_none_or_missing_physics_is_an_infrastructure_error_not_a_valid_zero():
    with pytest.raises(ValueError, match='native evidence'):
        score(None)
    with pytest.raises(ValueError, match='native evidence'):
        score(NativeEvidence('PR5', 'synthetic', 0, {}))


def test_catalog_loads_one_shared_builder_and_scorer():
    task = load_task('PR5')
    assert task.task_id == 'PR5' and task.scorer is score
    assert task.build.__module__ == 'mind2act_eval.tasks.PR5.scene'


@pytest.mark.parametrize('held', [False, True])
def test_final_pose_allows_held_or_released_stamp_without_withdrawal(held, tmp_path):
    from .test_evidence import contact, FINGER
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)[:16]  # first and only returned sample
    frames[-1]['tcps'] = {'Left': [0., 0., .77], 'Right': [0., 0., .77]}
    if held:
        frames[-1]['contacts'].append(contact('/World/Task/StampA/Handle', FINGER, [0., .01, 0.]))
    assert score(replace_frames(evidence, frames, tmp_path, tick=16)).value == 100.


@pytest.mark.parametrize('axis,offset,expected', [(0, 0.00399, 100.0), (0, 0.00401, 22), (2, 0.00199, 100.0), (2, 0.00201, 22)])
def test_final_return_translation_thresholds(axis, offset, expected, tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    frames[-1]['position'][axis] += offset
    assert score(replace_frames(evidence, frames, tmp_path)).value == expected


@pytest.mark.parametrize('axis', [0, 2])
@pytest.mark.parametrize('angle,expected', [(2.99, 100.0), (3.01, 22)])
def test_final_return_tilt_and_yaw_thresholds(axis, angle, expected, tmp_path):
    import math
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    q = [math.cos(math.radians(angle)/2), 0., 0., 0.]
    q[axis+1] = math.sin(math.radians(angle)/2)
    frames[-1]['orientation_wxyz'] = q
    assert score(replace_frames(evidence, frames, tmp_path)).value == expected


def test_return_is_measured_at_episode_end_not_latched_at_earlier_home(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    frames[-1]['position'] = [.02, .025, .8]
    assert score(replace_frames(evidence, frames, tmp_path)).value == 22


def test_return_requires_actual_holder_support_not_hovering_at_home(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    for frame in frames[15:]:
        frame['contacts'] = []
    assert score(replace_frames(evidence, frames, tmp_path)).value == 22


@pytest.mark.parametrize('normal', [0., -1.])
def test_side_or_downward_holder_contact_is_not_support(normal, tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    frames[-1]['contacts'][0]['support_normal_z'] = normal
    assert score(replace_frames(evidence, frames, tmp_path)).value == 22


def test_missing_holder_contact_ownership_is_infra(tmp_path):
    evidence = measured(tmp_path).evidence(76)
    frames = raw_frames(evidence)
    del frames[-1]['contacts'][0]['support_normal_z']
    with pytest.raises(ValueError, match='support normal'):
        score(replace_frames(evidence, frames, tmp_path))
