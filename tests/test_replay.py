from dataclasses import replace
import json
import numpy as np
import h5py
import pytest
from mind2act_eval.contracts import ClockSpec, InstanceSpec, Observation, to_json
from mind2act_eval.replay import H5Replay, ReplayMetadata


def training_spec():
    return InstanceSpec("demo", "train-001", "easy", "public-training", "v1", "none", "v1",
                        clock=ClockSpec(120, 120, 30))


def write_h5(path):
    with h5py.File(path, "w") as stream:
        for side in ("left", "right"):
            stream.create_dataset("action/" + side + "_arm_joint_states", data=[[180., 0., 0., 0., 0., 0.], [90., 0., 0., 0., 0., 0.]])
            stream.create_dataset("action/" + side + "_ee_joint_states", data=[[1.], [0.]])


def metadata():
    return ReplayMetadata("episode-001", "public-training", 120, 120, 30, "deg", "normalized", 0., 1., instance_spec=to_json(training_spec()))


def test_recorded_120hz_is_not_replayed_at_30hz_and_units_polarity_are_explicit(tmp_path):
    path = tmp_path / "trajectory.h5"
    write_h5(path)
    replay = H5Replay(path, metadata())
    try:
        replay.assert_spec(training_spec())
        with pytest.raises(ValueError, match="clock"):
            replay.assert_spec(replace(training_spec(), clock=ClockSpec(120, 30, 30)))
        first = replay.act(Observation(0, 0., (0.,)*6, (0.,)*6, 1., 1.))
        assert first.left_arm[0] == pytest.approx(np.pi)
        assert first.left_gripper == 0.
        second = replay.act(Observation(1, 1/120, (0.,)*6, (0.,)*6, 1., 1.))
        assert second.left_arm[0] == pytest.approx(np.pi / 2)
        assert second.left_gripper == 1.
        with pytest.raises(StopIteration):
            replay.act(Observation(2, 2/120, (0.,)*6, (0.,)*6, 1., 1.))
    finally:
        replay.close()


def test_replay_cannot_bind_to_another_difficulty_with_same_source_group(tmp_path):
    path = tmp_path / "trajectory.h5"
    write_h5(path)
    replay = H5Replay(path, metadata())
    try:
        with pytest.raises(ValueError, match="binding"):
            replay.assert_spec(replace(training_spec(), difficulty="hard"))
    finally:
        replay.close()


def test_replay_applies_explicit_source_to_x5_gripper_target_mapping(tmp_path):
    path = tmp_path / "trajectory.h5"
    write_h5(path)
    with h5py.File(path, "r+") as stream:
        for side in ("left", "right"):
            stream[f"action/{side}_ee_joint_states"][:] = [[0.], [1.]]
    source_metadata = ReplayMetadata(
        "episode-001", "public-training", 120, 120, 30, "deg", "normalized", 1., 0.,
        gripper_action_scale=.044 / .054,
        gripper_action_offset=.01 / .054,
        instance_spec=to_json(training_spec()),
    )
    replay = H5Replay(path, source_metadata)
    try:
        first = replay.act(Observation(0, 0., (0.,)*6, (0.,)*6, 1., 1.))
        second = replay.act(Observation(1, 1/120, (0.,)*6, (0.,)*6, 1., 1.))
        # The H5 normalized openings (0, 1) represent physical targets (0, .044 m).
        # They must be preserved through X5's target q=-.01+.054*u mapping.
        assert first.left_gripper == pytest.approx(.01 / .054)
        assert second.left_gripper == pytest.approx((.044 + .01) / .054)
        assert replay.receipt()["metadata"]["gripper_action_offset"] == pytest.approx(.01 / .054)
        assert "explicit affine" in replay.receipt()["conversion"]
    finally:
        replay.close()


def test_replay_rejects_changed_h5_bytes_before_action_execution(tmp_path):
    import hashlib
    path = tmp_path / "trajectory.h5"
    write_h5(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path, "r+") as stream:
        stream["action/left_arm_joint_states"][0, 0] = 0.
    with pytest.raises(ValueError, match="digest"):
        H5Replay(path, metadata(), expected_sha256=expected)
