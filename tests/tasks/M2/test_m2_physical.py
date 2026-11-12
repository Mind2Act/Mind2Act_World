"""Public synthetic CPU traces through the task's real mechanism/scene/scorer."""
from copy import deepcopy
import json
from math import cos, sin, radians
from types import SimpleNamespace as NS

import pytest

from mind2act_eval.contracts import InstanceSpec, NativeEvidence, to_json
from mind2act_eval.tasks.M2.binding import ProfileBlocked, load, sha256, validate
from mind2act_eval.tasks.M2.scene import MemoryScene, build
from mind2act_eval.tasks.M2.scorer import PAIRS, score


def quaternion(degrees):
    return [cos(radians(degrees)/2), 0., sin(radians(degrees)/2), 0.]


class Trace:
    def __init__(self, difficulty="easy"):
        self.spec = InstanceSpec("M2", "public-synthetic", difficulty, "public-synthetic", "1", "1", "v1",
                                 seed=7, config={"m2_mode": "diagnostic_training"})
        n = 2 * PAIRS[difficulty]
        self.binding = dict(schema="m2.source-binding.v1", identity={k: getattr(self.spec, k) for k in
            ("task_id", "instance_id", "difficulty", "source_id", "seed", "asset_version")},
            mode="diagnostic_training", source_manifest_sha256="a"*64, scene_sha256="b"*64,
            mechanism_sha256="2d5a516d615e32beafb1fa839b62dd2ff722fae0fe5b54cf2747ced782dc8e04",
            card_identities=[f"synthetic-secret-{i//2}" for i in range(n)],
            faceup_quaternions_wxyz=[quaternion(2)]*n, button_rest_m=[1.]*n)
        validate(self.spec, self.binding)
        self.state = dict(button_depths_m=[0.]*n, finger_contacts=[False]*n,
                          card_quaternions_wxyz=[quaternion(178) for _ in range(n)], card_hinge_angles_deg=[178.]*n)
        self.frozen = set()
        self.commands = []
        def drive(desired, matched):
            self.commands.append(tuple(desired))
            # Deterministic synthetic servo substitute, not a physics claim.
            for i, angle in enumerate(desired):
                if i not in self.frozen:
                    self.state["card_quaternions_wxyz"][i] = quaternion(angle)
                    self.state["card_hinge_angles_deg"][i] = angle
        sensors = NS(read=lambda: deepcopy(self.state), set_card_targets=drive)
        robot = NS(observe=lambda: dict(left_arm=[0]*6, right_arm=[0]*6, left_gripper=1., right_gripper=1.))
        self.scene = MemoryScene(self.spec, NS(robot=robot), self.binding, sensors)
        self.tick = 0

    def step(self, n=1):
        for _ in range(n):
            self.tick += 1
            self.scene.before_step(self.tick)
            self.scene.after_step(self.tick)

    def press(self, i, n=4, release=True, depth=.0011, contact=True):
        self.state["button_depths_m"][i] = depth
        self.state["finger_contacts"][i] = contact
        self.step(n)
        if release:
            self.state["button_depths_m"][i] = 0.
            self.state["finger_contacts"][i] = False
            self.step()

    def pair(self, index):
        self.press(index*2)
        self.step(84)
        self.press(index*2+1)
        self.step(108)

    def result(self):
        return score(self.scene.evidence(self.tick))


@pytest.mark.parametrize("difficulty", PAIRS)
def test_autonomous_physical_press_release_to_full_score_and_private_observation(difficulty):
    t = Trace(difficulty)
    assert not hasattr(t.scene.game, "enabled")
    for i in range(PAIRS[difficulty]):t.pair(i)
    assert t.result().value == 100
    assert t.scene.is_complete()
    assert t.result().components["completed_pairs"] == PAIRS[difficulty]
    public = json.dumps(to_json(t.scene.observe(t.tick)))
    assert "synthetic-secret" not in public
    assert "card_identities" not in public
    assert "pairs" not in to_json(t.scene.observe(t.tick))
    assert t.scene.result_metadata()["formal_evaluation_profile"] == "blocked"


@pytest.mark.parametrize("difficulty,pairs,expected", [('easy', 1, 15.0), ('medium', 1, 5.5), ('medium', 2, 11.0), ('hard', 3, 5.625), ('hard', 4, 7.5)])
def test_highest_stage_half_pair_boundaries(difficulty, pairs, expected):
    t = Trace(difficulty)
    for i in range(pairs):t.pair(i)
    assert t.result().value == expected


@pytest.mark.parametrize("difficulty", PAIRS)
@pytest.mark.parametrize("failure", ["angle", "release", "termination"])
def test_all_pairs_but_strict_settling_or_release_incomplete(difficulty, failure):
    t = Trace(difficulty)
    for i in range(PAIRS[difficulty]):t.pair(i)
    v = to_json(t.scene.evidence(t.tick).values)
    if failure == "angle":v["card_angle_errors_deg"][0] = 2.
    if failure == "release":v["all_buttons_released"] = False
    if failure == "termination":v["normal_termination"] = False
    result = score(NativeEvidence("M2", t.spec.instance_id, t.tick, v))
    assert result.value == ({'easy':30,'medium':22,'hard':15}[difficulty] if failure == 'angle' else 100)


@pytest.mark.parametrize("n,depth,contact", [(0, .0011, True), (4, .001, True)])
def test_short_shallow_or_contactless_press_never_selects(n, depth, contact):
    t = Trace()
    t.press(0, n=n, depth=depth, contact=contact)
    assert not t.scene.game.open
    assert t.result().value == 0


def test_button_hold_requires_release_and_selects_if_ready_at_release():
    t = Trace()
    t.press(0, release=False)
    assert not t.scene.game.open
    t.state["button_depths_m"][0] = .0003
    t.state["finger_contacts"][0] = False
    t.step()
    assert not t.scene.game.open  # release threshold is strictly below 0.3mm
    t.state["button_depths_m"][0] = .000299
    t.step()
    assert t.scene.game.open == [0]
    t.press(1, release=False)  # began/qualified while card 0 was flipping
    t.step(84)
    t.state["button_depths_m"][1] = 0.
    t.state["finger_contacts"][1] = False
    t.step()
    assert t.scene.game.open == [0, 1]


def test_mismatch_display_and_flipback_follow_source_ticks():
    t = Trace("medium")
    t.press(0)
    t.step(84)
    t.press(2)
    end = t.scene.game.mismatch_deadline
    t.step(539)
    assert t.tick == end-1
    assert t.scene.game.desired[0] == 2.
    t.step()
    assert t.scene.game.desired[0] == t.scene.game.desired[2] == 178.
    assert not t.scene.game.ready()
    t.step(84)
    assert t.scene.game.ready()
    assert not t.scene.game.pairs


def test_pair_timer_without_actual_faceup_does_not_manufacture_pair_evidence():
    t = Trace("medium")
    t.frozen.add(1)
    t.pair(0)
    assert not t.scene.game.pairs
    assert t.result().value == 0


def test_more_than_two_physically_unmatched_faceup_overrides_partial_score():
    t = Trace("medium")
    t.pair(0)
    t.pair(1)
    assert t.result().value == 11
    for i in (4, 5, 6):
        t.state["card_quaternions_wxyz"][i] = quaternion(2)
        t.state["card_hinge_angles_deg"][i] = 2.
    t.step()
    assert t.result().value == 0
    assert t.result().components["max_unmatched_faceup"] == 3


def test_missing_or_nonfinite_physical_state_is_invalid_not_zero():
    t = Trace()
    assert score(t.scene.evidence(0)).value == 0  # initial measured state, zero-row EOF
    with pytest.raises(ValueError, match="tick mismatch"):t.scene.evidence(1)
    t.state["card_quaternions_wxyz"][0] = [float("nan"), 0., 0., 0.]
    with pytest.raises(ValueError, match="quaternion"):t.step()
    assert not t.commands


def test_evaluation_profile_and_seed_mismatch_are_not_training_fallbacks():
    s = InstanceSpec("M2", "hard-slot0", "hard", "source", "1", "1", "v1")
    with pytest.raises(ProfileBlocked, match="profiles/slots"):build(s, None)
    t = Trace()
    t.binding["identity"]["seed"] = 8
    with pytest.raises(ValueError, match="binding mismatch"):validate(t.spec, t.binding)


def test_private_binding_and_prepared_scene_hashes(tmp_path):
    t = Trace()
    station = tmp_path/"synthetic.usda"
    station.write_text("public synthetic placeholder")
    t.binding["scene_sha256"] = sha256(station)
    path = tmp_path/"binding.json"
    path.write_text(json.dumps(t.binding))
    spec = InstanceSpec(**{**t.spec.__dict__, "config": {**t.spec.config, "m2_binding_sha256": sha256(path)}})
    assets = NS(resolve=lambda key: {"station": station, "m2_binding": path}[key])
    assert load(spec, assets)["identity"]["seed"] == 7
    station.write_text("tampered")
    with pytest.raises(ValueError, match="scene SHA"):load(spec, assets)


@pytest.mark.parametrize("tamper", [None, "seed", "condition", "source_version"])
def test_inspected_source_spec_conversion_preserves_case_seed_and_ignores_paths(tmp_path, tamper):
    t = Trace()
    source = dict(task_id="MEMORY_CARDS", case_id=t.spec.instance_id, difficulty="easy", split="train",
        binding=dict(declared_version="M2-frozen-source-sdk-v1", task_definition_version="memory-bridge-final",
                     config_sha256="c"*64),
        config=dict(seed=7, source_root="/synthetic-do-not-read", native_inputs=dict(
            args=dict(scene="/synthetic-do-not-open/scene.usda"),
            manifest=dict(mechanism=dict(rows=2, cols=2, card_types=t.binding["card_identities"],
                button_rest={str(i): 1. for i in range(4)})))))
    if tamper == "seed":source["config"]["seed"] = 8
    if tamper == "condition":source["binding"]["config_sha256"] = "d"*64
    if tamper == "source_version":source["binding"]["declared_version"] = "unselected-source"
    path = tmp_path/"source-spec.json"
    path.write_text(json.dumps(source))
    station = tmp_path/"source-scene.usda"
    station.write_text("public synthetic source placeholder")
    spec = InstanceSpec(**{**t.spec.__dict__, "source_id": "M2-frozen-source-sdk-v1",
        "assets": {"m2_source_spec": "source-spec.json", "station": "source-scene.usda"},
        "config": {"m2_mode": "diagnostic_training", "m2_source_spec_sha256": sha256(path),
                   "m2_scene_sha256": sha256(station), "m2_condition_sha256": "c"*64}})
    resolved = []
    def resolve(key):
        resolved.append(key)
        return {"m2_source_spec": path, "station": station}[key]
    if tamper:
        with pytest.raises(ValueError, match="source case/seed/condition/version"):load(spec, NS(resolve=resolve))
    else:
        binding = load(spec, NS(resolve=resolve))
        assert binding["identity"]["seed"] == 7
        assert binding["identity"]["instance_id"] == t.spec.instance_id
        assert binding["button_rest_m"] == [1.]*4
        assert resolved == ["m2_source_spec", "station"]


def test_catalog_and_source_boundary():
    from mind2act_eval.tasks.catalog import load_task
    from mind2act_eval.validation import check_sources
    assert load_task("M2").task_id == "M2"
    assert check_sources() == []


@pytest.mark.parametrize('release_offset', [0, 1])
def test_ready_at_release_accepts_press_started_during_previous_flip(release_offset):
    t = Trace()
    t.press(0)
    ready_at = t.scene.game.flip_until
    t.step(ready_at - t.tick - 5)
    t.press(1, release=False)
    assert t.tick == ready_at - 1
    t.step(release_offset)
    t.state['button_depths_m'][1] = 0.
    t.state['finger_contacts'][1] = False
    t.step()
    assert t.scene.game.open == [0, 1]
    t.step(108)
    assert len(t.scene.game.pairs) == 1


def test_release_while_still_busy_is_not_queued_for_later():
    t = Trace()
    t.press(0)
    ready_at = t.scene.game.flip_until
    t.step(ready_at - t.tick - 8)
    t.press(1)
    assert t.tick < ready_at
    t.step(200)
    assert t.scene.game.open == [0] and not t.scene.game.pairs


def test_release_on_pair_completion_tick_can_select_the_next_card():
    t = Trace('medium')
    t.press(0)
    t.step(84)
    t.press(1)
    end = t.scene.game.pair_deadline
    t.step(end - t.tick - 5)
    t.press(2, release=False)
    t.state['button_depths_m'][2] = 0.
    t.state['finger_contacts'][2] = False
    t.step()
    assert t.tick == end and len(t.scene.game.pairs) == 1
    assert t.scene.game.open == [2]


def test_one_tick_mechanical_press_without_finger_selects_only_after_return():
    t = Trace()
    t.press(0, n=1, contact=False, release=False)
    assert t.scene.game.open == []
    t.state['button_depths_m'][0] = 0.
    t.step()
    assert t.scene.game.open == [0]


def test_all_pairs_need_hinge_angles_not_world_vertical_or_idle_buttons():
    t = Trace()
    t.pair(0); t.pair(1)
    values = to_json(t.scene.evidence(t.tick).values)
    values['all_buttons_released'] = False
    values['normal_termination'] = False
    assert score(NativeEvidence('M2', t.spec.instance_id, t.tick, values)).value == 100


def test_hinge_measurement_uses_joint_frames_when_world_orientation_is_tilted():
    from scipy.spatial.transform import Rotation
    from mind2act_eval.tasks.M2.mechanism import hinge_angle_deg
    base = Rotation.from_euler('x', 30, degrees=True)
    card = base * Rotation.from_euler('y', 2, degrees=True)
    def wxyz(r):
        x,y,z,w = r.as_quat()
        return [float(w),float(x),float(y),float(z)]
    assert hinge_angle_deg(wxyz(card), wxyz(base)) == pytest.approx(2.)
