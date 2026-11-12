"""Synthetic CPU physical traces, never a CP07 native rollout or eval profile."""
from copy import deepcopy
from types import SimpleNamespace as NS
import json

import pytest

from mind2act_eval.contracts import InstanceSpec, NativeEvidence
from mind2act_eval.tasks.CP07.binding import ProfileBlocked, load, sha256, validate
from mind2act_eval.tasks.CP07.producer import BridgeFacts, rotation_error
from mind2act_eval.tasks.CP07.scene import BridgeScene, build
from mind2act_eval.tasks.CP07.scorer import COUNTS, score


def fixture(difficulty="easy"):
    count, stock = COUNTS[difficulty]
    spec = InstanceSpec("CP07", "public-synthetic", difficulty, "public-synthetic", "1", "1", "v1",
                        seed=7, config={"cp07_mode": "diagnostic_training"})
    pieces = [dict(id=f"p{i}", path=f"/World/Pieces/p{i}", kind="bar", geometry_signature=["synthetic-cube", .02, .02, .02])
              for i in range(stock)]
    binding = dict(schema="cp07.source-binding.v2", identity={k: getattr(spec, k) for k in
        ("task_id", "instance_id", "difficulty", "source_id", "seed", "asset_version")},
        mode="diagnostic_training", solutions_scope="known-source-solutions", pieces=pieces,
        source_code_sha256="fff77c74af87096814463f1be451cb21de50dded2df9fde67f3160e625425f16",
        source_manifest_sha256="a"*64, scene_sha256="b"*64,
        bridge_bounds_m={"lo": [-1., -.1, .9], "hi": [1., .1, 1.2]}, button_xy_m=[0., .3],
        car_start_x_m=-.19, roadtop_start_m=.824, roadtop_end_m=.84, lane_y_m=0., road_width_m=.06, finish_x_m=.4,
        solutions=[dict(slots=[dict(p_m=[i*.04, 0., 1.], q_wxyz=[1., 0., 0., 0.],
            kind="bar", orientation_rule="not_applicable", compatible_piece_ids=[p["id"] for p in pieces],
            support_paths=["/World/Terrain/Foundation"]) for i in range(count)])])
    state = dict(pieces={p["id"]: dict(p_m=[i*.04, .3, .8], q_wxyz=[1., 0., 0., 0.],
        robot_contact=False, bilateral=False, support_contacts=["/World/Worktable"], inside_bridge=False)
        for i, p in enumerate(pieces)}, button_z_m=1., button_robot_contact=False, hands_clear=False,
        joint_error_rad=.0, car=dict(p_m=[-.19, 0., .824], up=1., heading_rad=0.,far_bank_supported=True,far_bank_center_inside=False))
    validate(spec, binding)
    return spec, binding, state


class Trace:
    def __init__(self, difficulty="easy"):
        self.spec, self.binding, self.state = fixture(difficulty)
        self.drives = []
        def read():
            state=deepcopy(self.state);car=state['car']
            car['far_bank_center_inside']=bool(self.binding['finish_x_m']<=car['p_m'][0]<=.6 and abs(car['p_m'][1])<=.3)
            return state
        sensors = NS(read=read, drive=lambda *a: self.drives.append(a))
        robot = NS(observe=lambda: dict(left_arm=[0]*6, right_arm=[0]*6, left_gripper=1., right_gripper=1.))
        self.scene = BridgeScene(self.spec, NS(robot=robot), self.binding, sensors)
        self.tick = 0

    def step(self, n=1):
        for _ in range(n):
            self.tick += 1
            self.scene.before_step(self.tick)
            self.scene.after_step(self.tick)

    def place(self, index, name=None, bilateral=True, lift=.04):
        piece = self.state["pieces"][name or f"p{index}"]
        piece.update(robot_contact=True, bilateral=bilateral)
        self.step()
        piece["p_m"][2] += lift
        piece["support_contacts"] = []
        self.step()
        piece.update(p_m=self.binding["solutions"][0]["slots"][index]["p_m"][:],
                     robot_contact=False, bilateral=False,
                     support_contacts=["/World/Terrain/Foundation"], inside_bridge=True)
        self.step()

    def press(self, n=8, release=True):
        self.state.update(button_z_m=.999, button_robot_contact=True)
        self.step(n)
        if release:
            self.state.update(button_z_m=1., button_robot_contact=False)
            self.step()

    def result(self):
        return score(self.scene.evidence(self.tick))

    def assemble(self):
        for i in range(COUNTS[self.spec.difficulty][0]):
            self.place(i)


@pytest.mark.parametrize("difficulty", COUNTS)
def test_native_scene_hooks_full_bridge_press_release_clear_and_finish(difficulty):
    t=Trace(difficulty);t.assemble()
    assert t.result().value==0
    t.press(release=False)
    assert t.drives[0]==(True,)  # physical press starts despite held button/hands
    t.state['car']['p_m'][0]=.4;t.step()
    assert t.result().value==100
    assert t.result().components['completion']=={'k':1,'n':1,'ratio':1.}
    assert t.scene.is_complete()
    assert 'p0' not in t.scene.observe(t.tick).instruction
    t.scene.close();assert t.drives[-1]==(False,)


def test_partial_bridge_and_button_do_not_earn_score():
    t=Trace();t.press()
    assert t.result().value==0
    for i in range(3):t.place(i)
    assert t.result().value==0
    t.place(4);assert t.result().value==0


@pytest.mark.parametrize("failure", ["no_lift", "single_finger", "position", "held", "unsupported"])
def test_bridge_pose_or_contact_facts_do_not_earn_partial(failure):
    t = Trace()
    t.place(0, bilateral=failure != "single_finger", lift=0. if failure == "no_lift" else .04)
    p = t.state["pieces"]["p0"]
    if failure == "position":p["p_m"][1] = .012
    if failure == "held":p["robot_contact"] = True
    if failure == "unsupported":p["support_contacts"] = []
    t.step()
    assert t.result().value == 0


def test_interchangeable_hard_stock_does_not_bind_expert_ids():
    t = Trace("hard")
    # A spare ID of exactly equivalent geometry can be a valid chosen piece.
    for i in range(13):t.place(i, name=f"p{i+13}")
    assert t.result().components["completion"]["k"] == 0
    t.press()
    t.state["hands_clear"] = True
    t.step()
    t.state["car"]["p_m"][0] = .4
    t.step()
    assert t.result().value == 100
    assert t.result().components["solutions_scope"] == "known-source-solutions"


def test_any_supplied_valid_solution_and_augmenting_piece_assignment():
    t=Trace();alternative=deepcopy(t.binding['solutions'][0])
    for slot in alternative['slots']:slot['p_m'][1]=-.2
    t.binding['solutions'].insert(0,alternative)
    t.assemble();assert t.result().value==0
    t.press();t.state['car']['p_m'][0]=.4;t.step()
    assert t.result().value==100


@pytest.mark.parametrize("key,value,valid", [
    ("lift_m", .035, False), ("lift_m", .035001, True),
    ("bilateral_fraction", .8, False), ("bilateral_fraction", .80001, True),
    ("position_error_m", .012, False), ("position_error_m", .011999, True),
])
def test_old_bridge_measurement_boundaries_do_not_veto_real_arrival(key,value,valid):
    t=Trace();t.press();t.state['car']['p_m'][0]=.4;t.step()
    v=t.scene.facts.snapshot();v['legacy_bridge_measurement']={key:value}
    assert score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v)).value==100


def test_nonfinite_native_fact_is_rejected_before_start_command():
    t=Trace();t.state['car']['p_m'][0]=float('nan')
    with pytest.raises(ValueError,match='native state'):t.step()
    assert not t.drives


def test_table_contact_time_cannot_mask_single_finger_elevated_carry():
    t = Trace()
    piece = t.state["pieces"]["p0"]
    piece.update(robot_contact=True, bilateral=True)
    t.step(120)  # double-finger dwell on the table is not transport evidence
    piece.update(bilateral=False, support_contacts=[])
    piece["p_m"][2] += .04
    t.step()
    piece.update(p_m=[0., 0., 1.], robot_contact=False,
                 support_contacts=["/World/Terrain/Foundation"], inside_bridge=True)
    t.step()
    assert t.result().value == 0


def test_extra_stock_and_past_parking_motion_do_not_block_arrival():
    t=Trace('medium');t.assemble()
    t.state['pieces']['p17'].update(inside_bridge=True,support_contacts=['/World/Pieces/p2'])
    t.state['car']['p_m'][0]+=.009;t.step();t.state['car']['p_m'][0]-=.009
    t.press();assert t.scene.facts.launched
    t.state['car']['p_m'][0]=.4;t.step();assert t.result().value==100


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_corrected_prestart_displacement_is_not_a_permanent_failure(difficulty):
    t=Trace(difficulty);t.state['car']['p_m'][1]=.20;t.step()
    t.state['car']['p_m'][1]=0.;t.step();t.press()
    assert t.scene.facts.launched
    t.state['car']['p_m'][0]=.4;t.step();assert t.result().value==100


@pytest.mark.parametrize("offset,valid", [(.011, True), (.011001, False)])
def test_prestart_y_tolerance_uses_source_vehicle_and_lane_geometry(offset, valid):
    t = Trace("medium")
    t.state["car"]["p_m"][1] = offset
    t.step()
    assert t.scene.facts.parking_ok is valid


@pytest.mark.parametrize("failure", ["route", "upright", "tracking", "second_press"])
def test_old_route_upright_tracking_and_repeat_press_gates_do_not_veto_arrival(failure):
    t = Trace()
    t.assemble()
    t.press()
    t.state["hands_clear"] = True
    t.step()
    if failure == "route":t.state["car"]["p_m"][1] = .1
    if failure == "upright":t.state["car"]["up"] = .849
    if failure == "tracking":t.state["joint_error_rad"] = .12
    if failure == "second_press":t.press()
    t.state["car"]["p_m"][0] = .4
    t.step()
    assert t.result().value == 100


def test_physical_press_launches_without_extra_dwell_and_clock_gaps_are_invalid():
    t=Trace();t.press(1);assert t.drives[0]==(True,)
    with pytest.raises(ValueError,match='consecutive'):t.scene.after_step(t.tick+2)


def test_press_does_not_wait_for_hidden_bridge_qualification():
    t=Trace();t.press(1,release=False)
    assert t.drives==[(True,)] and t.scene.facts.valid_button


def test_missing_bank_evidence_is_invalid_but_missing_bridge_telemetry_is_irrelevant():
    t=Trace();t.press();t.state['car']['p_m'][0]=.4;t.step()
    v=t.scene.facts.snapshot();v['solutions']=[{'unknown':None}]
    assert score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v)).value==100
    del v['terminal_car']['far_bank_supported']
    with pytest.raises(ValueError,match='arrival'):score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v))
    assert rotation_error([1,0,0,0],[-1,0,0,0])==0
    with pytest.raises(ValueError):rotation_error([0,0,0,0],[1,0,0,0])


def test_evaluation_blocked_without_touching_sim_or_substituting_profile():
    spec = InstanceSpec("CP07", "agentic-hard-slot0", "hard", "s", "1", "1", "v1")
    with pytest.raises(ProfileBlocked, match="explicit diagnostic_training"):
        build(spec, None)
    spec, binding, _ = fixture()
    binding["identity"]["seed"] += 1
    with pytest.raises(ValueError, match="binding mismatch"):validate(spec, binding)


def test_private_binding_and_scene_digests_are_enforced(tmp_path):
    spec, binding, _ = fixture()
    station = tmp_path / "source-scene.usda"
    station.write_text("public synthetic only")
    binding["scene_sha256"] = sha256(station)
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    spec = InstanceSpec(**{**spec.__dict__, "config": {**spec.config, "cp07_binding_sha256": sha256(path)}})
    assets = NS(resolve=lambda name: {"station": station, "cp07_binding": path}[name])
    assert load(spec, assets)["identity"]["seed"] == 7
    station.write_text("tampered")
    with pytest.raises(ValueError, match="scene SHA"):load(spec, assets)
    path.write_text("{}")
    with pytest.raises(ValueError, match="binding SHA"):load(spec, assets)


def test_catalog_is_lazy_and_no_old_world_or_expert_imports():
    from mind2act_eval.tasks.catalog import load_task
    from mind2act_eval.validation import check_sources
    from pathlib import Path
    assert load_task("CP07").task_id == "CP07"
    root = Path(__file__).resolve().parents[3] / "src/mind2act_eval"
    assert check_sources(root) == []


def test_final_physical_press_launches_without_bridge_qualification_or_withdrawal():
    t=Trace('hard');t.press(n=1,release=False)
    assert t.drives and t.drives[0]==(True,)


def test_final_bridge_steps_have_no_partial_score():
    t=Trace();t.assemble()
    assert t.result().value==0


def test_final_arrival_is_binary_and_t0_cannot_keep_an_old_success():
    t=Trace('medium');t.press(n=1)
    t.state['car'].update(p_m=[.4,.1,.84],up=.7,far_bank_supported=True,far_bank_center_inside=True)
    t.step();assert t.result().value==100
    t.state['car'].update(p_m=[.7,.1,.4],far_bank_supported=False,far_bank_center_inside=False)
    t.step();assert t.result().value==0
