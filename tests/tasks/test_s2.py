"""Synthetic physical measurements for CPU rules; no native/GPU claims."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import hashlib

import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.S2.binding import digest, effective_config, VERSION
from mind2act_eval.tasks.S2.evidence import Rows, read_rows
from mind2act_eval.tasks.S2.scene import validate
from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.tasks.S2.binding import SOURCE_ID
from mind2act_eval.tasks.S2.measurements import validate_frame
from mind2act_eval.tasks.S2.scorer import score
from mind2act_eval.tasks.S2.scene import instruction
from mind2act_eval.tasks.S2.source_rules.buttons import Task
from mind2act_eval.tasks.S2.source_rules.body import segment_hits_box


def source_case(level="easy"):
    n, count = {"easy": (4,3), "medium": (5,7), "hard": (6,12)}[level]
    buttons = {name: {"center": [x,y,.803]} for name,x,y in (
        ("up",-.4,.01), ("left",-.485,-.075), ("right",-.315,-.075), ("down",-.4,-.16))}
    c = {"task":"S2", "version":"s2-four-case-native-v4", "difficulty":level, "seed":0,
         "physics_hz":120, "control_hz":30, "cue_ticks":600, "stable_ticks":60, "max_ticks":10000,
         "max_linear_speed":.015, "max_angular_speed":.15, "support_tolerance":.004,
         "objects":{"slider":{"position":[.0425,.0425,.792], "size":[.036,.036,.024]}},
         "board":{"size":n,"cell_size":.085,"origin":[0.,0.],"floor_z":.78,
                  "start":[0,0],"goal":[n-1,n-1],"shortest_steps":2*(n-1),
                  "blocked":[[x,y] for x in range(n-1) for y in range(1,n)][:count]},
         "board_presentation":{"initial_map_ticks":600}, "feedback_ticks":0,
         "direction_control":{"buttons":buttons,"press_depth_m":.003,"release_depth_m":.001,
                              "press_ticks":3,"one_cell_per_press":True,"require_release":True,
                              "arrival_tolerance_m":.004,"cell_delta":{"up":[0,1],"down":[0,-1],"left":[-1,0],"right":[1,0]}},
         "instruction":instruction()}
    c["case_sha256"] = digest(c)
    return c


def frame(c, tick, xy, target, pressed=(), *, depth=.003, robot_slider=False, speed=0.):
    z = .792
    contacts = ["/World/Task/Board/floor"] + (["/World/RobotLeft/Asset/link7/Collision"] if robot_slider else [])
    body = {"lower":[xy[0]-.018,xy[1]-.018,z-.012],"upper":[xy[0]+.018,xy[1]+.018,z+.012],
            "linear":[speed,0.,0.],"angular":[0.,0.,0.],"position":[*xy,z],"orientation_wxyz":[1.,0.,0.,0.],
            "contacts":contacts,"support_contacts":[{"path":contacts[0],"position":[*xy,.78],"normal":[0.,0.,1.],"separation":0.}]}
    caps, events = {}, []
    for key,spec in c["direction_control"]["buttons"].items():
        d = depth if key in pressed else 0.
        caps[key] = {"depth_m":d,"robot_contact":key in pressed,
                     "position_m":[spec["center"][0],spec["center"][1],spec["center"][2]-d],
                     "orientation_wxyz":[1.,0.,0.,0.],"linear_velocity_m_s":[0.,0.,0.],"angular_velocity_rad_s":[0.,0.,0.]}
        if key in pressed:
            events.append({"pair":[f"/World/Task/Design/Controls/{key}/Cap/Collision","/World/RobotLeft/Asset/link7/Collision"],
                           "active":True,"contact_count":1,"min_separation_m":0.,
                           "points":[{"position":caps[key]["position_m"],"normal":[0.,0.,1.],"impulse":[0.,0.,.1],"separation":0.}]})
    return {"tick":tick,"simulation_time_s":tick/120,"bodies":{"slider":body},"direction_buttons":caps,
            "stage_drive_target_m":list(target),"contact_events":events,"q":{"Left":[0.]*8,"Right":[0.]*8}}


class PhysicsFixture:
    def __init__(self, tmp_path, level="easy"):
        self.source = source_case(level)
        self.c = effective_config(self.source)
        self.session = Task(self.c)
        self.tick, self.terminal = 0, None
        self.xy = self.c["objects"]["slider"]["position"][:2]
        self.initial = frame(self.c,0,self.xy,[0.,0.])
        self.session.sample(self.initial)
        self.writer = Rows(tmp_path / f"{level}-measurements.jsonl")

    def step(self, count=1, *, pressed=(), settle=False, robot_slider=False, speed=0., depth=.003):
        for _ in range(count):
            self.tick += 1
            target = list(self.session.stage_target)
            if settle:
                self.xy = [self.c["objects"]["slider"]["position"][i]+target[i] for i in range(2)]
            row = frame(self.c,self.tick,self.xy,target,pressed,robot_slider=robot_slider,speed=speed,depth=depth)
            validate_frame(self.c,row)
            if self.session.status == "running":
                self.session.sample(row)
                if self.session.status != "running":
                    self.terminal = self.tick
            self.writer.append(row)

    def press(self, key):
        self.step(3,pressed=(key,))
        self.step(settle=True)

    def route(self, *, final_held=0):
        self.step(240)
        n = self.c["board"]["size"]
        for key in ["right"]*(n-1)+["up"]*(n-2):
            self.press(key)
        self.step(3,pressed=("up",))
        if final_held:
            self.step(final_held,pressed=("up",),settle=True)
        self.step(settle=True)

    def evidence(self):
        return NativeEvidence("S2","synthetic",self.tick,{"source_case":self.source,"initial":self.initial,
            "input_protocol_version":VERSION,
            "measurements":self.writer.receipt(),"terminal_tick":self.terminal})

    def close(self):
        self.writer.close()


def test_source_global_cutoff_reserves_environment_observation():
    source = source_case()
    source['max_ticks'] = 245
    source['case_sha256'] = digest({k:v for k,v in source.items() if k != 'case_sha256'})
    c = effective_config(source)
    session = Task(c)
    xy = c['objects']['slider']['position'][:2]
    for tick in range(246):
        session.sample(frame(c, tick, xy, [0., 0.]))
    assert session.status == 'running'  # 0.5 s belongs to the environment, not this cutoff.


@pytest.mark.parametrize("level",["easy","medium","hard"])
def test_current_rules_keep_the_original_source_case_and_only_update_cue(level):
    original = source_case(level)
    before = deepcopy(original)
    c = effective_config(original)
    assert original == before and c["cue_ticks"] == 240
    assert c["source_case_sha256"] == original["case_sha256"]
    assert c["board"] == original["board"] and c["objects"] == original["objects"]
    assert c["direction_control"] == original["direction_control"]
    original["board"]["origin"][0] += .1
    with pytest.raises(ValueError,match="checksum"):
        effective_config(original)


@pytest.mark.parametrize("level",["easy","medium","hard"])
def test_legal_goal_counts_without_extra_wait(tmp_path, level):
    p = PhysicsFixture(tmp_path,level)
    p.route()
    assert score(p.evidence()).value == 100
    p.step(58,settle=True)
    assert score(p.evidence()).value == 100
    p.step(settle=True)
    result = score(p.evidence())
    assert result.value == 100 and result.components["rubric"] == VERSION
    assert result.components["goal_stable_ticks"] > 0
    p.close()


def test_final_goal_needs_no_button_release_or_stability_timer(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.route(final_held=120)
    assert p.session.goal_ticks > 0
    assert score(p.evidence()).value == 100
    p.step(59,settle=True)
    assert score(p.evidence()).value == 100
    p.close()


def test_long_hold_does_not_repeat_and_simultaneous_press_requires_full_release(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    p.step(120,pressed=("right",),settle=True)
    assert p.session.commands == 1
    p.step(settle=True)
    p.step(5,pressed=("up","right"),settle=True)
    p.step(5,pressed=("right",),settle=True)
    assert p.session.commands == 1
    p.step(settle=True)
    p.press("right")
    assert p.session.commands == 2
    assert score(p.evidence()).value == 0
    p.close()


@pytest.mark.parametrize("failure",["direct_contact","forbidden","boundary","shallow"])
def test_strict_failure_and_partial_progress_earn_zero(tmp_path, failure):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    if failure == "direct_contact":
        p.step(robot_slider=True)
    elif failure == "forbidden":
        p.press("up")
    elif failure == "boundary":
        p.press("left")
    else:
        p.step(3,pressed=("right",),depth=.00299)
        p.step(settle=True)
        assert p.session.commands == 0
    assert score(p.evidence()).value == 0
    p.close()


def test_current_raw_contact_batch_must_agree_with_button_claims():
    c = effective_config(source_case())
    row = frame(c,240,[.0425,.0425],[0.,0.],("right",))
    row["contact_events"] = []
    with pytest.raises(ValueError,match="contact"):
        validate_frame(c,row)


def test_swept_footprint_includes_boundary_contact():
    assert segment_hits_box((0.,0.),(1.,0.),(.5,0.),(.6,.1))
    assert not segment_hits_box((0.,0.),(1.,0.),(.5,.01),(.6,.1))


def test_resolver_scene_and_requested_seed_are_bound(tmp_path):
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    c = source_case()
    cfg = {"source_case":c,"source_case_sha256":c["case_sha256"],"instruction":instruction(),"input_protocol_version":VERSION,
           "scene_sha256":hashlib.sha256(scene.read_bytes()).hexdigest()}
    s = InstanceSpec("S2","synthetic","easy",SOURCE_ID,VERSION,"test","v1",config=cfg)
    assets = NS(resolve=lambda asset_id:scene)
    assert validate(s,assets)[0] == c
    wrong = to_json(s)
    wrong["seed"] = 1
    with pytest.raises(ValueError,match="seed"):
        validate(InstanceSpec.from_dict(wrong),assets)
    scene.write_text("#usda 1.0\n# changed\n")
    with pytest.raises(ValueError,match="scene checksum"):
        validate(s,assets)


def test_drive_target_claim_cannot_replace_a_measured_press(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    native = p.evidence()
    p.close()
    values = to_json(native.values)
    path = p.writer.path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["stage_drive_target_m"] = [.085,0.]
    raw = b"".join((json.dumps(row)+"\n").encode() for row in rows)
    path.write_bytes(raw)
    values["measurements"].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    assert score(NativeEvidence("S2","synthetic",native.tick,values)).value == 0


def test_velocity_does_not_disqualify_actual_goal(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.route()
    p.step(30,settle=True)
    p.step(settle=True,speed=.016)
    assert p.session.goal_ticks > 0
    p.step(59,settle=True)
    assert score(p.evidence()).value == 100
    p.step(settle=True)
    assert score(p.evidence()).value == 100
    p.close()


def test_contact_setup_never_writes_global_station_or_robot_physics(monkeypatch):
    import pxr
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    from mind2act_eval.tasks.S2.scene import prepare_task_contacts
    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/Task/Objects/slider", "/World/RobotLeft/Asset/base_link", "/World/CameraStand"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage,path).GetPrim())
    physics = UsdPhysics.Scene.Define(stage,"/World/Physics").GetPrim()
    physics.CreateAttribute("physxScene:enableCCD",Sdf.ValueTypeNames.Bool).Set(False)
    touched = []
    class Reports:
        @staticmethod
        def Apply(prim):
            touched.append(str(prim.GetPath()))
            return NS(CreateThresholdAttr=lambda value:None)
    monkeypatch.setattr(pxr,"PhysxSchema",NS(PhysxContactReportAPI=Reports),raising=False)
    prepare_task_contacts(stage)
    assert touched == ["/World/Task/Objects/slider"]
    assert physics.GetAttribute("physxScene:enableCCD").Get() is False


def test_training_diagnostic_retains_its_explicit_source_cue_without_changing_data():
    from mind2act_eval.tasks.S2.binding import SOURCE_VERSION
    source = source_case()
    before = deepcopy(source)
    current = effective_config(source,input_version=VERSION)
    training = effective_config(source,input_version=SOURCE_VERSION)
    assert source == before
    assert current["cue_ticks"] == 240 and training["cue_ticks"] == 600
    assert current["board"] == training["board"]
    assert current["objects"] == training["objects"]
    for version in ("unknown",None):
        with pytest.raises(ValueError,match="input protocol"):
            effective_config(source,input_version=version)


def test_legal_goal_motion_without_button_cycle_earns_full_score(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    for axis in range(2):
        for _ in range(510):
            p.tick += 1
            p.xy[axis] += .0005
            row = frame(p.c,p.tick,p.xy,p.session.stage_target)
            row["bodies"]["slider"]["linear"][axis] = .06
            validate_frame(p.c,row)
            if p.session.status == "running":
                p.session.sample(row)
                if p.session.status != "running":
                    p.terminal = p.tick
            p.writer.append(row)
    p.step(60)
    result = score(p.evidence())
    assert result.value == 100
    assert result.components["reason"] is None
    assert result.components["button_commands"] == 0
    p.close()


def test_push_beyond_motor_segment_is_not_a_scoring_violation(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    p.step(3,pressed=("right",))
    p.xy[0] += .04
    p.step(speed=.06)
    assert p.session.status == "running"  # Real in-flight movement need not arrive in one tick.
    p.xy[1] += p.c["direction_control"]["arrival_tolerance_m"]+.00001
    p.step(speed=.06)
    assert score(p.evidence()).components["reason"] is None
    p.close()


def test_drift_after_motor_arrival_is_not_a_scoring_violation(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    p.press("right")
    assert not p.session.stage_busy
    p.xy[0] -= .01
    p.step(speed=.06)
    assert score(p.evidence()).value == 0
    assert p.session.reason is None
    p.close()


def test_no_four_mm_authorization_margin(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    tolerance = p.c["direction_control"]["arrival_tolerance_m"]
    p.xy[0] += tolerance
    p.step()
    assert p.session.status == "running"
    p.xy[0] += .000001
    p.step()
    assert p.session.reason is None
    p.close()


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_notice_contact_and_small_motion_are_not_terminal_penalties(tmp_path, level):
    p=PhysicsFixture(tmp_path,level)
    p.step(10,robot_slider=True)
    assert p.session.status=='running' and p.session.reason is None
    p.route();p.step(59,settle=True)
    assert score(p.evidence()).value==100
    p.close()


def test_direct_push_to_goal_is_accepted_immediately_but_leaving_goal_is_not(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    n = p.c['board']['size']
    p.xy = [(n-.5)*p.c['board']['cell_size'], .0425]
    p.step(robot_slider=True, speed=.1)
    p.xy[1] = (n-.5)*p.c['board']['cell_size']
    p.step(robot_slider=True, speed=.1)
    assert score(p.evidence()).value == 100
    p.xy[1] -= p.c['board']['cell_size']
    p.step()
    assert score(p.evidence()).value == 0
    p.close()


def test_forbidden_cell_is_not_inflated_by_slider_circumradius(tmp_path):
    p = PhysicsFixture(tmp_path)
    p.step(240)
    p.xy[1] = .065  # radius .025 crosses y=.085; actual center stays outside forbidden cells
    p.step(robot_slider=True)
    assert p.session.reason is None
    p.close()


def test_actual_footprint_crossing_forbidden_cell_is_still_a_violation(tmp_path):
    p=PhysicsFixture(tmp_path);p.step(240)
    p.xy[1]=.07
    p.step()
    assert 'forbidden_cell_crossed' in p.session.violations
    assert p.session.status=='running'
    p.close()


def test_recipe_requires_return_strictly_below_one_mm_before_repress(tmp_path):
    p=PhysicsFixture(tmp_path);p.step(240)
    p.step(pressed=('right',))
    assert p.session.commands==1
    p.step(pressed=('right',),depth=.001,settle=True)
    p.step(pressed=('right',))
    assert p.session.commands==1
    p.step(pressed=('right',),depth=.000999,settle=True)
    p.step(pressed=('right',))
    assert p.session.commands==2
    p.close()


def test_recipe_does_not_require_slider_arrival_after_button_has_returned(tmp_path):
    p=PhysicsFixture(tmp_path);p.step(240)
    p.step(pressed=('right',))
    assert p.session.commands==1 and p.session.stage_busy
    p.step()  # measured button return; slider is still travelling
    p.step(pressed=('right',))
    assert p.session.commands==2
    assert p.session.stage_target==[.17,0.]
    p.close()


def test_unpressed_direction_cap_does_not_block_previous_key_return(tmp_path):
    p=PhysicsFixture(tmp_path);p.step(240)
    p.step(pressed=('right',))
    p.step(pressed=('left',),depth=.0012,settle=True)  # left never reaches 3mm; right has returned
    p.step(pressed=('up',))
    assert p.session.commands==2
    p.close()
