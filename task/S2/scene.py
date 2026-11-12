"""S2 prepared native scene on shared robot/simulator; task-local XY mechanism."""
import hashlib
import json
from pathlib import Path

from mind2act_eval.contracts import Health, NativeEvidence, Observation, to_json
from .binding import SOURCE_ID, VERSION, effective_config
from .display import BoardDisplay
from .evidence import Rows, Manifolds, measure, SLIDER, CONTROLS
from .measurements import validate_frame
from .source_rules.buttons import Task


def instruction():
    return json.loads((Path(__file__).with_name("config") / "instructions.json").read_text())["entries"][0]["en"]


def validate(spec, assets):
    if spec.task_id != "S2" or spec.source_id != SOURCE_ID or spec.config_version != VERSION:
        raise ValueError("S2 source/rule binding mismatch")
    if spec.clock.physics_hz != 120 or spec.clock.control_hz not in (30, 120) or spec.clock.rgb_hz != 30:
        raise ValueError("S2 requires 120 Hz physics, 30/120 Hz control and 30 Hz RGB")
    source = to_json(spec.config["source_case"])
    if source["difficulty"] != spec.difficulty or source["seed"] != spec.seed:
        raise ValueError("S2 source case differs from requested difficulty/seed")
    if spec.config["instruction"] != instruction() or source["instruction"] != instruction():
        raise ValueError("S2 instruction differs from the canonical delivery")
    if spec.config["source_case_sha256"] != source["case_sha256"]:
        raise ValueError("S2 requested source case checksum mismatch")
    from mind2act_eval.sim.source_assets import verify_source_file
    verify_source_file(assets, "station", spec.config["scene_sha256"], error="S2 prepared scene checksum mismatch")
    return source, effective_config(source, input_version=spec.config["input_protocol_version"])


def prepare_task_contacts(stage):
    """Enable task measurements without overriding the shared physics profile."""
    from pxr import UsdPhysics, PhysxSchema
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/Task/"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


class NavigationScene:
    termination_kind = 'result'

    def __init__(self, spec, ctx, source, c, slider, buttons, robots, contacts):
        self.spec, self.ctx, self.source, self.c = spec, ctx, source, c
        self.slider, self.buttons, self.robots, self.contacts = slider, buttons, robots, contacts
        self.session = Task(c)
        self.display = BoardDisplay(ctx.sim.stage, c["board"])
        self.target = [0., 0.]
        self.terminal_tick = None
        self.initial = self._measure(0)
        validate_frame(c, self.initial)
        self.session.sample(self.initial)
        self.display.update(self.session.public())
        self.rows = Rows(Path(ctx.output_dir) / "private" / "S2" / "measurements.jsonl")

    def _measure(self, tick):
        return measure(tick, self.c, self.slider, self.buttons, self.robots,
                       self.contacts, self.target, self.ctx.world)

    def before_step(self, tick):
        from pxr import UsdPhysics
        self.contacts.events.clear()
        self.target = list(self.session.stage_target)
        joint = self.ctx.sim.stage.GetPrimAtPath("/World/Task/Guide/PlanarJoint")
        for i, axis in enumerate(("transX", "transY")):
            UsdPhysics.DriveAPI(joint, axis).GetTargetPositionAttr().Set(self.target[i])

    def after_step(self, tick):
        frame = self._measure(tick)
        validate_frame(self.c, frame)
        if self.session.status == "running":
            self.session.sample(frame)
            if self.session.status != "running":
                self.terminal_tick = tick
        # Expert replay may retain later physical samples for complete coverage;
        # the task terminal and motor target remain the real first terminal.
        self.rows.append(frame)
        self.display.update(self.session.public())

    def observe(self, tick):
        return Observation(tick, tick / 120, **self.ctx.robot.observe(),
                           instruction=self.spec.config["instruction"], protocol_version=self.spec.protocol_version)

    def evidence(self, tick):
        return NativeEvidence("S2", self.spec.instance_id, tick,
                              {"source_case": self.source, "initial": self.initial,
                               "input_protocol_version": self.spec.config["input_protocol_version"],
                               "measurements": self.rows.receipt(), "terminal_tick": self.terminal_tick})

    def is_complete(self):
        return self.session.at_goal and not self.session.violations

    def health(self):
        from math import isfinite
        return Health(all(isfinite(float(v)) for v in self.slider.get_world_pose()[0]),
                      "S2 slider pose must be finite")

    def result_metadata(self):
        physics = self.ctx.sim.stage.GetPrimAtPath("/World/Physics")
        observed = {name: physics.GetAttribute("physxScene:" + name).Get() if physics else None
                    for name in ("timeStepsPerSecond", "enableCCD", "solverType")}
        return {"source_id": SOURCE_ID, "rubric": VERSION, "native_status": self.session.status,
                "input_protocol_version": self.spec.config["input_protocol_version"],
                "source_cue_ticks": self.source["cue_ticks"], "effective_cue_ticks": self.c["cue_ticks"],
                "native_reason": self.session.reason, "terminal_tick": self.terminal_tick,
                "measurements": self.rows.receipt(), "initialization_physics_ticks": 0,
                "shared_physics_observation": observed,
                "source_wrapper_physics_reference": {"physics_hz":120, "enableCCD":True, "solverType":"TGS",
                    "prop_solver":self.source.get("prop_solver"), "contactOffset_m":.001, "restOffset_m":0.}}

    def is_success(self) -> bool:
        return self.is_complete()

    def close(self):
        self.rows.close()
        self.contacts.close()


def build(spec, ctx):
    source, c = validate(spec, ctx.assets)
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import UsdPhysics
    import numpy as np
    stage = ctx.sim.stage  # common launcher opened the resolver's prepared scene
    for path in (SLIDER, "/World/Task/Guide/PlanarJoint", *(CONTROLS + key + "/Cap" for key in c["direction_control"]["buttons"])):
        if not stage.GetPrimAtPath(path):
            raise ValueError("S2 prepared scene is missing " + path)
    ctx.robot.prepare_stage(stage)
    prepare_task_contacts(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    slider = world.scene.add(SingleRigidPrim(prim_path=SLIDER, name="S2Slider"))
    buttons = {key: world.scene.add(SingleRigidPrim(prim_path=CONTROLS + key + "/Cap", name="S2_" + key))
               for key in c["direction_control"]["buttons"]}
    ctx.robot.initialize(world)
    robots = ctx.robot.articulations(world)
    for body in (slider, *buttons.values()):
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
    for axis in ("transX", "transY"):
        UsdPhysics.DriveAPI(stage.GetPrimAtPath("/World/Task/Guide/PlanarJoint"), axis).GetTargetPositionAttr().Set(0.)
    contacts = Manifolds()
    contacts.subscribe()
    try:
        return NavigationScene(spec, ctx, source, c, slider, buttons, robots, contacts)
    except BaseException:
        contacts.close()
        raise
