"""E2 training props plus measured keys on shared station/robot/simulation."""
import math
from pathlib import Path

from mind2act_eval.contracts import Health, NativeEvidence, Observation
from .evidence import measure, Measurements
from .instances import SOURCE_ID, RUBRIC, validate
from .mechanism.clock import display_state, input_start
from .mechanism.keys import Keys


class ArithmeticScene:
    termination_kind = 'events'

    def __init__(self, spec, ctx, case, props, caps):
        self.spec, self.ctx, self.case = spec, ctx, case
        self.props, self.caps = props, caps
        self.keys = Keys()
        self.measurements = Measurements(Path(ctx.output_dir) / "private" / "E2" / "measurements.jsonl")
        self.completion_tick = None
        self._display(0)

    def _display(self, source_tick):
        from pxr import UsdGeom
        _, angle, show_expression = display_state(source_tick, self.case)
        self.props["lid_rotation"].Set(angle)
        registration = UsdGeom.Imageable(self.props["registration"].GetPrim())
        sequence = UsdGeom.Imageable(self.props["sequence"].GetPrim())
        if show_expression:
            registration.MakeInvisible()
            sequence.MakeVisible()
        else:
            registration.MakeVisible()
            sequence.MakeInvisible()

    def before_step(self, tick):
        # Source baseline updates display before physics at source step tick-1.
        self._display(tick - 1)

    def after_step(self, tick):
        robot = self.ctx.robot.observe()
        row = measure(tick, self.case, self.caps, self.ctx.sim.contacts,
                      (robot["left_gripper"], robot["right_gripper"]))
        self.measurements.append(row)
        self.keys.sample(tick, row["travel_m"], row["fingertip_contacts"],
                         enabled=tick >= input_start(self.case))
        if self.completion_tick is None and self.keys.complete(self.case["password"]):
            self.completion_tick = tick
        self._display(tick)

    def observe(self, tick):
        return Observation(tick, tick / 120, **self.ctx.robot.observe(),
                           instruction=self.spec.config["instruction"],
                           protocol_version=self.spec.protocol_version)

    def evidence(self, tick):
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
                              {"difficulty": self.spec.difficulty,
                               "target_digits": self.case["password"],
                               "timing": {key: self.case[key] for key in
                                          ("learn_seconds", "shuffle_seconds", "reasoning_seconds")},
                               "measurements": self.measurements.receipt(),
                               "normal_completion_tick": self.completion_tick})

    def is_complete(self):
        return len(self.keys.digits) >= len(self.case["password"]) and self.keys.released

    def health(self):
        valid = all(math.isfinite(float(v)) for cap in self.caps for v in cap.get_world_pose()[0])
        return Health(valid, "" if valid else "E2 non-finite cap pose")

    def result_metadata(self):
        return {"source_id": SOURCE_ID, "rubric": RUBRIC,
                "initialization_physics_ticks": 120,
                "measurements": self.measurements.receipt(),
                "normal_completion_tick": self.completion_tick}

    def events_complete(self) -> bool:
        return self.is_complete()

    def close(self):
        self.measurements.close()


def build(spec, ctx):
    case = validate(spec)
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Usd, UsdGeom, UsdPhysics
    from .mechanism.props import build_props

    # The common CLI opens the resolver station (and optional hash-pinned
    # relocation) before this builder. Task code never substitutes a station.
    stage = ctx.sim.stage
    if stage.GetPrimAtPath("/World/EKeyboard"):
        raise ValueError("E2 requires the clean station, not already-authored task props")
    ctx.robot.prepare_stage(stage)
    for root in ("/World/Worktable", "/World/TableFrame", "/World/PedestalLeft",
                 "/World/PedestalRight", "/World/CameraStand", "/World/Floor"):
        prim = stage.GetPrimAtPath(root)
        if not prim:
            raise ValueError("E2 selected station is missing " + root)
        for part in Usd.PrimRange(prim):
            if part.IsA(UsdGeom.Mesh) or part.IsA(UsdGeom.Cube):
                UsdPhysics.CollisionAPI.Apply(part).CreateCollisionEnabledAttr(True)
                if part.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(part).CreateApproximationAttr("none")
    props = build_props(stage, Path(ctx.output_dir) / "private" / "E2", case,
                        ctx.assets.resolve("font"))
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    caps = [world.scene.add(SingleRigidPrim(prim_path=p + "/Cap", name=f"E2Cap{i}"))
            for i, p in enumerate(props["buttons"])]
    contacts = ctx.sim.contacts
    ctx.robot.initialize(world)
    # The training baseline settles at the shared explicit rest pose for 120
    # ticks, outside the episode clock. Do not add physical tail ticks.
    rest = ctx.robot.rest_pose()
    from mind2act_eval.contracts import Action
    from mind2act_eval.robot import normalized_observation
    closed = normalized_observation(rest[6])
    action = Action(rest[:6], rest[:6], closed, closed)
    for _ in range(120):
        ctx.robot.apply(action)
        ctx.sim.step(render=False)
    contacts.clear()
    return ArithmeticScene(spec, ctx, case, props, caps)
