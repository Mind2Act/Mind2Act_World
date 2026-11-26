"""Shared-world M2 port, autonomous after source initialization; native pending."""
from pathlib import Path

from mind2act_eval.contracts import Health
from mind2act_eval.tasks._observation import observation
from .binding import ProfileBlocked, load
from .mechanism import MemoryMechanism, hinge_angle_deg
from .evidence import RawTrace, advance, checked_frame, snapshot
from .scorer import score

INSTRUCTION = "Press and release a card's yellow button to turn it face up, revealing at most two unmatched cards at a time. Matching pictures stay face up; mismatches turn face down automatically. Wait until the flip or pair result finishes before selecting another card. Remember the pictures and their positions and match every pair."


class MemoryScene:
    termination_kind = 'result'

    def __init__(self, spec, ctx, binding, sensors):
        self.spec, self.ctx, self.binding, self.sensors = spec, ctx, binding, sensors
        self.game = MemoryMechanism(binding["card_identities"])
        self.last = checked_frame(sensors.read(), self.game.count)
        output = getattr(ctx, "output_dir", None)
        self.trace = RawTrace(Path(output)/"private"/"M2"/"physical.jsonl", spec, binding, self.last) if output is not None else None

    def before_step(self, tick):
        pass

    def after_step(self, tick):
        self.last = self.sensors.read()
        advance(self.game, tick, self.last)
        if self.trace is not None:
            self.trace.append(tick, self.last)
        self.sensors.set_card_targets(self.game.desired, self.game.matched)

    def observe(self, tick):
        return observation(self.spec, self.ctx.robot, tick, INSTRUCTION)

    def evidence(self, tick):
        evidence = snapshot(self.spec, self.binding, self.game, self.last, tick)
        if self.trace is not None:
            from mind2act_eval.contracts import NativeEvidence
            evidence = NativeEvidence(evidence.task_id, evidence.instance_id, tick,
                                      dict(evidence.values, raw_physical=self.trace.receipt()))
        return evidence

    def is_complete(self):
        return self.game.tick > 0 and score(self.evidence(self.game.tick)).value == 100

    def health(self):
        return Health(True)

    def result_metadata(self):
        return dict(binding_mode=self.binding["mode"],
                    formal_evaluation_profile="not_claimed" if self.binding['mode'] in ('source_bound_test', 'source_bound_validation') else "blocked",
                    rule="m2-spec-20261030-v1", initialization_physics_ticks=240,
                    source_manifest_sha256=self.binding["source_manifest_sha256"],
                    source_scene_sha256=self.binding["scene_sha256"],
                    native_structure=getattr(self.sensors, "structure", None),
                    raw_physical=self.trace.receipt() if self.trace is not None else None)

    def is_success(self) -> bool:
        return self.is_complete()

    def irreversible_failure(self) -> str | None:
        if self.game.tick > 0 and self.game.max_unmatched_faceup > 2:
            return "too_many_unmatched_faceup"
        return None

    def close(self):
        if self.trace is not None:
            self.trace.close()


class NativeSensors:
    def __init__(self, ctx, binding):
        from pxr import UsdPhysics, UsdGeom, PhysxSchema
        from isaacsim.core.prims import SingleRigidPrim
        stage = ctx.sim.stage
        from .native_stage import check_stage
        self.structure = check_stage(stage, len(binding["card_identities"]))
        ctx.robot.prepare_stage(stage)
        ctx.robot.attach_cameras(stage)
        from mind2act_eval.variations import prepare_station
        prepare_station(ctx)
        world = ctx.world
        arms = ctx.robot.articulations(world)
        n = len(binding["card_identities"])
        def body(path, name):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("M2 required native task prim missing: " + path)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            return world.scene.add(SingleRigidPrim(prim_path=path, name=name))
        self.buttons = [body(f"/World/Memory/Buttons/Key{i}", f"m2_button{i}") for i in range(n)]
        self.cards = [body(f"/World/Memory/Cards/Card{i}", f"m2_card{i}") for i in range(n)]
        self.drives, self.paired, self.hinge_frames = [], [], []
        for i in range(n):
            prim = stage.GetPrimAtPath(f"/World/Memory/Joints/Card{i}")
            drive = UsdPhysics.DriveAPI(prim, "angular")
            paired = stage.GetPrimAtPath(f"/World/Memory/Paired{i}")
            if not prim or not paired or not drive.GetTargetPositionAttr():
                raise ValueError("M2 authored card hinge/paired marker missing")
            joint = UsdPhysics.RevoluteJoint(prim)
            def quaternion(value):
                return [float(value.GetReal()), *map(float, value.GetImaginary())]
            self.hinge_frames.append((quaternion(joint.GetLocalRot0Attr().Get()),
                                      quaternion(joint.GetLocalRot1Attr().Get())))
            self.drives.append(drive)
            self.paired.append(UsdGeom.Imageable(paired))
        self.contacts = ctx.sim.contacts
        world.reset()
        ctx.robot.reset_to_rest()
        ctx.robot.verify(arms, stage)
        # The old expert owned Wait(2). Initialization now owns it and enables
        # the physical mechanism on construction, before the first observation.
        for _ in range(240):
            world.step(render=False)
        self.contacts.clear()
        # Preserve the original mechanism's authored button zero; do not
        # silently replace it with a new post-settle calibration.
        self.rest = list(binding["button_rest_m"])

    def read(self):
        quaternions = [body.get_world_pose()[1].tolist() for body in self.cards]
        return dict(button_depths_m=[z - float(body.get_world_pose()[0][2]) for z, body in zip(self.rest, self.buttons)],
            finger_contacts=[any(self.contacts.touch(f"/World/Memory/Buttons/Key{i}", robot_only=True, link=link)
                for link in ("link7", "link8")) for i in range(len(self.buttons))],
            card_quaternions_wxyz=quaternions,
            card_hinge_angles_deg=[hinge_angle_deg(q, *frames) for q, frames in zip(quaternions, self.hinge_frames)])

    def set_card_targets(self, desired, matched):
        for drive, target in zip(self.drives, desired):
            # Task hinge attributes use source USD angular degrees; robot units
            # and robot drive configuration remain exclusively shared-owned.
            drive.GetTargetPositionAttr().Set(target)
        for i in matched:
            self.paired[i].MakeVisible()


def build(spec, ctx):
    if spec.config.get("m2_mode") not in ("diagnostic_training", "source_bound_test", "source_bound_validation"):
        raise ProfileBlocked("M2 formal evaluation profiles/slots unavailable; no training substitution")
    binding = load(spec, ctx.assets)
    return MemoryScene(spec, ctx, binding, NativeSensors(ctx, binding))
