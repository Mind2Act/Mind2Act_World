"""S1 scene: the single `build(spec, ctx)` shared by every consumer.

Baseline: S-147 `direct-continuous-340aff8-run925de9` (builder `cases/S/runtime/builder.py` digest 79a0b996a4ad;
native world `physco_eval/worlds_v2/s.py`). The prepared scene (scene.usda + private_manifest.json) is an
asset of the instance produced offline by that builder. Carousel drive, block collider/mass/friction and
station-static setup are taken unchanged from the baseline world. Robot stage preparation (fixed base,
articulation root, actuator gains) is NOT done here: the shared X5 robot module owns it.
The source timeline is explicit in spec.config. Formal fixed50 uses the ruled one-second cue with no gap;
training replay retains the source timeline. The effective timeline is recorded in evidence and results.
"""
import json
import re
from pathlib import Path

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import Action, BuildContext, Health, InstanceSpec, NativeEvidence, Observation, Scene
from mind2act_eval.tasks._observation import observation, require_physics_hz
from .mechanism.placement import CarouselLedger, SPlacementReleaseTracker
from .mechanism.schedule import schedule, timeline_config

WARMUP_TICKS = 120


def _pair_has(pair, root):
    return any(path == root or path.startswith(root + "/") for path in pair)


class SScene:
    termination_kind = 'result'

    def __init__(self, spec, ctx, world, stage, tables, blocks, carousel, rotor, body_offsets, ledger, tracker, timeline_cfg):
        self.spec, self.ctx, self.stage, self.T = spec, ctx, stage, tables
        self.world, self.blocks, self.carousel, self.rotor = world, blocks, carousel, rotor
        self.body_offsets, self.ledger, self.tracker = body_offsets, ledger, tracker
        self.cfg, self.gap = timeline_cfg, timeline_cfg["cue_gap_s"]
        self.names = list(tables)
        self.angles = {k: 0. for k in tables}
        self.contact_pairs = {}
        self.phase, self.execution_start_step, self.tick = "cue", None, 0
        self._health = Health(True)
        self._subscription = None
        self._cue_visible = None

    # -- helpers ------------------------------------------------------------------------------------
    def center(self, key):
        from scipy.spatial.transform import Rotation
        pos, quat = self.blocks[key].get_world_pose()
        return pos + Rotation.from_quat([*quat[1:], quat[0]]).apply(self.body_offsets[key])

    def _set_cue(self, visible: bool) -> None:
        if visible == self._cue_visible:
            return
        from pxr import UsdShade
        for table in self.T.values():
            material = table["cue_material"] if visible else table["original_material"]
            UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath(table["target_floor_prim"])).Bind(
                UsdShade.Material(self.stage.GetPrimAtPath(material)))
        self._cue_visible = visible

    # -- runner hooks -------------------------------------------------------------------------------
    def before_step(self, tick: int) -> None:
        self.tick = tick
        phase, visible, angles = schedule(tick - 1, self.cfg, self.T, self.gap)
        self.phase = phase
        self.angles.update(angles)
        self._set_cue(visible)
        if phase == "execution" and self.execution_start_step is None:
            self.execution_start_step = tick - 1
        from isaacsim.core.utils.types import ArticulationAction
        self.carousel.apply_action(ArticulationAction(joint_positions=np.array([self.angles[self.names[0]]]),
                                                      joint_velocities=np.zeros(1)))

    def after_step(self, tick: int) -> None:
        try:
            observations, finished, rotations = self._sample()
            self.ledger.observe(tick, observations, program_finished=finished,
                                rotation_state=None, cue_visible=self.phase == 'cue')
        except ValueError as error:
            self._health = Health(False, "S1 evidence error: " + str(error))

    # -- Scene protocol -----------------------------------------------------------------------------
    def observe(self, tick: int) -> Observation:
        return observation(self.spec, self.ctx.robot, tick, public_instruction(self.spec))

    def evidence(self, tick: int) -> NativeEvidence:
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
                              dict(self.ledger.snapshot(), timeline=dict(self.cfg), difficulty=self.spec.difficulty))

    def health(self) -> Health:
        return self._health

    def is_complete(self) -> bool:
        from .scorer import score
        return self.tick > 0 and bool(self.ledger.blocks) and score(self.evidence(self.tick)).value == 100

    def result_metadata(self) -> dict:
        return {"s1_timeline": dict(self.cfg)}  # cue/rotation timing actually used, recorded in Result.task_metadata

    def is_success(self) -> bool:
        return self.is_complete()

    def close(self) -> None:
        self._subscription = None

    # -- measurement (old eval `sample_environment`, same geometry)
    # tick convention (T0): before_step/after_step are called once per physics tick, first tick = 1;
    # schedule/release steps use the 0-based baseline step = tick - 1. ----------------------------------
    def _sample(self):
        from scipy.spatial.transform import Rotation
        finished = self.execution_start_step is not None
        contacts = [(pair, record) for pair, record in self.contact_pairs.items() if record["active"]]
        rotor_roots = {spec["rotor_root"] + "/rotor" for spec in self.T.values()}
        observations, rotations = {}, {}
        for key, spec in self.T.items():
            center = np.asarray(self.center(key), dtype=float)
            rotor_pos, rotor_quat = self.rotor.get_world_pose()
            rotor_rot = Rotation.from_quat([*rotor_quat[1:], rotor_quat[0]]).as_matrix()
            local = rotor_rot.T @ (center - np.asarray(rotor_pos))
            angle = float(self.carousel.get_joint_positions()[0])
            rotations[key] = dict(actual_angle_rad=angle, program_target_rad=float(self.angles[key]))
            planar = Rotation.from_euler("z", angle).as_matrix()
            relative = planar.T @ np.r_[center[:2] - np.asarray(spec["center_xy"]), 0.]
            _, quat = self.blocks[key].get_world_pose()
            body = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
            dimensions = abs(planar.T @ body) @ np.asarray(spec["block_dimensions"])
            root = re.sub(r"/tray\d+/floor$", "", spec["target_floor_prim"])
            if root == spec["target_floor_prim"]:
                raise ValueError("S1 frozen physical slot floor path is unavailable")
            supported_slots = [i for i in range(len(spec['pad_local_xy']))
                               if self.ctx.sim.contacts.supported(spec['block_prim'], root + '/tray%d/floor' % i)]
            contained = [i for i, xy in enumerate(spec['pad_local_xy']) if np.all(
                abs(relative[:2] - np.asarray(xy)) <= np.asarray(spec['tray_clear_dimensions'][i]) / 2)]
            matches = sorted(set(supported_slots) & set(contained))
            slot = matches[0] if len(matches) == 1 else None
            released = not any(_pair_has(pair, spec["block_prim"]) and any("/Robot" in p for p in pair) for pair, _ in contacts)
            carousel_support = any(_pair_has(pair, spec["block_prim"]) and any(_pair_has(pair, r) for r in rotor_roots)
                                   for pair, _ in contacts)
            observations[spec["block_prim"]] = dict(
                position_local=local, inside=slot is not None, support=slot is not None, released=bool(released),
                slot_id="%s/tray%d" % (root, slot) if slot is not None else None, carousel_supported=bool(carousel_support),
                original_entity_slot_match=slot == spec["target_local_index"],
                placed_after_program_finished=bool(self.tracker.placed_after_stop(spec["block_prim"], self.execution_start_step)))
        return observations, finished, rotations


def _setup_stage(stage, tables, manifest):
    """Carousel drive, static station colliders and dynamic blocks, as in the baseline world."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    first = next(iter(tables.values()))
    root = first["rotor_root"]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(True)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root + "/root_joint"))
    joint.CreateLocalPos0Attr(Gf.Vec3f(*first["center_xy"], manifest["station"]["world_support_z"]))
    drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(root + "/spin_joint"), "angular")
    drive.CreateStiffnessAttr(600.)
    drive.CreateDampingAttr(40.)
    drive.CreateMaxForceAttr(20.)
    for prefix in ["/World/Worktable", "/World/TableFrame", "/World/CameraStand", "/World/PedestalLeft", "/World/PedestalRight", "/World/Floor"]:
        top = stage.GetPrimAtPath(prefix)
        if not top:
            continue
        for prim in Usd.PrimRange(top):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(.9)
    pm.CreateDynamicFrictionAttr(.8)
    pm.CreateRestitutionAttr(0.)
    for t in tables.values():
        block = stage.GetPrimAtPath(t["block_prim"])
        for prim in Usd.PrimRange(block):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                col.CreateContactOffsetAttr(.001)
                col.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
        rb = UsdPhysics.RigidBodyAPI.Apply(block)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(block).CreateMassAttr(.08)
        PhysxSchema.PhysxContactReportAPI.Apply(block).CreateThresholdAttr(0.)
        PhysxSchema.PhysxRigidBodyAPI.Apply(block).CreateSolverPositionIterationCountAttr(16)


def build(spec: InstanceSpec, ctx: BuildContext) -> Scene:
    if spec.instance_id.startswith('VAL-S1-') or 'clean_validation_authority' in spec.config:
        from .validation_inputs import validate
        validate(spec, ctx.assets)
    require_physics_hz(spec)
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools
    from scipy.spatial.transform import Rotation

    cfg = timeline_config(spec)
    manifest = json.loads(Path(ctx.assets.resolve("manifest")).read_text())
    tables = {t["name"]: t for t in manifest["tables"]}
    if len(tables) != 4:
        raise ValueError("S1 requires four task blocks, got %d" % len(tables))
    ctx.sim.open_stage(Path(ctx.assets.resolve("scene")).resolve())
    stage = ctx.sim.stage
    _setup_stage(stage, tables, manifest)
    ctx.robot.prepare_stage(stage)  # pinned X5 asset + official configuration (baseline: configure_stage before World)

    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    names = list(tables)
    blocks = {s: world.scene.add(SingleRigidPrim(prim_path=tables[s]["block_prim"], name="Block" + s)) for s in names}
    carousel = world.scene.add(SingleArticulation(prim_path=tables[names[0]]["rotor_root"], name="Carousel"))
    rotor = world.scene.add(SingleRigidPrim(prim_path=tables[names[0]]["rotor_root"] + "/rotor", name="Rotor"))
    ctx.robot.initialize(world)  # registers both X5 arms, world.reset(), rest pose, native verification

    body_offsets = {}
    for s, b in blocks.items():
        pos, quat = b.get_world_pose()
        body_offsets[s] = Rotation.from_quat([*quat[1:], quat[0]]).inv().apply(np.array(tables[s]["block_center_world"]) - pos)

    tracker = SPlacementReleaseTracker({t["block_prim"]: t["block_prim"] for t in tables.values()})
    ledger = CarouselLedger([t["block_prim"] for t in tables.values()])
    scene = SScene(spec, ctx, world, stage, tables, blocks, carousel, rotor, body_offsets, ledger, tracker, cfg)

    def on_contact(headers, data):
        events = []
        for h in headers:
            pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(v)) for v in [h.collider0, h.collider1]))
            if not any("/Robot" in path or any(path.startswith(t["block_prim"] + "/") for t in tables.values()) for path in pair):
                continue
            record = scene.contact_pairs.setdefault(pair, dict(active=False))
            record["active"] = h.type != ContactEventType.CONTACT_LOST
            events.append(dict(colliders=list(pair), active=record["active"]))
        tracker.observe(scene.tick - 1, events)
    scene._subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

    arms = tuple(float(v) for v in ctx.robot.rest_pose()[:6])
    rest = Action(left_arm=arms, right_arm=arms, left_gripper=1.0, right_gripper=1.0)
    for n in range(WARMUP_TICKS):
        ctx.robot.apply(rest)
        ctx.sim.step(render=False)
    scene.contact_pairs.clear()
    return scene
