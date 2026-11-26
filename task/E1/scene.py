"""E1 scene: the single `build(spec, ctx)` shared by every consumer.

Baseline: E-142 (`e-current-reconstruction-camera-v2`, development_world.create, digest 7c986b91653b).
The prepared scene (scene.usda / manifest.json / plan.json / alignment.json) is an *asset* of the
instance, produced offline by the E1 preparation tool from the recipe. build() never aligns the
scene to a training scene (no Carrier/SubmissionTray removal, no "Carrier must exist" check).
Robot stage setup and articulation control belong to ctx.robot (shared X5); nothing here sets robot
gains, forces or gripper parameters.

Isaac imports are lazy so the module imports on CPU. GPU behaviour is verified by T4 replay only.
"""
import copy
import json
from pathlib import Path

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import Action, BuildContext, Health, InstanceSpec, NativeEvidence, Observation, Scene
from mind2act_eval.tasks._observation import observation, require_physics_hz
from .evidence import Evidence
from .scorer import N_BY_DIFFICULTY
from .mechanism import stage_prep
from .mechanism.timeline import PHYSICS_HZ, Timeline
from .mechanism.transport import build_transport

ASSET_IDS = ("scene", "manifest", "plan", "alignment")


def _load(ctx: BuildContext, asset_id: str):
    path = ctx.assets.resolve(asset_id)
    return path if asset_id == "scene" else json.loads(Path(path).read_text())


class EScene:
    termination_kind = 'events'

    """Scene protocol implementation. The runner steps physics; it must call before_step/after_step."""

    def __init__(self, spec, ctx, stage, manifest, transport, intervals, objects, initial_poses, offsets,
                 submission_z_bounds, evidence):
        self.spec, self.ctx, self.stage, self.manifest = spec, ctx, stage, manifest
        self.transport, self.intervals = transport, intervals
        self.objects, self.initial_poses, self.offsets = objects, initial_poses, offsets
        self.submission_z_bounds = submission_z_bounds
        self.evidence_recorder = evidence
        self.contact_pairs = {}
        self.latest = None
        self.tick = 0
        self.timeline = Timeline(transport, manifest["rounds"], manifest["case"]["timing"], intervals,
                                 transport.carrier_x, evidence)
        self._health = Health(True)
        self._subscription = None

    # -- runner hooks -------------------------------------------------------------------------------
    def before_step(self, tick: int) -> None:
        self.tick = tick  # T0 convention: once per physics tick, first tick = 1; baseline step = tick - 1
        self.timeline.before_tick(tick - 1)
        self.transport.before_step((tick - 1) / PHYSICS_HZ)

    def after_step(self, tick: int) -> None:
        self.transport.after_step((tick - 1) / PHYSICS_HZ, "scene")
        measured = self._measure()
        measured["physics_tick"] = tick
        try:
            self.timeline.after_tick(tick - 1, measured)
        except ValueError as error:  # invalid measured evidence is an infrastructure fault, not a score
            self._health = Health(False, "E1 evidence error: " + str(error))
        else:
            self.latest = measured

    # -- Scene protocol -----------------------------------------------------------------------------
    def observe(self, tick: int) -> Observation:
        return observation(self.spec, self.ctx.robot, tick, public_instruction(self.spec))

    def evidence(self, tick: int) -> NativeEvidence:
        return self.evidence_recorder.native_evidence(self.spec.task_id, self.spec.instance_id, tick)

    def health(self) -> Health:
        return self._health

    def is_complete(self) -> bool:
        rec = getattr(self, 'evidence_recorder', None)
        return rec is not None and len(rec.results) >= N_BY_DIFFICULTY[self.spec.difficulty]

    def events_complete(self) -> bool:
        return self.is_complete() or bool(self._settlement_reason() and self._has_credit())

    def _has_credit(self):
        return any(row['pass'] for row in self.evidence_recorder.results)

    def _settlement_reason(self):
        row = getattr(self, 'latest', None)
        if not row or row['phase'] != 'batch_execute':
            return None
        rn = row['round']
        if not 1 <= rn <= N_BY_DIFFICULTY[self.spec.difficulty] or len(self.evidence_recorder.results) >= rn:
            return None
        items = row['items']
        if not items:
            return None
        dropped = {item['path'] for item in items
                   if item['center'][2] < row['z_bounds'][0] - .40 and not item['touch_robot']}
        if len(dropped) == len(items):
            return 'active_batch_items_dropped'
        if rn == N_BY_DIFFICULTY[self.spec.difficulty]:
            target = next(item['path'] for item in items if item['identity'] == self.evidence_recorder.answers[rn])
            if target in dropped or self.evidence_recorder.registered.get(rn, set()) - {target}:
                return 'final_batch_failed'
        return None

    def irreversible_failure(self) -> str | None:
        reason = self._settlement_reason()
        return reason if reason and not self._has_credit() else None

    def close(self) -> None:
        if self._subscription is not None:
            self._subscription = None
        self.transport.finish(Path(self.ctx.output_dir))

    # -- measurement (sdk_world.measure) ----------------------------------------------------------
    def _measure(self) -> dict:
        from pxr import Usd, UsdGeom
        from scipy.spatial.transform import Rotation
        rn = self.timeline.round
        layout = self.manifest["case"]["layout"]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        contacts = [dict(paths=list(pair)) for pair, r in self.contact_pairs.items() if r["active"]]
        submission_floor = self.transport.submission_floors[rn]
        items = []
        for item in self.manifest["rounds"][rn]["items"]:
            path = item["prim_path"]
            bounds = cache.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
            center, dims = np.asarray(bounds.GetMidpoint()), np.asarray(bounds.GetSize())
            touching = [p for p in contacts if any(x == path or x.startswith(path + "/") for x in p["paths"])]
            finger = lambda side, link: any(any("/Robot" + side + "/Asset/" + link + "/" in x for x in p["paths"]) for p in touching)
            pos, quat = self.initial_poses[path]
            baseline = pos + Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).apply(self.offsets[path])
            items.append(dict(
                path=path, identity=item["visible_identity"], center=center.tolist(), dimensions=dims.tolist(),
                baseline_z=float(baseline[2]),
                bilateral=any(finger(s, "link7") and finger(s, "link8") for s in ("Left", "Right")),
                touch_robot=any(any("/Robot" in x for x in p["paths"]) for p in touching),
                support=(submission_floor is not None and
                         self.ctx.sim.contacts.supported(path, submission_floor))))
        return dict(
            items=items, contacts=contacts,
            goal_xy=(np.asarray(layout["submission_center_xy"]) + [self.transport.carrier_x[rn], layout["carrier_center"][1]]).tolist(),
            inner_xy=(np.asarray(layout["submission_size_xy"]) / 2 - .018).tolist(),
            z_bounds=list(self.submission_z_bounds))


def build(spec: InstanceSpec, ctx: BuildContext) -> Scene:
    if spec.instance_id.startswith('VAL-E1-') or 'clean_validation_authority' in spec.config:
        from .validation_inputs import validate
        validate(spec, ctx.assets)
    require_physics_hz(spec)
    from pxr import Usd, UsdPhysics
    from scipy.spatial.transform import Rotation
    from isaacsim.core.prims import SingleRigidPrim
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools
    import omni.physics.tensors as tensors
    from .submission import submission_height_bounds

    manifest = copy.deepcopy(_load(ctx, "manifest"))
    plan = _load(ctx, "plan")
    alignment = _load(ctx, "alignment")
    scene_path = _load(ctx, "scene")
    if spec.difficulty != manifest["case"]["difficulty"].lower():
        raise ValueError("E1 resolved difficulty %s does not match prepared scene %s" % (spec.difficulty, manifest["case"]["difficulty"]))

    ctx.sim.open_stage(Path(scene_path).resolve())
    stage = ctx.sim.stage
    intervals = stage_prep.compute_visible_intervals(stage, manifest)
    ctx.robot.prepare_stage(stage)  # pinned X5 asset + official configuration, as in the baseline (after intervals, before transport)
    transport = build_transport(stage, manifest, plan, str(scene_path), intervals)
    stage_prep.align_colliders(stage, alignment)
    submission_z_bounds = submission_height_bounds(plan, manifest["station"])
    stage_prep.make_station_static(stage)
    all_items = [i for r in manifest["rounds"] for i in r["items"]]
    stage_prep.make_items_dynamic(stage, all_items)

    from mind2act_eval.variations import prepare_station
    prepare_station(ctx, table_surfaces=())  # source E conveyor station has no tabletop
    world = ctx.world
    objects = {item["prim_path"]: SingleRigidPrim(prim_path=item["prim_path"], name="Item" + str(index))
               for index, item in enumerate(all_items)}
    ctx.robot.initialize(world)  # registers both X5 arms, world.reset(), rest pose, native verification
    for obj in objects.values():
        obj.initialize()
    view = tensors.create_simulation_view("numpy")
    view.set_subspace_roots("/")

    initial_poses = {p: obj.get_world_pose() for p, obj in objects.items()}
    for item in all_items:
        p = item["prim_path"]
        rn = int(p.split("Batch")[1].split("/")[0])
        pos, q = initial_poses[p]
        initial_poses[p] = (np.asarray(pos) - [transport.initial_x[rn], 0, .004], q)
    transport.initialize(view)

    offsets = {}
    for item in all_items:
        p = item["prim_path"]
        pos, q = initial_poses[p]
        rotation = Rotation.from_quat([*q[1:], q[0]])
        rn = int(p.split("Batch")[1].split("/")[0])
        xy = np.array(item["center_xy_world"])
        xy[0] -= manifest["rounds"][rn]["observation_center_offset_x"]
        offsets[p] = rotation.inv().apply(np.r_[xy, item["bottom_z_world"] + item["dimensions_m"][2] / 2] - pos)

    arms = tuple(float(v) for v in ctx.robot.rest_pose()[:6])
    rest = Action(left_arm=arms, right_arm=arms, left_gripper=1.0, right_gripper=1.0)
    transport.settle(lambda: ctx.robot.apply(rest), lambda: ctx.sim.step(render=False))

    evidence = Evidence(manifest["private_answer_identities"], spec.difficulty)
    scene = EScene(spec, ctx, stage, manifest, transport, intervals, objects, initial_poses, offsets,
                   submission_z_bounds, evidence)

    def on_contact(headers, data):
        for h in headers:
            pair = tuple(sorted([str(PhysicsSchemaTools.intToSdfPath(h.collider0)),
                                 str(PhysicsSchemaTools.intToSdfPath(h.collider1))]))
            if not any("/Robot" in p or "/Item" in p for p in pair):
                continue
            r = scene.contact_pairs.setdefault(pair, dict(active=False))
            r["active"] = h.type != ContactEventType.CONTACT_LOST
    scene._subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    return scene
