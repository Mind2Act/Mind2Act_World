"""P1 scene: the single `build(spec, ctx)` shared by every consumer.

Baseline: P-148 `b-P-work-20261025-v6` (`cases_unified/P/world.build`, digest 41f70882ed7d). The prepared scene,
manifest (foods, containers, orders, station, screen texture attribute) and cup/food geometry (.npz) are
instance assets. Food and container physics (convex-hull colliders, 0.06 kg, CCD, contact offsets), order
publication clock and terminal-screen cards follow the baseline. X5 stage/lifecycle belongs to ctx.robot.

Card textures come from the `cards` asset directory (asset resolver), never a manifest-embedded absolute path;
a missing card is an explicit asset gap. Per-phase wall time is kept in `scene.timings` and written to
`p1_timings.json` on close, so slow instances can be attributed (task hook, containment, contacts, ledger).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation, Scene
from mind2act_eval.tasks._observation import observation, require_physics_hz
from .mechanism.cup_geometry import inside_cup, OccupancyVolume
from .mechanism.batched_rays import BatchedCupRays
from .mechanism.occupancy import OccupancyLedger, SAMPLE_DT
from .mechanism.orders import OrderClock

IMPULSE_MIN = 1e-5
REQUIRED_CARDS = ("hard_idle.png", "hard_all_sent.png")


def _pair_owner(path, foods, cups):
    for food, item in foods.items():
        if path.startswith(item["prim_path"] + "/"):
            return food
    for cup, item in cups.items():
        if path.startswith(item["prim_path"] + "/"):
            return "cup:" + cup
    return path


class PScene:
    termination_kind = 'result'

    def __init__(self, spec, ctx, stage, manifest, bodies, offsets, cup_meshes, hulls, cards, screen_attr, clock):
        self.spec, self.ctx, self.stage, self.M = spec, ctx, stage, manifest
        self.bodies, self.offsets, self.cup_meshes, self.hulls = bodies, offsets, cup_meshes, hulls
        from scipy.spatial import ConvexHull
        self.cup_regions = {cup: ConvexHull(np.unique(mesh.vertices[:, :2], axis=0)).equations
                            for cup, mesh in cup_meshes.items()}
        self.cup_queries = {}  # Historical timing counters; scoring no longer casts hull rays.
        self.foods = {x["id"]: x for x in manifest["foods"]}
        self.cups = {x["id"]: x for x in manifest["containers"]}
        self.cards, self.screen_attr, self.clock = cards, screen_attr, clock
        self.ledger = OccupancyLedger(self.foods, SAMPLE_DT)
        self.contact_pairs = {}
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.tick = 0
        self._health = Health(True)
        self._subscription = None

    def _timed(self, name, started):
        self.timings[name] += time.perf_counter() - started

    # -- runner hooks -------------------------------------------------------------------------------
    def set_screen(self, name):
        from pxr import Sdf
        card = self.cards / name
        if not card.is_file():
            raise FileNotFoundError("P1 card texture asset gap: " + str(card))
        self.screen_attr.Set(Sdf.AssetPath(str(card)))

    def before_step(self, tick: int) -> None:
        started = time.perf_counter()
        self.tick = tick
        screen = self.clock.step(tick - 1)
        if screen is not None:
            self.set_screen(screen)
        self._timed("task_hook_s", started)

    def after_step(self, tick: int) -> None:
        try:
            started = time.perf_counter()
            objects = self._measure()
            self._timed("measure_s", started)
            started = time.perf_counter()
            self.ledger.observe(tick, objects, published_demand=self.clock.published_demand(),
                                all_orders_published=self.clock.all_published(tick - 1))
            self._timed("ledger_s", started)
        except ValueError as error:
            self._health = Health(False, "P1 evidence error: " + str(error))

    # -- Scene protocol -----------------------------------------------------------------------------
    def observe(self, tick: int) -> Observation:
        return observation(self.spec, self.ctx.robot, tick, public_instruction(self.spec))

    def evidence(self, tick: int) -> NativeEvidence:
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
                              dict(self.ledger.snapshot(), difficulty=self.spec.difficulty))

    def health(self) -> Health:
        return self._health

    def is_complete(self) -> bool:
        from .scorer import score
        return self.tick > 0 and score(self.evidence(self.tick)).value == 100

    def result_metadata(self) -> dict:
        return {"p1_phase_timings": dict(self.timings, **{k + "_calls": v for k, v in self.counts.items()},
            ray_calls=sum(q.calls for q in self.cup_queries.values()),
            ray_points=sum(q.ray_points for q in self.cup_queries.values()),
            scalar_candidate_fallback_calls=sum(q.fallback_calls for q in self.cup_queries.values()), ticks=self.tick)}

    def is_success(self) -> bool:
        return self.is_complete()

    def close(self) -> None:
        self._subscription = None
        report = self.result_metadata()["p1_phase_timings"]
        (Path(self.ctx.output_dir) / "p1_timings.json").write_text(json.dumps(report, indent=2) + "\n")

    # -- measurement ---------------------------------------------------------------------------------
    def _support(self, food, cup, graph):
        todo, seen = [food], {food}
        while todo:
            for nxt in graph.get(todo.pop(), ()):
                if nxt == "cup:" + cup:
                    return True
                if nxt in self.foods and nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return False

    def _measure(self):
        from scipy.spatial.transform import Rotation
        started = time.perf_counter()
        active = [(pair, r) for pair, r in self.contact_pairs.items() if r["active"]]
        graph = {}
        ledger = self.ctx.sim.contacts
        for pair, (live, impulse, separation) in ledger.pairs.items():
            if not live or not (impulse > 0. or separation <= 0.):
                continue
            u, v = (_pair_owner(p, self.foods, self.cups) for p in pair)
            if u not in self.foods and v not in self.foods:
                continue
            if '/Robot' in pair[0] or '/Robot' in pair[1]:
                continue
            normals = ledger.normals.get(pair)
            if not normals:
                raise ValueError('P1 target support contact lacks measured normal')
            if any(n[2] > 0. for n in normals):
                graph.setdefault(u, set()).add(v)
            if any(n[2] < 0. for n in normals):
                graph.setdefault(v, set()).add(u)
        self._timed("contacts_s", started)
        support_z = self.M["station"]["world_support_z"]
        objects = {}
        for food, body in self.bodies.items():
            position, quat = body.get_world_pose()
            pose = np.r_[position, quat]
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("P1 finite native food pose required")
            center = position + Rotation.from_quat([*quat[1:], quat[0]]).apply(self.offsets[food])
            started = time.perf_counter()
            occupied = [cup for cup, equations in self.cup_regions.items()
                        if np.all(equations[:, :2] @ center[:2] + equations[:, 2] <= 0.)
                        and self._support(food, cup, graph)]
            inside = len(occupied) == 1
            self._timed("containment_s", started)
            self.counts["containment_food_ticks"] += 1
            self.counts["containment_pairs"] += len(self.cups)
            released = not any(any(_pair_owner(p, self.foods, self.cups) == food for p in pair)
                               and any("/Robot" in p for p in pair) for pair, _ in active)
            objects[food] = dict(occupied_containers=occupied, position=center.tolist(), kind=self.foods[food]["type"],
                                 inside=inside,
                                 support=bool(len(occupied) == 1 and self._support(food, occupied[0], graph)),
                                 released=bool(released))
        return objects


def _load_json(ctx, asset_id):
    return json.loads(Path(ctx.assets.resolve(asset_id)).read_text())


def _setup_stage(stage, manifest):
    """Static station/containers and dynamic foods, as in the baseline world."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    for prefix in ["/World/Worktable", "/World/TableFrame", "/World/CameraStand", "/World/PedestalLeft", "/World/PedestalRight",
                   "/World/Floor", "/World/Containers", "/World/OrderTerminal"]:
        top = stage.GetPrimAtPath(prefix)
        if not top:
            continue
        for prim in Usd.PrimRange(top):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr("collision" not in str(prim.GetPath()).lower())
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    material = UsdShade.Material.Define(stage, "/World/FoodPhysics")
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(.9)
    pm.CreateDynamicFrictionAttr(.8)
    pm.CreateRestitutionAttr(0.)
    for item in manifest["foods"]:
        root = stage.GetPrimAtPath(item["prim_path"])
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                auxiliary = "collision" in str(prim.GetPath()).lower()
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(not auxiliary)
                if not auxiliary:
                    if prim.IsA(UsdGeom.Mesh):
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                    cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    cp.CreateContactOffsetAttr(.001)
                    cp.CreateRestOffsetAttr(0.)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        rb = UsdPhysics.RigidBodyAPI.Apply(root)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.06)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        body.CreateEnableCCDAttr(True)
        body.CreateMaxDepenetrationVelocityAttr(.5)
        body.CreateSolverPositionIterationCountAttr(64)
        body.CreateSolverVelocityIterationCountAttr(8)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(True)
        if str(prim.GetPath()).startswith("/World/Containers/") and prim.HasAPI(UsdPhysics.CollisionAPI):
            cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            cp.CreateContactOffsetAttr(.003)
            cp.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics",
                                                         bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def build(spec: InstanceSpec, ctx: BuildContext) -> Scene:
    if spec.instance_id.startswith('VAL-P1-') or 'clean_validation_authority' in spec.config:
        from .validation_inputs import validate
        validate(spec, ctx.assets)
    require_physics_hz(spec)
    import trimesh
    from isaacsim.core.prims import SingleRigidPrim
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools
    from scipy.spatial.transform import Rotation

    manifest = _load_json(ctx, "manifest")
    cards = Path(ctx.assets.resolve("cards"))
    missing = [n for n in REQUIRED_CARDS + tuple("hard_order%d.png" % i for i in range(len(manifest["case"]["orders"])))
               if not (cards / n).is_file()]
    if missing:
        raise FileNotFoundError("P1 card texture asset gap (no fallback): " + ",".join(missing))
    ctx.sim.open_stage(Path(ctx.assets.resolve("scene")).resolve())
    stage = ctx.sim.stage
    # SDK-owned live RenderProducts share /Render with source opinions.
    ctx.robot.prepare_stage(stage)
    ctx.robot.attach_cameras(stage)
    _setup_stage(stage, manifest)

    foods = {x["id"]: x for x in manifest["foods"]}
    cups = {x["id"]: x for x in manifest["containers"]}
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    bodies = {f: world.scene.add(SingleRigidPrim(prim_path=x["prim_path"], name=f)) for f, x in foods.items()}
    ctx.robot.initialize(world)  # registers both X5 arms, world.reset(), rest pose, native verification

    with np.load(ctx.assets.resolve("geometry"), allow_pickle=False) as f:
        geometry = {k: f[k] for k in f.files}
    cup_meshes = {c: trimesh.Trimesh(vertices=geometry["cup_%s_vertices" % c], faces=geometry["cup_%s_faces" % c], process=False)
                  for c in cups}
    hulls = {f: geometry["food_" + f] for f in foods}
    offsets = {}
    for food, body in bodies.items():
        pos, q = body.get_world_pose()
        offsets[food] = Rotation.from_quat([*q[1:], q[0]]).inv().apply(np.asarray(foods[food]["center"]) - pos)

    clock = OrderClock(manifest["case"]["orders"])
    scene = PScene(spec, ctx, stage, manifest, bodies, offsets, cup_meshes, hulls, cards,
                   stage.GetAttributeAtPath(manifest["screen_texture_attr"]), clock)
    scene.set_screen("hard_idle.png")

    def on_contact(headers, data):
        for h in headers:
            pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(v)) for v in [h.collider0, h.collider1]))
            if not any("/Robot" in p or any(p.startswith(x["prim_path"] + "/") for x in foods.values()) for p in pair):
                continue
            r = scene.contact_pairs.setdefault(pair, dict(active=False, max_impulse_Ns=0.))
            r["active"] = h.type != ContactEventType.CONTACT_LOST
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                r["max_impulse_Ns"] = max(r["max_impulse_Ns"], float(np.linalg.norm(data[i].impulse)))
    scene._subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    return scene
