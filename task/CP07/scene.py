"""CP07 shared-world prepared-scene port; native GPU validation is pending.

Only physical sensors and the source's four car wheel drives are task-owned.
Robot setup/actions, clocks, cameras and sealing remain shared framework owners.
"""
import itertools

from mind2act_eval.contracts import Health, NativeEvidence
from mind2act_eval.tasks._observation import observation
from .binding import ProfileBlocked, load
from .producer import BridgeFacts
from .scorer import score

INSTRUCTIONS = {
    "easy": "Build a stable bridge across the gap with continuous wheel paths on both sides. Release the bridge pieces, then press and release the button and let the car drive across to the opposite bank.",
    "medium": "Build a stable bridge across the gap with continuous wheel paths on both sides. Choose suitable pieces by shape and size from the stock, which includes extra blocks, and leave unused blocks aside. Release the bridge pieces, then press and release the button and let the car drive across to the opposite bank.",
}
INSTRUCTIONS["hard"] = INSTRUCTIONS["medium"]


class BridgeScene:
    def __init__(self, spec, ctx, binding, sensors):
        self.spec, self.ctx, self.binding, self.sensors = spec, ctx, binding, sensors
        self.facts = BridgeFacts(binding, sensors.read())
        self._arrival_brake_tick = None

    def before_step(self, tick):
        pass

    def after_step(self, tick):
        measured = self.sensors.read()
        launch = self.facts.after_step(tick, measured)
        if launch:self.sensors.drive(True)
        if self.facts.snapshot()['car_reached'] and self._arrival_brake_tick is None:
            # Preserve the task-owned bank brake, while continuing real pose
            # measurements: a later displacement must change the T0 result.
            self.sensors.drive(False)
            self._arrival_brake_tick=tick
        if (self.facts.launched and self._arrival_brake_tick is None and self.binding['mode']!='source_bound_test'
                and self.spec.difficulty != "easy" and tick % 6 == 0):
            car = measured["car"]
            correction = max(-50., min(50., 4500 * (car["p_m"][1] - self.binding["lane_y_m"])
                                      + 140 * car["heading_rad"]))
            self.sensors.drive(True, correction)

    def observe(self, tick):
        return observation(self.spec, self.ctx.robot, tick, INSTRUCTIONS[self.spec.difficulty])

    def evidence(self, tick):
        if tick != self.facts.tick:
            raise ValueError("CP07 evidence tick mismatch")
        values = self.facts.snapshot()
        if self.spec.config.get('layout_split')=='validation':
            from mind2act_eval.contracts import to_json
            values['validation_source_spec']=to_json(self.spec)
        if 'train_source' in self.spec.config:
            from mind2act_eval.contracts import to_json
            values.update(seed=self.spec.seed,train_source=to_json(self.spec.config['train_source']),
                          source_layout_sha256=self.spec.config['source_layout_sha256'])
        return NativeEvidence("CP07", self.spec.instance_id, tick, values)

    def is_success(self):
        return bool(self.facts.sampled and self.facts.snapshot()['car_reached'])

    is_complete = is_success

    def health(self):
        return Health(True)

    def result_metadata(self):
        return dict(binding_mode=self.binding["mode"], formal_evaluation_profile="not_claimed",
                    solutions_scope=self.binding["solutions_scope"],
                    parking_basis="source x/z/up and vehicle/lane y geometry; heading tolerance not authored",
                    completion_policy="actual_car_at_t0",
                    arrival_brake_tick=self._arrival_brake_tick, success_lock_tick=None,
                    initialization_physics_ticks=180, source_code_sha256=self.binding["source_code_sha256"],
                    source_manifest_sha256=self.binding["source_manifest_sha256"])

    def close(self):
        self.sensors.drive(False)


class NativeSensors:
    def __init__(self, ctx, binding):
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        from isaacsim.core.prims import SingleRigidPrim
        self.np, self.ctx, self.binding = np, ctx, binding
        stage = ctx.sim.stage
        ctx.robot.prepare_stage(stage)
        ctx.robot.attach_cameras(stage)
        from mind2act_eval.variations import prepare_station
        prepare_station(ctx)
        world = ctx.world
        self.arms = ctx.robot.articulations(world)
        self.contacts = ctx.sim.contacts
        self.bodies = {}
        def rigid(path, name):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("CP07 required native prim missing: " + path)
            return world.scene.add(SingleRigidPrim(prim_path=path, name=name))
        def task_body(path, name):
            if path.startswith("/World/Robot"):
                raise ValueError("CP07 contact authoring is limited to task bodies")
            body = rigid(path, name)
            prim = stage.GetPrimAtPath(path)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            return body
        for piece in binding["pieces"]:
            self.bodies[piece["id"]] = task_body(piece["path"], "cp07_" + piece["id"])
        self.car = task_body("/World/Car/Chassis", "cp07_car")
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Car')):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        self.button = task_body("/World/StartButton", "cp07_button")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bank=stage.GetPrimAtPath('/World/Terrain/BankRight')
        if not bank:raise ValueError('CP07 actual opposite bank geometry missing')
        bank_range=cache.ComputeWorldBound(bank).ComputeAlignedRange()
        self.bank_lower=np.asarray(bank_range.GetMin())[:2]
        self.bank_upper=np.asarray(bank_range.GetMax())[:2]
        car_bounds=cache.ComputeUntransformedBound(stage.GetPrimAtPath('/World/Car/Chassis')).ComputeAlignedRange()
        self.car_local_center=(np.asarray(car_bounds.GetMin())+np.asarray(car_bounds.GetMax()))/2
        self.piece_corners = {}
        for piece in binding["pieces"]:
            bounds = cache.ComputeUntransformedBound(stage.GetPrimAtPath(piece["path"])).ComputeAlignedRange()
            self.piece_corners[piece["id"]] = np.array(list(itertools.product(*zip(bounds.GetMin(), bounds.GetMax()))))
        from mind2act_eval.robot.read_only import robot_body_readers
        hand_paths = {"cp07_"+side+link: "/World/Robot"+side+"/Asset/"+link
                      for side in ("Left", "Right") for link in ("link6", "link7", "link8", "camera_base", "camera")}
        hand_readers = robot_body_readers(stage, hand_paths, world=world)
        self.hands = []
        for side in ("Left", "Right"):
            for link in ("link6", "link7", "link8", "camera_base", "camera"):
                path = "/World/Robot" + side + "/Asset/" + link
                body = hand_readers["cp07_"+side+link]
                bounds = cache.ComputeUntransformedBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
                corners = np.array(list(itertools.product(*zip(bounds.GetMin(), bounds.GetMax()))))
                self.hands.append((body, corners))
        self.wheels = [UsdPhysics.DriveAPI(p, "angular")
                       for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Car"))
                       if p.IsA(UsdPhysics.RevoluteJoint)]
        if len(self.wheels) != 4 or any(not d.GetTargetVelocityAttr() for d in self.wheels):
            raise ValueError("CP07 four authored wheel drives missing")
        self.drive(False)
        world.reset()
        ctx.robot.reset_to_rest()
        ctx.robot.verify(self.arms, stage)
        # Source initial settle was an expert Wait(1.5). It is initialization
        # here, before first observation/task clock, independent of any expert.
        for _ in range(180):
            world.step(render=False)
        self.contacts.clear()
        self.button_rest = float(self.button.get_world_pose()[0][2])
        self.support_paths = set(p for s in binding["solutions"] for slot in s["slots"] for p in slot["support_paths"])
        self.support_paths.update(p["path"] for p in binding["pieces"])
        for path in self.support_paths:
            if not stage.GetPrimAtPath(path):
                raise ValueError("CP07 native support prim missing: " + path)

    @staticmethod
    def rotation(q):
        # Isaac rigid poses are world-frame metres, quaternion wxyz.
        from scipy.spatial.transform import Rotation
        return Rotation.from_quat([*q[1:], q[0]]).as_matrix()

    def drive(self, on, correction=0.):
        for drive in self.wheels:
            side = 1 if drive.GetPrim().GetName().endswith("1") else -1
            drive.GetTargetVelocityAttr().Set(160 + side * correction if on else 0)

    def read(self):
        np, b = self.np, self.binding
        pieces = {}
        for piece in b["pieces"]:
            path = piece["path"]
            p, q = self.bodies[piece["id"]].get_world_pose()
            corners = self.piece_corners[piece["id"]] @ self.rotation(q).T + p
            pieces[piece["id"]] = dict(p_m=p.tolist(), q_wxyz=q.tolist(),
                bilateral=self.contacts.bilateral(path), robot_contact=self.contacts.touch(path, robot_only=True),
                support_contacts=[other for other in self.support_paths if other != path and self.contacts.touch(path, other=other)],
                inside_bridge=bool(np.all(corners.max(0) >= b["bridge_bounds_m"]["lo"])
                                   and np.all(corners.min(0) <= b["bridge_bounds_m"]["hi"])))
        bp = self.button.get_world_pose()[0]
        lo = np.array([*b["button_xy_m"], self.button_rest]) + [-.044, -.044, -.010]
        hi = np.array([*b["button_xy_m"], self.button_rest]) + [.044, .044, .044]
        clo = np.array([b["car_start_x_m"]-.045, b["lane_y_m"]-.050, .790])
        chi = np.array([b["finish_x_m"]+.045, b["lane_y_m"]+.050, b["roadtop_end_m"]+.070])
        clear = True
        for body, corners in self.hands:
            p, q = body.get_world_pose()
            v = corners @ self.rotation(q).T + p
            low, high = v.min(0), v.max(0)
            clear &= not ((np.all(high > lo) and np.all(low < hi))
                          or (np.all(high > clo) and np.all(low < chi)))
        p, q = self.car.get_world_pose()
        rotation = self.rotation(q)
        car_center=np.asarray(p)+rotation@self.car_local_center
        bank_supported=bool(self.contacts.supported('/World/Car',other='/World/Terrain/BankRight'))
        bank_inside=bool(np.all(car_center[:2]>=self.bank_lower) and np.all(car_center[:2]<=self.bank_upper))
        return dict(pieces=pieces, button_z_m=float(bp[2]),
            button_robot_contact=self.contacts.touch("/World/StartButton", robot_only=True),
            hands_clear=bool(clear), joint_error_rad=None,
            car=dict(p_m=p.tolist(),far_bank_supported=bank_supported,far_bank_center_inside=bank_inside,
                     center_world_m=car_center.tolist(),bank_bounds_xy_m=[self.bank_lower.tolist(),self.bank_upper.tolist()], up=float(rotation[2, 2]),
                     heading_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0]))))


def build(spec, ctx):
    from mind2act_eval.public_layouts import validate
    validate(spec)
    mode = spec.config.get("cp07_mode")
    if mode == "train_layout_test":
        from mind2act_eval.cp07_layout_identity import validate_identity
        validate_identity(spec)
    elif mode not in ("diagnostic_training","source_bound_test","independent_validation"):
        raise ProfileBlocked("CP07 requires explicit diagnostic_training source binding")
    binding = load(spec, ctx.assets)
    return BridgeScene(spec, ctx, binding, NativeSensors(ctx, binding))
