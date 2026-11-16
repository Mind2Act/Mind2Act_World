"""DR1 build(spec, ctx): prepared FR1 scene + belt/parcel mechanism on the shared X5. No policy name here."""
import hashlib
import math

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import (Action, BuildContext, Health, InstanceSpec, NativeEvidence,
                                     Observation)

from .instances import layout, require_baseline_clock
from .scorer import parcel_state


def validate(spec: InstanceSpec) -> None:
    lay = layout()
    require_baseline_clock(spec.clock)
    count = spec.config['parameters']['count']
    if spec.task_id != 'DR1' or count != lay['count_by_difficulty'][spec.difficulty]:
        raise ValueError('not a consistent DR1 spec')
    if len(spec.config['source_recipe']['items']) != count or len(spec.config['stream_goods']) != count:
        raise ValueError('DR1 recipe/goods count differs from the tier count')


def validate_station_source(spec, assets):
    validate(spec)
    from mind2act_eval.sim.source_assets import source_file
    return source_file(assets, 'station')


class ConveyorScene:
    def __init__(self, spec, ctx, handles, bodies, recorder, contacts, activity, subscription, tray, stage):
        self.spec, self._ctx, self._h, self._bodies = spec, ctx, handles, bodies
        self._recorder, self._contacts, self._active = recorder, contacts, activity
        self._subscription, self._tray, self._stage = subscription, tray, stage
        self._pose_set = [0] * len(bodies)
        self.instruction = public_instruction(spec)

    def before_step(self, tick: int) -> None:
        if tick != 1:
            return
        from . import mechanism
        for item in self.spec.config['source_recipe']['items']:
            i = item['index']
            if self._pose_set[i]:
                raise RuntimeError('original FR1 parcel already placed')
            mechanism.release_parcel(self._stage, self._h, self._bodies, item, i)
            self._active[i] = True
            self._pose_set[i] += 1

    def after_step(self, tick: int) -> None:
        self._recorder.record(tick)

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz,
                           instruction=self.instruction, protocol_version=self.spec.protocol_version,
                           **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        from pxr import UsdGeom
        h = self._h
        cache = UsdGeom.XformCache()
        import numpy as np
        matrix_error = max((float(np.max(np.abs(np.asarray(cache.GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(p))) - m))) for p, m in h.fixed_matrices0.items()), default=0.)
        now = [float(h.head_camera.GetFocalLengthAttr().Get()), float(h.head_camera.GetHorizontalApertureAttr().Get()),
               float(h.head_camera.GetVerticalApertureAttr().Get())]
        cfg = self.spec.config
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick, dict(
            ticks=self._recorder.ticks, count=len(self._bodies), tray_bounds_m=self._tray,
            belt_speed_m_s=float(cfg['parameters']['speed_m_s']),
            station=dict(matrix_max_error=matrix_error,
                         head_intrinsics_max_error=float(np.max(np.abs(np.asarray(now) - h.camera_intrinsics0))),
                         pose_set_count=list(self._pose_set)),
            contact_pairs=self._contacts.summary()))

    def health(self) -> Health:
        for i, body in enumerate(self._bodies):
            position, _ = body.get_world_pose()
            if not all(math.isfinite(float(x)) for x in position):
                return Health(False, f'parcel {i} pose is not finite')
            kinematic = self._h.rbapis[i].GetKinematicEnabledAttr().Get()
            if type(kinematic) is not bool:
                return Health(False, f'parcel {i} kinematic flag unreadable ({kinematic!r})')
            if kinematic == bool(self._active[i]):
                return Health(False, f'parcel {i} dynamic role inconsistent with activation')
        return Health(True)

    def is_success(self) -> bool:
        """All parcels must currently have real target support and centers inside."""
        ticks, count = self._recorder.ticks, len(self._bodies)
        if not ticks or not count:
            return False
        if len(ticks[-1]['parcels']) != count:
            raise ValueError('DR1 completion requires every parcel measurement')
        return all(parcel_state(ticks, i, self._tray)['placed'] for i in range(count))

    def is_complete(self) -> bool:
        return self.is_success()

    def close(self) -> None:
        unsubscribe = getattr(self._subscription, 'unsubscribe', None)
        if callable(unsubscribe):
            unsubscribe()


def build(spec: InstanceSpec, ctx: BuildContext) -> ConveyorScene:
    import numpy as np
    from . import mechanism
    from .evidence import ContactTable, Recorder
    scene_path = validate_station_source(spec, ctx.assets)
    lay = layout()
    stage = ctx.sim.stage
    ctx.robot.restore_asset(stage)  # pinned X5 replaces the prepared legacy robot, mounts preserved
    handles = mechanism.author_scene(stage, spec, lay)
    mechanism.snapshot_fixed(stage, lay, handles)
    from isaacsim.core.prims import SingleRigidPrim
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    bodies = [world.scene.add(SingleRigidPrim(path, name='StreamGoods%02d' % i))
              for i, path in enumerate(handles.package_roots)]
    ctx.robot.initialize(world)
    rest = [math.radians(d) for d in lay['rest_joints_deg']]
    action = Action(rest, rest, lay['rest_gripper'], lay['rest_gripper'])
    for _ in range(lay['warmup_ticks']):  # original 120 initialization ticks precede task time
        ctx.robot.apply(action)
        ctx.sim.step(render=False)
    n = len(bodies)
    activity = [False] * n
    contacts = ContactTable(handles.package_roots)
    recorder = Recorder(bodies, handles.dims, contacts, activity, mechanism.parcel_extent,
                        lambda: list(handles.speed_attr.Get()))
    subscription = mechanism.subscribe_contacts(contacts)
    tray = spec.config['tray_bounds_m']
    return ConveyorScene(spec, ctx, handles, bodies, recorder, contacts, activity, subscription, tray, stage)
