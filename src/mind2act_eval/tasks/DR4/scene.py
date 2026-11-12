"""Original seeded DR4 geometry and physics on the framework simulator/X5."""
import hashlib
import numpy as np

from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, Observation
from .instances import validate
from .evidence import Recorder, rotation


def _rigid_bodies(world, ball_count):
    from isaacsim.core.prims import SingleRigidPrim
    paths = dict(tray='/World/PR7/Tray', **{'Ball'+str(i): '/World/PR7/Ball'+str(i) for i in range(ball_count)})
    return {name: world.scene.add(SingleRigidPrim(path, name='DR4_'+name)) for name, path in paths.items()}


def _flanges(stage):
    from pxr import UsdPhysics
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.prims import is_prim_non_root_articulation_link
    paths = {s:'/World/Robot'+s+'/Asset/link6' for s in ('Left','Right')}
    # IsaacSim v5.1.0 SingleRigidPrim does not expose sensor preparation:
    # its underlying RigidPrim defaults to authoring sleepThreshold=0.
    # Require existing non-root links before any SDK construction, so neither
    # missing APIs nor SDK post-reset pose/velocity handling can write robots.
    for path in paths.values():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not is_prim_non_root_articulation_link(prim_path=path):
            raise ValueError('DR4 flange reader requires an existing non-root rigid articulation link')
    views = {s: RigidPrim(prim_paths_expr=path, name='DR4_tcp_'+s,
        reset_xform_properties=False, prepare_contact_sensors=False, track_contact_forces=False,
        disable_stablization=False, contact_filter_prim_paths_expr=[]) for s,path in paths.items()}
    for view in views.values():
        view.initialize()
    return views


class TrayScene:
    def __init__(self, spec, ctx, bodies, flanges, contacts, subscription, geometry, seeded):
        self.spec, self.ctx, self.bodies, self.flanges = spec, ctx, bodies, flanges
        self.contacts, self.subscription, self.seeded = contacts, subscription, seeded
        self._health = Health(True)
        initial = self.measure(0)
        self.recorder = Recorder('DR4', spec.instance_id, spec.config['source_case'], geometry, initial,
                                trace_path=ctx.output_dir/'DR4'/'raw-trace.jsonl')

    def measure(self, tick):
        tray = self.bodies['tray']
        p, q = tray.get_world_pose()
        tcps = {}
        for side, view in self.flanges.items():
            positions, orientations = view.get_world_poses()
            position, orientation = positions[0], orientations[0]
            tcps[side] = (np.asarray(position) + rotation(orientation) @ np.array([.145,0.,0.])).tolist()
        return dict(tick=tick, time_s=float(self.ctx.world.current_time), tray_position=np.asarray(p).tolist(),
            tray_rotation_wxyz=np.asarray(q).tolist(), tray_linear_velocity=np.asarray(tray.get_linear_velocity()).tolist(),
            tray_angular_velocity=np.asarray(tray.get_angular_velocity()).tolist(),
            balls={name: dict(position=np.asarray(body.get_world_pose()[0]).tolist(),
                              linear_velocity=np.asarray(body.get_linear_velocity()).tolist())
                   for name, body in self.bodies.items() if name != 'tray'},
            tcps=tcps, contacts=self.contacts.snapshot())

    def before_step(self, tick):
        self.contacts.clear()

    def after_step(self, tick):
        try:
            self.recorder.record(self.measure(tick))
        except ValueError as error:
            self._health = Health(False, str(error))
            raise

    def observe(self, tick):
        return Observation(tick=tick, sim_time_s=tick/self.spec.clock.physics_hz,
            instruction=self.spec.config['instruction'], protocol_version=self.spec.protocol_version,
            **dict(self.ctx.robot.observe()))

    def evidence(self, tick):
        return self.recorder.evidence(tick)

    def health(self):
        return self._health

    def is_complete(self):
        return self.is_success()

    def is_success(self):
        return bool(self.recorder.facts()['success'])

    def result_metadata(self):
        return dict(source_id=self.spec.source_id, native_task_alias='PR7',
            source_spec_sha256=self.spec.config['source_spec_sha256'],
            source_config_sha256=self.spec.config['source_config_sha256'],
            source_task_condition_sha256=self.spec.config['source_case']['task_condition_sha256'],
            task_condition_content_sha256=self.spec.config['task_condition_content_sha256'],
            scene_sha256=self.spec.config['scene_sha256'], bootstrap_physics_ticks=240,
            seeded_scene=self.seeded, physical_tuning_applied=False)

    def close(self):
        try:
            if self.subscription is not None:
                self.subscription.unsubscribe()
        finally:
            self.subscription = None
            self.recorder.close()


def build(spec: InstanceSpec, ctx: BuildContext):
    from pxr import UsdGeom
    from .mechanism.seeded import apply
    from .mechanism.contacts import ContactReadings, subscribe
    validate(spec)
    from mind2act_eval.sim.source_assets import verify_source_file
    verify_source_file(ctx.assets, 'station', spec.config['scene_sha256'], error='DR4 selected prepared scene changed')
    stage = ctx.sim.stage
    seeded = apply(stage, spec)  # exact original condition offsets, before reset; never a resampled scene
    shell = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath('/World/PR7/Tray/Shell')).GetPointsAttr().Get())
    if shell.shape != (240,3):
        raise ValueError('DR4 original tray shell geometry missing')
    condition = spec.config['task_condition']
    geometry = dict(polygon=shell[96:144,:2].tolist(), floor_z=float(shell[192,2]), rim_z=float(shell[144,2]),
        goal=[condition['goal_offset_xy_m'][0], .15+condition['goal_offset_xy_m'][1], spec.config['source_support_z_m']+.016])
    ctx.robot.prepare_stage(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    bodies = _rigid_bodies(ctx.world, spec.config['source_case']['ball_count'])
    ctx.robot.initialize(ctx.world)
    for body in bodies.values():
        body.initialize()
    # Original source settles 240 real physics ticks before the evaluation reset anchor.
    for _ in range(240):
        ctx.sim.step(render=False)
    flanges = _flanges(stage)
    contacts = ContactReadings()
    subscription = subscribe(contacts)
    try:
        return TrayScene(spec, ctx, bodies, flanges, contacts, subscription, geometry, seeded)
    except BaseException:
        subscription.unsubscribe()
        raise
