"""M1 build(spec, ctx): prepared M station, compiled environment choreography, native tracker for termination."""
import math

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation

from .evidence import ContactTable, Recorder
from .instances import layout, plain, require_baseline_clock
from .scorer import Tracker


class RouteScene:
    termination_kind = 'events'

    def __init__(self, spec, ctx, p):
        self.spec, self._ctx = spec, ctx
        self.__dict__.update(p)
        self.instruction = public_instruction(spec)
        self.physical_start = None
        self.tracker = None
        self.target_initial_z = None
        self.closed_baseline = {}

    def before_step(self, tick: int) -> None:
        from . import mechanism
        i = tick - 1
        if i < len(self.frames):
            label, phase, writes = self.frames[i]
            for ident, value in writes.items():
                self.writer.write(tick, phase, self.bodies[ident], self.B[ident]['prim_path'], value)
        elif i == len(self.frames):
            self.physical_start = tick
            self.target_initial_z = float(self.centers(self.target)[2])
            self.closed_baseline = {c: float(self.centers(c)[2]) for c in self.covers}
            for ident in (self.target, self.cover):
                mechanism.set_dynamic(self.stage, self.B[ident]['prim_path'])
            self.writer.lock([self.B[x]['prim_path'] for x in (self.target, self.cover)])
            self.tracker = Tracker(self._tracker_context(), self.lay)

    def _tracker_context(self):
        return dict(physical_start=self.physical_start, closed_baseline=self.closed_baseline,
                    target_initial_z=self.target_initial_z, submission_stations=self.M['submission_stations'],
                    target_dimensions=self.B[self.target]['dimensions'],
                    expected_path=self.M['case']['expected_path'])

    def after_step(self, tick: int) -> None:
        row = self.recorder.record(tick, self.target)
        if self.tracker is not None:
            self.tracker.observe(row)

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz, instruction=self.instruction,
                           protocol_version=self.spec.protocol_version, **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        M, C = self.M, self.M['case']
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick, dict(
            ticks=self.recorder.ticks, physical_start=self.physical_start, arrivals=self.arrivals,
            station_xy={k: v['center_xy'] for k, v in M['stations'].items()},
            submission_stations=M['submission_stations'], target_dimensions=self.B[self.target]['dimensions'],
            closed_baseline=self.closed_baseline, target_initial_z=self.target_initial_z,
            target_cover_id=self.cover, expected_path=C['expected_path']))

    def health(self) -> Health:
        if self.recorder.ticks and not np.isfinite(self.recorder.ticks[-1]['target_center']).all():
            return Health(False, 'target pose is not finite')
        return Health(True)

    def is_complete(self) -> bool:
        """The round ends at submission N (any station); later actions never execute."""
        return self.tracker is not None and len(self.tracker.ledger.placements) >= len(self.M['case']['expected_path'])

    def events_complete(self) -> bool:
        return self.is_complete()

    def close(self) -> None:
        unsubscribe = getattr(self.subscription, 'unsubscribe', None)
        if callable(unsubscribe):
            unsubscribe()


def build(spec: InstanceSpec, ctx: BuildContext) -> RouteScene:
    from isaacsim.core.prims import SingleRigidPrim
    from mind2act_eval.robot import normalized_observation
    from . import mechanism
    from .environment import compile_environment
    lay = layout()
    M = plain(spec.config['manifest'])
    B = M['bodies']
    if spec.task_id != 'M1':
        raise ValueError('not an M1 spec')
    require_baseline_clock(spec.clock)
    frames, arrivals, occupants, _ = compile_environment(M, lay)
    stage = ctx.sim.stage
    ctx.robot.restore_asset(stage)
    ctx.robot.configure_stage(stage)
    # station base poses go to the shared FixedBase joints; legacy anchors are switched off (as the source did)
    ctx.robot.mount_bases(stage, [dict(r) for r in M['station']['robots']], legacy_anchors=True)
    ctx.robot.attach_cameras(stage)
    mechanism.author_stage(stage, M, lay)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    bodies = {ident: world.scene.add(SingleRigidPrim(prim_path=s['prim_path'], name='M_' + ident)) for ident, s in B.items()}
    ctx.robot.initialize(world)
    mechanism.runtime_collision(stage, M)
    contacts = ContactTable()
    subscription = mechanism.subscribe_contacts(contacts)
    covers = [k for k, s in B.items() if s['kind'] == 'cover']
    target, cover = M['target_body_id'], M['target_cover_id']

    def centers(ident):
        p, q = bodies[ident].get_world_pose()
        return mechanism.body_center(p, q, B[ident].get('center_offset_local', [0, 0, 0]))
    threshold = normalized_observation(lay['gripper_open_finger_m'])
    robot = ctx.robot
    recorder = Recorder({k: s['prim_path'] for k, s in B.items()}, centers,
                        lambda: (bodies[target].get_linear_velocity(), bodies[target].get_angular_velocity()), contacts,
                        {k: s['pad_prim'] for k, s in M['submission_stations'].items()}, covers,
                        lambda: [robot.observe()[s + '_gripper'] > threshold for s in ('left', 'right')])
    return RouteScene(spec, ctx, dict(
        M=M, B=B, lay=lay, frames=frames, arrivals=arrivals, writer=mechanism.EnvironmentWriter(), bodies=bodies,
        stage=stage, centers=centers, covers=covers, target=target, cover=cover, recorder=recorder,
        subscription=subscription))
