"""One DR3 build on the selected prepared source scene and shared X5/Isaac.

The original single/dual world supplied object/gate initialization, 120 quiet
initialization ticks and the selected initial gate phase. No old world, Kit
launcher, expert, source execution or task-owned robot driver is imported.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mind2act_eval.contracts import Action, BuildContext, Health, InstanceSpec, NativeEvidence, Observation
from .capture import Capture, contact_measurement
from .audit import AppendOnlyTrace, write_manifest
from .instances import SOURCE_ID, validate
from .mechanism import gate_definitions, gates, single_target


class ContactSamples:
    """Actual current contact reports, preserving collider-normal ownership."""

    def __init__(self):
        self.rows, self.errors, self.counter = {}, [], [0]

    def receive(self, headers, data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        for header in headers:
            self.counter[0] += 1
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(value))
                         for value in (header.collider0, header.collider1))
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            try:
                row = contact_measurement(pair, points)
                row['active'] = header.type != ContactEventType.CONTACT_LOST
                key = tuple(sorted(pair))
                previous = self.rows.get(key)
                if row['active'] and not len(points) and previous and previous['active']:
                    # A sleeping/persisting manifold need not emit fresh force
                    # points. Its owned geometry remains valid until LOST.
                    row['item_upward_normals'] = list(previous['item_upward_normals'])
                self.rows[key] = row
            except (ValueError, TypeError, AttributeError) as error:
                if str(error) not in self.errors:
                    self.errors.append(str(error))


class GateScene:
    def __init__(self, spec, ctx, item, doors, schedules, contacts, subscription, capture):
        self.spec, self.ctx, self.item = spec, ctx, item
        self.doors, self.schedules = doors, schedules
        self.contacts, self.subscription, self.capture = contacts, subscription, capture
        self.rows = []
        self._success_analysis_tick = None
        self._success_analysis = None
        self._score_violation_latched = False
        self.initial_time = float(ctx.world.current_time)
        audit_dir = Path(ctx.output_dir) / 'DR3'
        self.trace = AppendOnlyTrace(audit_dir / 'raw-trace.jsonl')
        try:
            self.manifest_receipt = write_manifest(audit_dir / 'manifest.json', capture.manifest)
            self._sample(0)
        except BaseException:
            self.trace.close()
            raise

    def _sample(self, tick):
        if self.rows and tick == self.rows[-1]['physics_tick']:
            return
        if tick != len(self.rows):
            raise ValueError('DR3 physics evidence must be sampled exactly once in order')
        row = dict(physics_tick=tick, simulation_time_s=float(self.ctx.world.current_time),
                   detector=self.capture.snapshot(self.contacts.rows, tick,
                                                  float(self.ctx.world.current_time)))
        self.trace.append(row)
        self.rows.append(row)

    def before_step(self, tick):
        if tick != len(self.rows):
            raise ValueError('DR3 before-step clock mismatch')
        self.capture.begin_step(tick - 1)
        _drive(self.doors, self.schedules, self.spec.config['source_case'],
               (tick - 1 + self.spec.config['phase_ticks']) / 120)

    def after_step(self, tick):
        self._sample(tick)

    def observe(self, tick):
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz,
                           instruction=self.spec.config['instruction'],
                           protocol_version=self.spec.protocol_version,
                           **dict(self.ctx.robot.observe()))

    def evidence(self, tick):
        if tick != self.rows[-1]['physics_tick']:
            raise ValueError('DR3 evidence requested before the physical sample')
        if self.trace.rows != tick + 1:
            raise ValueError('DR3 append-only audit sample count differs from task clock')
        return NativeEvidence('DR3', self.spec.instance_id, tick,
                              dict(difficulty=self.spec.difficulty, source_id=SOURCE_ID,
                                   condition=dict(self.spec.config['condition']),
                                   manifest=self.manifest_receipt,
                                   raw_trace=self.trace.receipt()))

    def health(self):
        if self.contacts.errors:
            return Health(False, 'DR3 contact measurement failed: ' + self.contacts.errors[0])
        for pose in self.rows[-1]['detector']['poses'].values():
            if not all(math.isfinite(float(value)) for value in pose['position'] + pose['quaternion_wxyz']):
                return Health(False, 'DR3 measured pose is not finite')
        return Health(True)

    def is_complete(self):
        return False  # Violations never terminate control through this legacy hook.

    def is_success(self):
        """Action-completion probe: cheap delivery check before route replay.

        This is called after a model action, not after every physics tick.
        An observed route violation only locks the score, never the raw clock.
        """
        if self._score_violation_latched:
            return False
        manifest = self.capture.manifest
        items = [(name, body) for name, body in manifest.get('bodies', {}).items()
                 if body['role'] == 'item']
        if len(items) != 1:
            return False
        from .events import _contact_state, _center_in_tray, rebuild_events
        from .geometry import finite_array, transform_vertices
        frame = self.rows[-1]['detector']
        if not _contact_state(frame, manifest)['tray_support']:
            return False
        name, item = items[0]
        vertices = transform_vertices(item['vertices'], frame['poses'][name])
        center = (vertices.min(axis=0)+vertices.max(axis=0))/2
        local = center-finite_array(manifest['tray']['origin'], (3,))
        if not _center_in_tray(local[:2], manifest['tray']):
            return False
        tick = self.rows[-1]['physics_tick']
        if self._success_analysis_tick != tick:
            self._success_analysis = rebuild_events(self.rows, manifest)
            self._success_analysis_tick = tick
        result = self._success_analysis
        if result['evidence_complete'] and result['violation_latched']:
            self._score_violation_latched = True
        return bool(result['evidence_complete'] and result['success'])

    def result_metadata(self):
        return dict(source_id=SOURCE_ID, condition=dict(self.spec.config['condition']),
                    physics_hz=120, control_hz=30, initial_phase_ticks=self.spec.config['phase_ticks'],
                    initialization_ticks=120, sampling=self.capture.sampling_receipt(),
                    historical_gpu_qualification_is_scoring_precondition=False)

    def close(self):
        subscription, self.subscription = self.subscription, None
        try:
            unsubscribe = getattr(subscription, 'unsubscribe', None)
            if callable(unsubscribe):
                unsubscribe()
        finally:
            self.trace.close()


def _drive(doors, schedules, source, time):
    from isaacsim.core.utils.types import ArticulationAction
    for name, door in doors.items():
        gate = schedules[name]
        target = single_target(time, source) if gate is None else gate.target(time)
        door.apply_action(ArticulationAction(joint_positions=np.asarray([target])))


def build(spec: InstanceSpec, ctx: BuildContext) -> GateScene:
    validate(spec)
    from mind2act_eval.sim.source_assets import verify_source_file
    verify_source_file(ctx.assets, 'station', spec.config['scene_sha256'], error='DR3 prepared source scene changed')
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_simulation_interface
    stage = ctx.sim.stage
    ctx.robot.prepare_stage(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world, source = ctx.world, spec.config['source_case']
    item = world.scene.add(SingleRigidPrim('/World/Task/Item', name='DR3Item'))
    schedules = gates(source, spec.difficulty)
    doors = {name: world.scene.add(SingleArticulation(
        '/World/Task/Gate' if name == 'single' else '/World/Task/Gate/' + name,
        name='DR3Gate_' + name)) for name in schedules}
    ctx.robot.initialize(world)
    robots = ctx.robot.articulations(world)
    for door in doors.values():
        door.set_joint_positions(np.zeros(1))
        door.set_joint_velocities(np.zeros(1))
        door._articulation_view.set_gains(kps=np.asarray([[15000.]]), kds=np.asarray([[300.]]))
        door.apply_action(ArticulationAction(joint_positions=np.zeros(1)))
    item.set_world_pose(np.asarray(source['SOURCE']), np.asarray([1., 0., 0., 0.]))
    item.set_linear_velocity(np.zeros(3))
    item.set_angular_velocity(np.zeros(3))
    contacts = ContactSamples()
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts.receive)
    try:
        rest = ctx.robot.rest_pose()
        from mind2act_eval.robot import normalized_observation
        hold = Action(rest[:6], rest[:6], normalized_observation(rest[6]), normalized_observation(rest[6]))
        for _ in range(120):  # source initialization, not model observation warmup
            ctx.robot.apply(hold)
            ctx.sim.step(render=False)
        for tick in range(spec.config['phase_ticks']):
            _drive(doors, schedules, source, tick / 120)
            ctx.robot.apply(hold)
            ctx.sim.step(render=False)
        capture = Capture(stage, item, robots, source,
                          {name: (lambda door=door: door.get_joint_positions()[0])
                           for name, door in doors.items()},
                          gate_definitions(source, spec.difficulty))
        capture.subscription_active = subscription is not None
        capture.contact_errors = contacts.errors
        capture.contact_report_counter = contacts.counter
        result = GateScene(spec, ctx, item, doors, schedules, contacts, subscription, capture)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        (ctx.output_dir / 'dr3-initialization.json').write_text(json.dumps(result.result_metadata(), indent=2) + '\n')
        return result
    except BaseException:
        unsubscribe = getattr(subscription, 'unsubscribe', None)
        if callable(unsubscribe):
            unsubscribe()
        raise
