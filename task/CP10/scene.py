"""CP10 prepared native station; shared robot and runner own their lifecycles.

No task-owned robot, physics loop, or policy controller is constructed.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mind2act_eval.contracts import Health, NativeEvidence, to_json
from mind2act_eval.tasks._observation import observation, require_physics_hz
from mind2act_eval.tasks.CP10.protocol import PROTOCOL_ID, validate
from .machine import ArenaMachine
from .prefix import PhysicalPrefix

ROOT = '/World/Task/Design'
OBJECT_ROOT = '/World/Task/Objects/'
DIGITS = {'0':'abcdef','1':'bc','2':'abdeg','3':'abcdg','4':'bcfg',
          '5':'acdfg','6':'acdefg','7':'abc','8':'abcdefg','9':'abcdfg'}


class ArenaScene:
    def __init__(self, spec, ctx, stage, objects, launcher, names, indices, config, instruction):
        self.spec, self.ctx, self.stage = spec, ctx, stage
        self.objects, self.launcher, self.names, self.indices = objects, launcher, names, indices
        self.config, self.instruction = config, instruction
        self.tick = 0
        self.active_contacts = {}
        self.emitted = set()
        self.original_inputs = None
        if 'test_source' in spec.config:
            from .original_inputs import audit_source
            self.original_inputs=dict(spec=to_json(spec),audit=audit_source(spec,ctx.assets))
        self.machine = ArenaMachine(config, spec.config['collection_protocol'], spec.config['rule_version'])
        self.prefix = PhysicalPrefix(self.machine, ctx.output_dir)
        self.subscription = None
        self.public_state = None
        self.source_physics_diagnostics = {}
        self._health = Health(True)

    def before_step(self, tick):
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Usd, UsdGeom, UsdPhysics
        if type(tick) is not int or tick != self.prefix.tick + 1:
            raise ValueError('CP10 native hardware/shared runner clock divergence')
        self.tick = tick
        h = self.machine.hardware
        q = np.zeros(self.launcher.num_dof)
        q[self.indices['Yaw']] = math.radians(h['target_yaw_deg'])
        q[self.indices['Pitch']] = math.radians(h['target_pitch_deg'])
        q[self.indices['Slide']] = h['target_slide_x_m']
        self.launcher.apply_action(ArticulationAction(joint_positions=q))
        name = 'ball' + str(h['shot_number'] - 1)
        if self.machine.status == 'running' and h['release_now'] and name not in self.emitted:
            prim = self.stage.GetPrimAtPath(OBJECT_ROOT + name)
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(True)
            index = self.names.index(name)
            self.objects.enable_gravities(indices=np.array([index]))
            velocity = self.config['objects'][name]['launch_velocity']
            self.objects.set_velocities(np.array([velocity + [0., 0., 0.]], dtype=np.float32),
                                        indices=np.array([index]))
            UsdGeom.Imageable(prim).MakeVisible()
            self.emitted.add(name)

    def after_step(self, tick):
        try:
            was_running = self.machine.status == 'running'
            self.prefix.observe(self._measure(tick))
            if was_running:
                self._update_public()
        except (ValueError, TypeError, KeyError, OSError) as error:
            self._health = Health(False, 'CP10 physical evidence error: ' + str(error))

    def _measure(self, tick):
        from scipy.spatial.transform import Rotation
        positions, quaternions = self.objects.get_world_poses()
        velocities = self.objects.get_velocities()
        frame = dict(tick=tick, bodies={}, emitted_balls=sorted(self.emitted))
        for index, name in enumerate(self.names):
            path = OBJECT_ROOT + name
            contacts = set(); physical_contacts = set()
            for pair, facts in self.active_contacts.items():
                active=bool(facts['physical'])
                if not facts['active']:
                    continue
                for own, other in (pair, pair[::-1]):
                    if own == path or own.startswith(path + '/'):
                        if other.startswith(OBJECT_ROOT):
                            other = '/'.join(other.split('/')[:5])
                        contacts.add(other)
                        if active:physical_contacts.add(other)
            p, q = np.asarray(positions[index]), np.asarray(quaternions[index])
            size = self.config['objects'][name]
            half = np.asarray(size['size']) / 2
            if size.get('asset') != 'authored_sphere':
                half = abs(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()) @ half
            frame['bodies'][name] = dict(lower=(p - half).tolist(), upper=(p + half).tolist(),
                position=p.tolist(), orientation_wxyz=q.tolist(), linear=velocities[index, :3].tolist(),
                angular=velocities[index, 3:].tolist(), contacts=sorted(contacts),
                physical_contacts=sorted(physical_contacts),
                physical_robot_contact=any(p.startswith('/World/Robot') for p in physical_contacts),
                support_nets={n:bool(self.ctx.sim.contacts.supported(path,other=ROOT+'/Nets/'+n.title())) for n in ('near','far')})
        q, v = self.launcher.get_joint_positions(), self.launcher.get_joint_velocities()
        frame['launcher'] = dict(yaw_deg=math.degrees(float(q[self.indices['Yaw']])),
            pitch_deg=math.degrees(float(q[self.indices['Pitch']])),
            yaw_speed_deg_s=math.degrees(float(v[self.indices['Yaw']])),
            pitch_speed_deg_s=math.degrees(float(v[self.indices['Pitch']])),
            slide_x_m=float(q[self.indices['Slide']]), slide_speed_m_s=float(v[self.indices['Slide']]))
        return frame

    def _update_public(self):
        from pxr import Gf, UsdGeom, UsdShade
        from .mechanism.arena_preparation import launch_pose
        h = self.machine.hardware
        rows = [] if self.machine.cycle is None else self.machine.cycle.results
        correct = sum(bool(row['correct']) for row in rows if row['score_eligible'])
        total = self.config['arena']['scored_shots']
        state = (h['phase'], h['target_yaw_deg'], h['countdown'], h['target_slide_x_m'], correct, total)
        if state == self.public_state:
            return

        def digit(path, value, color):
            for segment in 'abcdefg':
                name = 'DigitOn_' + color if segment in DIGITS.get(str(value), '') else 'DigitOff'
                UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath(path + '/Segment_' + segment)).Bind(
                    UsdShade.Material(self.stage.GetPrimAtPath('/World/Materials/Four_' + name)))

        digit(ROOT + '/Countdown', h['countdown'], 'cyan')
        for group, value, color in (('Score', correct, 'cyan'), ('Total', total, 'amber')):
            digit(ROOT + '/Scoreboard/' + group + '/Tens', value // 10, color)
            digit(ROOT + '/Scoreboard/' + group + '/Ones', value % 10, color)
        (x, y), (dx, dy) = launch_pose(self.config['arena'], h['target_yaw_deg'], h['target_slide_x_m'])
        z, length = self.config['arena']['surface_z'], self.config['arena']['cue_length_m']
        UsdGeom.BasisCurves(self.stage.GetPrimAtPath(ROOT + '/Cue')).GetPointsAttr().Set(
            [Gf.Vec3f(x + dx * t, y + dy * t, z + .0008) for t in (.035, length)])
        ax, ay, nx, ny = x + dx * length, y + dy * length, -dy, dx
        vertices = [(-.005,-.014),(.005,-.014),(.005,.001),(.014,.001),(0,.017),(-.014,.001),(-.005,.001)]
        UsdGeom.Mesh(self.stage.GetPrimAtPath(ROOT + '/CueArrow')).GetPointsAttr().Set(
            [Gf.Vec3f(ax + a * nx + b * dx, ay + a * ny + b * dy, z + .001) for a, b in vertices])
        for path in (ROOT + '/Cue', ROOT + '/CueArrow'):
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).GetVisibilityAttr().Set(
                'inherited' if h['phase'] == 'prepare' else 'invisible')
        self.public_state = state

    def observe(self, tick):
        return observation(self.spec, self.ctx.robot, tick, self.instruction)

    def evidence(self, tick):
        if self.prefix.tick != tick:
            raise ValueError('CP10 evidence outer clock differs from physical/tail journal')
        values = self.prefix.values()
        values.update(difficulty=self.spec.difficulty, collection_protocol=to_json(self.spec.config['collection_protocol']),
                      rule_version=self.spec.config['rule_version'], normal_termination=True)
        if getattr(self,'original_inputs',None) is not None:
            values.update(input_binding_kind='original_test',original_test_inputs=self.original_inputs)
        from mind2act_eval.cp10_validation_inputs import evidence_fields
        values.update(evidence_fields(self.spec))
        return NativeEvidence('CP10', self.spec.instance_id, tick, values)

    def health(self):
        return self._health

    def events_complete(self):
        return self.machine.cycle is not None and self.machine.cycle.phase == 'complete'

    is_complete=events_complete

    def replay_eof_settle_ticks(self):
        # Source ShotCycle transitions from the final measured receipt to its
        # terminal state on the next tick (selected easy: 6288 -> 6289).
        # This request is independent of correct/incorrect shot outcomes.
        cycle = self.machine.cycle
        return int(self.machine.status == 'running' and cycle is not None
                   and cycle.phase == 'collected'
                   and len(self.machine.shots) == len(self.machine.colors))

    def result_metadata(self):
        return dict(native_terminal_tick=self.machine.terminal_tick, native_terminal_reason=self.machine.reason,
                    scored_until_tick=self.machine.tick, tail_after_native_terminal=bool(self.prefix.tail_count),
                    postterminal_replay_ticks=self.prefix.tail_count,
                    source_physics_diagnostics=self.source_physics_diagnostics)

    def close(self):
        subscription = self.subscription
        self.subscription = None
        try:
            unsubscribe = getattr(subscription, 'unsubscribe', None)
            if callable(unsubscribe):
                unsubscribe()
        finally:
            self.prefix.close()


def build(spec, ctx):
    require_physics_hz(spec)
    c = validate_binding(spec, ctx.assets)
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysxSchema, PhysicsSchemaTools, Usd, UsdPhysics

    # The shared CLI opens the exact relocated station before this builder;
    # reopening the original scene here would discard resolver-derived paths.
    stage = ctx.sim.stage
    task_root = stage.GetPrimAtPath('/World/Task')
    if not task_root:
        raise ValueError('CP10 prepared task root is missing')
    # Preserve source physics only on task-owned mechanisms and objects.
    # Global physics, station props and robot opinions belong to shared owners.
    for prim in Usd.PrimRange(task_root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateEnableCCDAttr(True)
            rb.CreateSolverPositionIterationCountAttr(c['prop_solver']['position_iterations'])
            rb.CreateSolverVelocityIterationCountAttr(c['prop_solver']['velocity_iterations'])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr(.001)
            contact.CreateRestOffsetAttr(0.)
    ctx.robot.prepare_stage(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    objects = world.scene.add(RigidPrim(OBJECT_ROOT + '.*', name='CP10Objects'))
    launcher = world.scene.add(SingleArticulation(ROOT + '/Emitter', name='ArenaLauncher'))
    ctx.robot.initialize(world)
    names = [p.rsplit('/', 1)[-1] for p in objects.prim_paths]
    if set(names) != set(c['objects']):
        raise ValueError('CP10 rigid body inventory differs')
    objects.set_velocities(np.zeros_like(objects.get_velocities()))
    indices = {name: launcher.get_dof_index(name) for name in ('Slide', 'Yaw', 'Pitch')}
    q = np.zeros(launcher.num_dof)
    q[indices['Yaw']], q[indices['Slide']] = math.radians(c['arena']['initial_yaw_deg']), c['arena']['initial_slide_x_m']
    launcher.set_joint_positions(q)
    launcher.set_joint_velocities(np.zeros_like(q))
    launcher.apply_action(ArticulationAction(joint_positions=q))
    world.physics_sim_view.update_articulations_kinematic()
    get_physx_interface().update_transformations(True, True, True, False)
    instructions = json.loads((Path(__file__).parent / 'config/instructions.json').read_text())
    instruction = next(row['en'] for row in instructions if spec.difficulty in row['d'])
    scene = ArenaScene(spec, ctx, stage, objects, launcher, names, indices, c, instruction)
    scene.source_physics_diagnostics = source_physics_diagnostics(stage, c)

    def report(headers, data):
        for h in headers:
            pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(value)) for value in (h.collider0, h.collider1)))
            if any('/World/Task/' in path for path in pair):
                active=h.type != ContactEventType.CONTACT_LOST
                points=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
                physical=bool(points and any(np.linalg.norm(p.impulse)>0 or float(p.separation)<=0 for p in points))
                prior=scene.active_contacts.get(pair)
                if active and not points and prior is not None:
                    continue  # Preserve a sleeping manifold until actual CONTACT_LOST.
                scene.active_contacts[pair]=dict(active=active,physical=physical)

    scene.subscription = get_physx_simulation_interface().subscribe_contact_report_events(report)
    scene.prefix.observe(scene._measure(0))
    scene._update_public()
    return scene


def encode_config(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def source_physics_diagnostics(stage, config):
    """Record the source wrapper's global requests without applying them."""
    requested = {'physxScene:timeStepsPerSecond': 120, 'physxScene:enableCCD': True,
                 'physxScene:solverType': 'TGS'}
    physics = stage.GetPrimAtPath('/World/Physics')
    observed = {}
    for name in requested:
        attr = physics.GetAttribute(name) if physics else None
        observed[name] = attr.Get() if attr else None
    return dict(source_wrapper_global_requested=requested, observed_shared_scene=observed,
                source_non_robot_requested=dict(enableCCD=True, solver=config['prop_solver'],
                                               contactOffset=.001, restOffset=0.),
                runtime_write_scope='/World/Task', global_and_station_opinions_applied=False)


def validate_binding(spec, assets):
    c = to_json(spec.config['source_config'])
    validate(spec.config['collection_protocol'], spec.config['rule_version'])
    if spec.task_id != 'CP10' or c['task'] != 'CP10' or c['difficulty'] != spec.difficulty or c['seed'] != spec.seed:
        raise ValueError('CP10 spec/source instance binding differs')
    if spec.instance_id.startswith('VAL-CP10-') or spec.config.get('layout_split')=='validation':
        from mind2act_eval.cp10_validation_inputs import validate_assets as validation_assets
        return validation_assets(spec,assets)
    if 'train_source' in spec.config:
        from mind2act_eval.cp10_training_inputs import validate_assets as training_validate_assets
        training_validate_assets(spec, assets)
        return c
    if 'test_source' in spec.config:
        from .original_inputs import validate_binding as original_binding
        return original_binding(spec,assets)
    pins = json.loads((Path(__file__).parent / 'config/source.json').read_text())
    cohort = pins['training_cohorts'][spec.difficulty]
    if hashlib.sha256(encode_config(c)).hexdigest() != cohort['source_config_canonical_sha256']:
        raise ValueError('CP10 selected source configuration hash differs')
    from mind2act_eval.sim.source_assets import verify_source_file
    verify_source_file(assets, 'scene', cohort['prepared_scene_sha256'],error='CP10 prepared source scene hash differs')
    return c
