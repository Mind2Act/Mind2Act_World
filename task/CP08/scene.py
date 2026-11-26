"""CP08 task hooks; shared framework owns robot actions and physics stepping."""
from pathlib import Path
from mind2act_eval.contracts import NativeEvidence, Health, to_json
from mind2act_eval.tasks._observation import observation
from .audit import Audit
from .instances import validate
from .projection import cue_visible
from .replay import Replay
from .physical import evaluate, StabilityWindow


class StructureScene:
    def __init__(self, spec, ctx, sensors):
        validate(spec)
        self.spec, self.ctx, self.sensors = spec, ctx, sensors
        self.original_source={}
        if spec.config.get('layout_split') == 'validation':
            self.original_source['validation_source_spec']=to_json(spec)
        if 'train_source' in spec.config:
            self.original_source['training_source_spec']=to_json(spec)
        if 'test_source' in spec.config:
            from .instances import original_case
            from mind2act_eval.original_test_inputs import audit_source
            original_case(spec,ctx.assets)
            self.original_source=audit_source(spec,ctx.assets)
        self.projection = to_json(spec.config['source_projection'])
        self.tick = 0
        self.initial = sensors.read(0)
        # Formal tests start at source tick 0; selected H5 replays start at 72.
        self.replay = Replay(to_json(spec.config['source_case']), self.initial,
                             source_state_tick=self.projection['source_state_tick'])
        self.windows = {b['id']: StabilityWindow(b, self.replay.config) for b in self.replay.config['blocks']}
        self.audit = Audit(Path(ctx.output_dir)/'CP08'/'raw-trace.jsonl')
        self.audit.append(self.initial)

    def before_step(self, tick):
        if tick != self.tick + 1:
            raise ValueError('CP08 before-step clock differs')
        self.sensors.set_cue(cue_visible(tick, self.projection))

    def after_step(self, tick):
        if tick != self.tick + 1:
            raise ValueError('CP08 missing physical tick')
        raw = self.sensors.read(tick)
        normalized = self.replay.sample(raw)
        for name, window in self.windows.items():
            window.append(normalized['blocks'][name])
        self.audit.append(raw)
        self.tick = tick

    def observe(self, tick):
        return observation(self.spec,self.ctx.robot,tick,self.spec.config['instruction'])

    def evidence(self, tick):
        if tick != self.tick:
            raise ValueError('CP08 evidence requested before physical sample')
        return NativeEvidence('CP08',self.spec.instance_id,tick,dict(
            source_case=to_json(self.spec.config['source_case']),
            source_projection=self.projection,audit=self.audit.receipt(),**self.original_source))

    def is_success(self):
        if self.is_complete():
            return True
        settlement = self._placement_ceiling()
        return bool(settlement and settlement[0] == settlement[1] > 0 and settlement[2])

    def is_complete(self):
        # Reference placements alone do not describe the whole success state:
        # distractors can move while every reference block remains correct.
        windows, replay = getattr(self, 'windows', None), getattr(self, 'replay', None)
        initial = getattr(self, 'initial', None)
        return bool(windows and replay and initial and all(window.stable for window in windows.values())
                    and evaluate(replay.rows, replay.config, initial)['success'])

    def _placement_ceiling(self):
        replay, windows = getattr(self, 'replay', None), getattr(self, 'windows', None)
        if replay is None or not replay.rows or not windows:
            return None
        frame = replay.rows[-1]
        if not frame.get('support_measurements_complete'):
            return None
        blocks, config = frame['blocks'], replay.config
        targets = config['blocks']
        dropped = {b['id'] for b in targets if blocks[b['id']]['position'][2] < config['table_z'] - .40
                   and not blocks[b['id']]['robot_contact']}
        if not dropped:
            return None
        # A missing physical support prevents its immediate child's check.
        # A surviving child can itself support other blocks without passing
        # its own placement check (for example, on a temporary support).
        dead = dropped | {b['id'] for b in targets if dropped.intersection(b['supports'])}
        core = {b['id'] for b in targets}
        if config['level'] == 'hard':
            core.remove('A')  # The preinstalled block is a full-success gate only.
        available = core - dead
        current = sum(windows[name].stable for name in core)
        settled = all(windows[name].stable and not blocks[name]['robot_contact'] for name in available)
        return current, len(available), settled

    def irreversible_failure(self) -> str | None:
        settlement = self._placement_ceiling()
        if settlement and settlement[0] == settlement[1] == 0 and not self.is_complete():
            return 'core_blocks_fell_off_station'
        return None

    def health(self):
        return Health(not self.sensors.errors, '; '.join(self.sensors.errors))

    def result_metadata(self):
        return dict(source_projection=self.projection,source_spec_sha256=self.spec.config['source_spec_sha256'],
                    native_validation='not_run')

    def close(self):
        self.audit.close()
        self.sensors.close()


def build(spec, ctx):
    from .instances import validate_assets
    row=validate_assets(spec,ctx.assets)
    import hashlib
    import math
    import numpy as np
    from isaacsim.core.prims import SingleRigidPrim
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysxSchema,Usd,UsdGeom
    from mind2act_eval.contracts import Action
    from mind2act_eval.robot import normalized_observation
    from .sensors import Contacts, Sensors
    from mind2act_eval.sim.source_assets import source_file
    scene=source_file(ctx.assets, 'station')
    textures={name:ctx.assets.resolve(key) for name,key in
              (('plan.png','cue_plan'),('screen-idle.png','cue_idle'))}
    for name,path in textures.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['cue_texture_sha256'][name]:
            raise ValueError('CP08 selected cue texture changed')
    stage=ctx.sim.stage
    ctx.robot.prepare_stage(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world=ctx.world
    bodies={}
    for block in spec.config['source_case']['blocks']+spec.config['source_case']['distractors']:
        path='/World/CP11/Blocks/'+block['id']
        prim=stage.GetPrimAtPath(path)
        if not prim:raise ValueError('CP08 required source body missing: '+path)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        bodies[block['id']]=world.scene.add(SingleRigidPrim(path,name='CP08_'+block['id']))
    from .tcp import flange_readers
    flanges=flange_readers(stage,world)
    ctx.robot.initialize(world)
    arms=ctx.robot.articulations(world)
    rest=np.r_[[math.radians(v) for v in (0,65,100,-65,0,0)],.044,.044]
    if not np.allclose(ctx.robot.rest_pose(),rest,rtol=0,atol=1e-12):
        raise ValueError('CP08 selected H5/source initial rest differs')
    for block in spec.config['source_case']['blocks']+spec.config['source_case']['distractors']:
        body=bodies[block['id']]
        body.set_world_pose(np.asarray(block['initial']),np.array([1.,0.,0.,0.]))
        body.set_linear_velocity(np.zeros(3));body.set_angular_velocity(np.zeros(3))
    screen=stage.GetAttributeAtPath('/World/CP11Looks/Card_World_CP11_GoalMonitor_Screen/Texture.inputs:file')
    if not screen:raise ValueError('CP08 source plan display input missing')
    contacts=Contacts()
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(contacts.receive)
    area=stage.GetPrimAtPath('/World/CP11/BuildArea')
    if not area:raise ValueError('CP08 authored build area missing')
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(area).ComputeAlignedRange()
    area_bounds=dict(lower=list(bounds.GetMin())[:2],upper=list(bounds.GetMax())[:2])
    sensors=Sensors(stage,bodies,arms,screen,textures,contacts,subscription,flanges,area_bounds)
    try:
        hold=Action(rest[:6],rest[:6],normalized_observation(rest[6]),normalized_observation(rest[6]))
        sensors.set_cue(True)
        # Original source reset holds120 physics ticks; selected H5 begins at
        # source state/control tick72, while its matched action begins at76.
        for _ in range(120+spec.config['source_projection']['source_state_tick']):
            ctx.robot.apply(hold);ctx.sim.step(render=False)
        return StructureScene(spec,ctx,sensors)
    except BaseException:
        sensors.close()
        raise
