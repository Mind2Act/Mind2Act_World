"""CP06 prepared source station; shared owners control robot/world lifecycle."""
import math
from pathlib import Path
import numpy as np
from mind2act_eval.contracts import NativeEvidence, Health, to_json
from mind2act_eval.tasks._observation import observation
from .instances import validate, validate_assets, public_parameters
from .audit import Audit
from .physical import evaluate, terminal_units
from .terminal import TerminalFacts


class BalanceScene:
    def __init__(self,spec,ctx,sensors):
        validate(spec)
        self.spec,self.ctx,self.sensors=spec,ctx,sensors
        self.original_source={}
        if spec.config.get('layout_split') == 'validation':
            self.original_source['validation_source_spec']=to_json(spec)
        elif spec.config.get('layout_split') == 'mini':
            self.original_source['mini_source_spec']=to_json(spec)
        if 'train_source' in spec.config:
            self.original_source['training_source_spec']=to_json(spec)
        if 'test_source' in spec.config:
            from mind2act_eval.original_test_inputs import audit_source
            self.original_source=audit_source(spec,ctx.assets)
        self.config=to_json(spec.config['source_task']);self.tick=0;self.rows=[]
        self.initial=sensors.read();self.last=self.initial
        self.facts=TerminalFacts(self.config,self.initial)
        self.audit=Audit(Path(ctx.output_dir)/'CP06'/'raw-trace.jsonl')
        self.audit.append(dict(tick=0,raw=self.initial))
        self._candidate=False;self._complete=False;self._error_latched=False

    def before_step(self,tick):
        if tick!=self.tick+1:raise ValueError('CP06 before-step clock differs')
        self.sensors.before_step()

    def after_step(self,tick):
        if tick!=self.tick+1:raise ValueError('CP06 missing physics sample')
        self.sensors.after_step();raw=self.sensors.read()
        previous=self.last['simulation_time_s']
        if abs(raw['simulation_time_s']-previous-1/120)>1e-7:
            raise ValueError('CP06 native simulation clock is not 120Hz')
        self.rows.append(dict(physics_tick=tick,simulation_time_s=raw['simulation_time_s'],raw=raw))
        self.audit.append(dict(tick=tick,raw=raw))
        self.facts.append(raw);self.tick=tick;self.last=raw
        candidate=self.facts.candidate()
        if not candidate:self._complete=False
        elif not self._candidate:self._complete=evaluate(self.config,self.rows,self.initial)['success']
        self._candidate=candidate

    def observe(self,tick):
        return observation(self.spec,self.ctx.robot,tick,self.spec.config['instruction'])

    def evidence(self,tick):
        if tick!=self.tick:raise ValueError('CP06 physical sample not available')
        return NativeEvidence('CP06',self.spec.instance_id,tick,dict(difficulty=self.spec.difficulty,
            source_spec_sha256=self.spec.config['source_spec_sha256'],
            source_task_sha256=self.spec.config['declared_task_sha256'],
            source_task=self.config,audit=self.audit.receipt(),**self.original_source))

    def is_success(self):
        if self.is_complete():
            return True
        settlement = self._placement_ceiling()
        return bool(settlement and settlement[0] == settlement[1] > 0)

    def is_complete(self):
        return bool(getattr(self, '_complete', False))

    def _placement_ceiling(self):
        facts, initial = getattr(self, 'facts', None), getattr(self, 'initial', None)
        if facts is None or not facts.tail or not initial or not initial['objects']:
            return None
        state = facts.tail[-1]
        ref_z = getattr(getattr(self, 'sensors', None), 'support_z', None)
        if ref_z is None:
            # The lowest reset object origin is the fallback stock support datum.
            ref_z = min(obj['position'][2] for obj in initial['objects'].values())
        unusable = facts.drops.dropped | {
            path for path, obj in state['objects'].items()
            if obj['position'][2] < ref_z - .40 and not obj['robot_contact']
        }
        objects = facts.config['all_objects']
        fruits = {o['path'] for o in objects if o['kind'] == 'fruit'}
        weights = {o['path'] for o in objects if o['kind'] == 'weight'}
        available_weights = len(weights - unusable)
        if not (facts.drops.dropped or fruits & unusable or available_weights < 2):
            return None
        # Only pan placements earn partial credit. Use the scorer's actual
        # drop latch for the current count, including any measured pan support.
        _, placed_fruits, placed_weights = terminal_units(facts.config, state, facts.zero, facts.drops.dropped)
        return (len(placed_fruits) + min(2, len(placed_weights)),
                len(fruits - unusable) + min(2, available_weights))

    def irreversible_failure(self) -> str | None:
        if self._placement_ceiling() == (0, 0):
            return 'all_weighable_items_dropped'
        return None

    def health(self):
        values=[self.last['simulation_time_s'],self.last['beam_deg'],self.last['corrected_angle_deg'],
                self.last['rider_m'],self.last['rider_velocity_m_s']]
        values.extend(v for o in self.last['objects'].values() for v in o['position'])
        return Health(bool(np.isfinite(values).all()),'CP06 nonfinite native measurement' if not np.isfinite(values).all() else '')

    def result_metadata(self):
        if self.spec.config.get('layout_split') == 'validation':
            return dict(validation_source=self.spec.config['validation_source'],
                source_spec_sha256=self.spec.config['source_spec_sha256'],
                original_task_config_sha256=self.spec.config['declared_task_sha256'],
                prepared_scene_sha256=self.spec.config['prepared_scene_sha256'],
                public_parameters=public_parameters(self.config),native_validation='not_run')
        if self.spec.config.get('layout_split') == 'mini':
            return dict(layout_split='mini',
                source_spec_sha256=self.spec.config['source_spec_sha256'],
                original_task_config_sha256=self.spec.config['declared_task_sha256'],
                prepared_scene_sha256=self.spec.config['prepared_scene_sha256'],
                public_parameters=public_parameters(self.config),native_validation='not_run')
        if 'test_source' in self.spec.config:
            return dict(source_spec_sha256=self.spec.config['source_spec_sha256'],
                original_task_config_sha256=self.spec.config['declared_task_sha256'],
                source_geometry_proof_sha256=self.spec.config['source_geometry_proof_sha256'],
                original_scene_available=False,geometry_method='authored_original_parameters',
                public_parameters=public_parameters(self.config),native_validation='not_run')
        return dict(source_spec_sha256=self.spec.config['source_spec_sha256'],
            original_task_config_sha256=self.spec.config['declared_task_sha256'],
            resource_projection_proof_sha256=self.spec.config['resource_projection_proof_sha256'],
            composition_receipt_sha256=self.spec.config['composition_receipt_sha256'],
            public_parameters=public_parameters(self.config),native_validation='not_run')

    def close(self):
        self.audit.close()


def build(spec,ctx):
    row=validate_assets(spec,ctx.assets)
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from isaacsim.core.prims import SingleRigidPrim
    from .controls import BalanceControls
    from .rider_latch import RiderLatch
    from .pose_views import flanges
    from .sensors import Sensors, BALANCE
    stage=ctx.sim.stage;task=stage.GetPrimAtPath('/World/Task')
    if not task:raise ValueError('CP06 prepared full station/Task composition missing')
    ctx.robot.prepare_stage(stage)
    ctx.robot.mount_bases(stage,row['station_robots'],legacy_anchors=True)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world=ctx.world;bodies={}
    for prim in Usd.PrimRange(task):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path=str(prim.GetPath())
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr(128);rb.CreateSolverVelocityIterationCountAttr(8)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            bodies[path]=world.scene.add(SingleRigidPrim(path,name='CP06Body'+str(len(bodies))))
    required={o['path'] for o in spec.config['source_task']['all_objects']}|{
        BALANCE+'/Base',BALANCE+'/Beam',BALANCE+'/PanLeft',BALANCE+'/PanRight'}
    if not required<=set(bodies):raise ValueError('CP06 source rigid body coverage incomplete')
    flange_views=flanges(stage,world)
    ctx.robot.initialize(world);arms=ctx.robot.articulations(world)
    rest=np.r_[[math.radians(v) for v in (0,65,100,-65,0,0)],.044,.044]
    if not np.allclose(ctx.robot.rest_pose(),rest,rtol=0,atol=1e-12):raise ValueError('CP06 source rest changed')
    controls=None if spec.difficulty=='easy' else BalanceControls(stage,bodies['/World/Task/CalibrationButton'],bodies[BALANCE+'/Base'],bodies[BALANCE+'/Beam'])
    latch=RiderLatch(bodies[BALANCE+'/Beam'],bodies[BALANCE+'/Rider']) if spec.difficulty=='hard' else None
    if latch:
        drive=UsdPhysics.DriveAPI(stage.GetPrimAtPath(BALANCE+'/Joints/RiderSlide'),'linear')
        if any(float(getattr(drive,'Get'+key+'Attr')().Get() or 0)>0 for key in ('Stiffness','Damping')):
            raise ValueError('CP06 source rider latch cannot coexist with legacy slider drive')
    sensors=Sensors(ctx,to_json(spec.config['source_task']),bodies,arms,flange_views,controls,latch)
    table = stage.GetPrimAtPath('/World/Worktable')
    if table:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        sensors.support_z = float(cache.ComputeWorldBound(table).ComputeAlignedRange().GetMax()[2])
    return BalanceScene(spec,ctx,sensors)
