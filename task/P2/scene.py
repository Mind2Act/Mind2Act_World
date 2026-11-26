"""Passive source steamer scene, native raw measurements and shared robot API."""
from pathlib import Path
from mind2act_eval.contracts import Action,Health,NativeEvidence
from mind2act_eval.tasks._observation import observation
from .binding import SOURCE_ID,VERSION,SCORING_VERSION,DISMANTLING_RULE,EVALUATION_MODE,instruction,validate
from .evidence import Rows,Manifolds,measure
from .measurements import robot_paths,validate_frame
from .stages import stage_value


def prepare_task_contacts(stage):
    from pxr import UsdPhysics,PhysxSchema
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Task/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


def author_task(stage,source,g,assets):
    """Compose original task USD and the exact source reset overlays before reset."""
    from pxr import Gf,Sdf
    existing=stage.GetPrimAtPath('/World/Task')
    if existing and existing.GetChildren():
        raise ValueError('P2 requires a clean station, not another populated task scene')
    stage.DefinePrim('/World/Task','Xform').GetReferences().AddReference(str(assets.resolve('task_scene')))
    for name,attr,value,kind in (
            ('Stove','xformOp:translate',g['stove_xyz'],Gf.Vec3d),
            ('Board','xformOp:translate',g['board_xyz'],Gf.Vec3d),
            ('Stove/PotGeometry','xformOp:scale',g['pot_geometry_scale'],Gf.Vec3f),
            ('Stove/PreplacedTier','xformOp:translate',g['preplaced_tier_translate'],Gf.Vec3d)):
        prim=stage.GetPrimAtPath('/World/Task/'+name)
        if not prim or not prim.GetAttribute(attr):
            raise ValueError('P2 original task transform missing: '+name)
        prim.GetAttribute(attr).Set(kind(*value))
    for name,xyz in g['decoy_xyz'].items():
        stage.GetPrimAtPath('/World/Task/'+name).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*xyz))
    names=[*g['target_initial_xyz'],*g['decoy_xyz']]
    for name,food in zip(names,source['target_foods_bottom_to_top']+source['decoy_foods']):
        prim=stage.GetPrimAtPath('/World/Task/'+name+'/Food')
        if not prim:
            raise ValueError('P2 original food reference missing: '+name)
        prim.GetReferences().SetReferences([Sdf.Reference(str(assets.resolve('food_'+food)))])


class MeasurementScene:
    termination_kind = 'result'

    def __init__(self,spec,ctx,source,geometry,bodies,robots,contacts):
        self.spec,self.ctx,self.source,self.geometry=spec,ctx,source,geometry
        self.bodies,self.robots,self.contacts=bodies,robots,contacts
        self.initial=self._measure(0)
        validate_frame(self.initial,bodies)
        from .progress import Progress
        self.progress=Progress(source,self.initial)
        self.rows=Rows(Path(ctx.output_dir)/'private'/'P2'/'measurements.jsonl')

    def _measure(self,tick):
        return measure(tick,self.bodies,self.robots,self.contacts,self.ctx.world,self.ctx.sim.stage)

    def before_step(self,tick):
        pass

    def after_step(self,tick):
        row=self._measure(tick)
        validate_frame(row,self.bodies)
        self.progress.sample(row)
        self.rows.append(row)

    def observe(self,tick):
        return observation(self.spec,self.ctx.robot,tick,instruction(self.spec.difficulty))

    def evidence(self,tick):
        return NativeEvidence('P2',self.spec.instance_id,tick,dict(source_instance=self.source,
            source_record_sha256=self.spec.config['source_record_sha256'],initial=self.initial,
            measurements=self.rows.receipt(),
            dismantling_rule_version=DISMANTLING_RULE,
            input_protocol_version=self.spec.config['input_protocol_version']))

    def is_complete(self):
        progress=getattr(self,'progress',None)
        return progress is not None and progress.result()['task_success']

    def _prefix_ceiling(self):
        progress=getattr(self,'progress',None)
        geometry=getattr(self,'geometry',None)
        if progress is None or not geometry or not progress.facts.tick:
            return None
        # The board origin is at z=0; the stove anchors at the tabletop.
        cutoff=geometry['stove_xyz'][2]-.40
        row=progress.facts.last
        for i,name in enumerate([*progress.targets,'Lid']):
            if row['poses'][name]['position'][2]<cutoff and not robot_paths(row,name):
                return i
        return None

    def irreversible_failure(self) -> str | None:
        if self._prefix_ceiling()==0 and self.progress.prefix()==0:
            return 'tier1_fell_off_station'
        return None

    def health(self):
        from math import isfinite
        return Health(all(isfinite(float(v)) for b in self.bodies.values() for v in b.get_world_pose()[0]),
                      'P2 native steamer poses must be finite')

    def result_metadata(self):
        return dict(source_id=SOURCE_ID,public_task_id='P2',native_task_id='P_STEAMER',rubric=SCORING_VERSION,
            task_protocol_version=SCORING_VERSION,source_input_version=VERSION,
            evaluation_mode=EVALUATION_MODE,formal_eligible=True,accepted_expert_episode=False,
            source_record_sha256=self.spec.config['source_record_sha256'],
            input_protocol_version=self.spec.config['input_protocol_version'],
            initialization_physics_ticks=960,native_physics_ticks=self.rows.count,
            measurements=self.rows.receipt(),dismantling_rule_version=DISMANTLING_RULE,
            physical_progress=self.progress.result(),
            source_world_time=self.spec.config.get('source_world_time'),
            source_pre_record_warmup_ticks=960,warmup_used_as_score_gate=False)

    def is_success(self) -> bool:
        progress=getattr(self,'progress',None)
        if progress is None:
            return False
        result=progress.result()
        if result['task_success']:
            return True
        ceiling=self._prefix_ceiling()
        source=getattr(self,'source',None)
        if ceiling is None or ceiling==0 or source is None:
            return False
        stage=dict(valid_disassembly=result['valid_disassembly'],target_count=result['target_count'],final_conditions_met=False,
                   difficulty=source['difficulty'])
        return stage_value(restored_prefix=result['restored_prefix'],**stage)==stage_value(restored_prefix=ceiling,**stage)

    def close(self):
        self.rows.close();self.contacts.close()


def build(spec,ctx):
    source,g=validate(spec,ctx)
    from isaacsim.core.prims import SingleRigidPrim
    import numpy as np
    stage=ctx.sim.stage
    ctx.robot.prepare_stage(stage)
    author_task(stage,source,g,ctx.assets)
    names=[*g['target_initial_xyz'],'Lid',*g['decoy_xyz']]
    for path in ['/World/Task/'+n for n in names]+['/World/Task/Stove/PreplacedTier']:
        if not stage.GetPrimAtPath(path):
            raise ValueError('P2 prepared source scene is missing '+path)
    prepare_task_contacts(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world=ctx.world
    # World defaults override the authored scene flag; restore the collection setting.
    from pxr import Sdf, UsdPhysics
    physics_scene=stage.GetPrimAtPath('/World/Task/Physics')
    if not physics_scene or not physics_scene.IsA(UsdPhysics.Scene):
        raise ValueError('P2 source PhysicsScene is missing')
    physics_scene.CreateAttribute('physxScene:enableStabilization',
        Sdf.ValueTypeNames.Bool, custom=False).Set(True)
    bodies={name:world.scene.add(SingleRigidPrim(prim_path='/World/Task/'+name,name='P2_'+name)) for name in names}
    ctx.robot.initialize(world)
    robots=ctx.robot.articulations(world)
    for body in bodies.values():
        body.set_linear_velocity(np.zeros(3));body.set_angular_velocity(np.zeros(3))
    contacts=Manifolds();contacts.subscribe()
    try:
        # Actual source world.py pre-record settle. It precedes evidence tick0,
        # contributes no stage/stability credit, and does not consume H5 rows.
        rest=spec.config['robot_rest']
        action=Action(tuple(rest[:6]),tuple(rest[:6]),1.,1.)
        for _ in range(960):
            ctx.robot.apply(action)
            ctx.sim.step(render=False)
        return MeasurementScene(spec,ctx,source,g,bodies,robots,contacts)
    except BaseException:
        contacts.close();raise
