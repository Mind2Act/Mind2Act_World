"""Prepared original bell/curve geometry on the framework's single World/X5."""
import hashlib
import json
import numpy as np
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.contracts import Health, Observation
from .instances import validate
from .evidence import Recorder

E2E_INSTRUCTION_VERSION = 'pr6-handoff-static-e2e-v2'
TASK_PROTOCOL_VERSION = SCORING_VERSION


def configure_props(stage, case):
    """Exact selected training native.py settings; no task robot overrides."""
    from pxr import Usd, UsdPhysics, PhysxSchema
    for name in case['objects']:
        root = stage.GetPrimAtPath('/World/Task/Objects/'+name)
        if not root.IsValid():
            raise ValueError('PR6 selected task object missing: '+name)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb.CreateEnableCCDAttr(True)
                rb.CreateSolverPositionIterationCountAttr(case['prop_solver']['position_iterations'])
                rb.CreateSolverVelocityIterationCountAttr(case['prop_solver']['velocity_iterations'])
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(.001)
                collision.CreateRestOffsetAttr(0.)


def _objects(world):
    from isaacsim.core.prims import RigidPrim
    return world.scene.add(RigidPrim('/World/Task/Objects/.*',name='PR6_objects'))


class DominoScene:
    def __init__(self,spec,ctx,objects,contacts,subscription):
        self.spec,self.ctx,self.objects = spec,ctx,objects
        self.contacts,self.subscription = contacts,subscription
        self.names = [p.rsplit('/',1)[-1] for p in objects.prim_paths]
        if set(self.names) != set(spec.config['source_case']['objects']):
            raise ValueError('PR6 native rigid bodies differ from selected scene')
        self.recorder = Recorder(spec.instance_id,spec.config['source_case'],trace_path=ctx.output_dir/'PR6'/'raw-trace.jsonl')
        try:
            self.recorder.record(self.measure(0))
        except BaseException:
            self.recorder.close()
            raise

    def measure(self,tick):
        positions,orientations = self.objects.get_world_poses()
        velocities = self.objects.get_velocities()
        bodies = {name:dict(position=np.asarray(positions[i]).tolist(),orientation_wxyz=np.asarray(orientations[i]).tolist(),
            linear=np.asarray(velocities[i,:3]).tolist(),angular=np.asarray(velocities[i,3:]).tolist()) for i,name in enumerate(self.names)}
        return dict(tick=tick,time_s=float(self.ctx.world.current_time),bodies=bodies,contacts=self.contacts.snapshot())

    def before_step(self,tick):
        pass

    def after_step(self,tick):
        self.recorder.record(self.measure(tick))

    def observe(self,tick):
        return Observation(tick=tick,sim_time_s=tick/120,instruction=self.spec.config['instruction'],
            protocol_version=self.spec.protocol_version,**dict(self.ctx.robot.observe()))

    def evidence(self,tick):
        return self.recorder.evidence(tick)

    def health(self):
        return Health(True)

    def is_success(self):
        # A recorded violation may force score zero, but must not stop physics.
        return bool(self.recorder.rules.facts()['success'])

    def is_complete(self):
        return self.is_success()

    def result_metadata(self):
        return dict(source_id=self.spec.source_id,native_task_alias='PR8',native_task_id='CP11',
            source_spec_sha256=self.spec.config['source_spec_sha256'],
            source_config_sha256=self.spec.config['source_config_sha256'],
            scene_sha256=self.spec.config['scene_sha256'],public_notice_interface_pending=False,
            public_notice_interface='agent recipe and physical scene; no E2E instruction suffix',
            e2e_instruction_version=E2E_INSTRUCTION_VERSION,
            task_protocol_version=TASK_PROTOCOL_VERSION,
            prompt_model_input='normal observation and control from tick zero',
            scoring_version=SCORING_VERSION,
            public_notice_source='rules/session.py:notice,action_enabled,public',
            source_notice_ticks=self.spec.config['source_case']['cue_ticks'],
            source_global_physics_expectation=dict(physics_hz=120,enable_ccd=True,solver_type='TGS'),
            global_physics_owner='shared simulator; task does not author global/station settings')

    def close(self):
        try:
            if self.subscription is not None:
                self.subscription.unsubscribe()
        finally:
            self.subscription = None
            self.recorder.close()


def build(spec,ctx):
    from .mechanism.contacts import Contacts,subscribe
    validate(spec)
    from mind2act_eval.sim.source_assets import source_file
    source_file(ctx.assets, 'station')
    stage = ctx.sim.stage
    ctx.robot.prepare_stage(stage)
    configure_props(stage,spec.config['source_case'])
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    objects = _objects(ctx.world)
    ctx.robot.initialize(ctx.world)
    objects.initialize()
    objects.set_velocities(np.zeros_like(objects.get_velocities()))
    # Source initialization is zero-step. Shared renderer owns non-physics warmup.
    ctx.world.physics_sim_view.update_articulations_kinematic()
    from omni.physx import get_physx_interface
    get_physx_interface().update_transformations(True,True,True,False)
    contacts = Contacts()
    subscription = subscribe(contacts)
    try:
        return DominoScene(spec,ctx,objects,contacts,subscription)
    except BaseException:
        subscription.unsubscribe()
        raise
