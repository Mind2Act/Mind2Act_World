"""F2 native blocks and spring confirm mechanism through shared framework services."""
import hashlib
import json
from pathlib import Path

from mind2act_eval.contracts import Health, NativeEvidence, Observation, to_json
from .provenance import SOURCE_ID, VERSION, effective_config, read_origin, validate_start
from .mechanism import Task
from .geometry import Body
from .evidence import CAP, BUTTON_ROOT, OBJECT_ROOT, Rows, Manifolds, measure, validate_frame

DIGITS = {'0':'abcdef','1':'bc','2':'abdeg','3':'abcdg','4':'bcfg',
          '5':'acdfg','6':'acdefg','7':'abc','8':'abcdefg','9':'abcdfg'}


def instruction():
    return json.loads((Path(__file__).with_name('config')/'instructions.json').read_text())['entries'][0]['en']


def validate(spec, ctx):
    if spec.instance_id.startswith('VAL-F2-composition-') or spec.config.get('gap_composition_authority')=='gap-composition-v1':
        from mind2act_eval.gap_composition_inputs import validate as validate_gap
        validate_gap(spec, ctx.assets)
    if spec.task_id != 'F2' or spec.config_version != VERSION:
        raise ValueError('F2 source/rule identity mismatch')
    if spec.clock.physics_hz != 120 or spec.clock.control_hz not in (30,120) or spec.clock.rgb_hz != 30:
        raise ValueError('F2 requires 120 Hz physics, explicit 30/120 control and 30 Hz RGB')
    cfg = spec.config
    source = to_json(cfg['source_case'])
    if source['difficulty'] != spec.difficulty:
        raise ValueError('F2 requested source condition differs from the frozen case')
    if cfg['instruction'] != instruction():
        raise ValueError('F2 instruction differs from canonical delivery')
    from mind2act_eval.sim.source_assets import asset_bytes, source_file
    source_file(ctx.assets, 'station')
    c = effective_config(source, input_version=cfg['input_protocol_version'])
    start = cfg['source_start_tick']
    receipt = None
    if 'source_receipt' in spec.assets:
        origin_path = ctx.assets.resolve('source_receipt')
        raw = asset_bytes(ctx.assets, 'source_receipt')
        receipt = dict(path=str(origin_path),sha256=cfg['source_receipt_sha256'],instance_id=cfg['source_instance_id'])
    origin = read_origin(receipt,spec.difficulty,source_bytes=raw if receipt is not None else None)
    validate_start(start,origin)
    if origin is not None:
        if origin['source_clock'] != {'physics_hz':spec.clock.physics_hz,'control_hz':spec.clock.control_hz,'rgb_hz':spec.clock.rgb_hz}:
            raise ValueError('F2 source replay clock differs from the frozen instance')
    return source,c,start,receipt


def prepare_task_contacts(stage):
    from pxr import UsdPhysics, PhysxSchema
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Task/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


class FeedbackDisplay:
    def __init__(self,stage):
        self.stage,self.current=stage,None

    def update(self,public):
        from pxr import UsdShade
        digit=(public.get('feedback') or {}).get('correct_positions')
        if digit == self.current:
            return
        for segment in 'abcdefg':
            material='DigitOn_amber' if segment in DIGITS.get(str(digit),'') else 'DigitOff'
            prim=self.stage.GetPrimAtPath('/World/Task/Design/Count/Segment_'+segment)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(
                self.stage.GetPrimAtPath('/World/Materials/Four_'+material)))
        self.current=digit


class ExperimentScene:
    termination_kind = 'events'

    def __init__(self,spec,ctx,source,c,start,origin_receipt,bodies,button,cap,contacts):
        self.spec,self.ctx,self.source,self.c=spec,ctx,source,c
        self.start,self.origin_receipt=start,origin_receipt
        self.bodies,self.button,self.cap,self.contacts=bodies,button,cap,contacts
        self.machine=Task(c,source_start_tick=start)
        self.display=FeedbackDisplay(ctx.sim.stage)
        self.terminal_tick=None
        self.initial=self._measure(0)
        self.latest=self.initial
        validate_frame(c,self.initial,native_tick=0,source_start=start)
        self.machine.sample(self.initial['f2_raw'])
        self.rows=Rows(Path(ctx.output_dir)/'private'/'F2'/'measurements.jsonl')

    def _measure(self,tick):
        return measure(tick,self.start,self.c,self.bodies,self.button,self.cap,self.contacts,self.ctx.world)

    def before_step(self,tick):
        self.contacts.events.clear()

    def after_step(self,tick):
        row=self._measure(tick)
        validate_frame(self.c,row,native_tick=tick,source_start=self.start)
        if self.machine.status=='running':
            self.machine.sample(row['f2_raw'])
            if self.machine.status!='running':
                self.terminal_tick=tick
        self.rows.append(row)
        self.latest=row
        self.display.update(self.machine.public())

    def observe(self,tick):
        return Observation(tick,tick/120,**self.ctx.robot.observe(),instruction=self.spec.config['instruction'],
                           protocol_version=self.spec.protocol_version)

    def evidence(self,tick):
        from mind2act_eval.gap_composition_inputs import evidence_fields
        return NativeEvidence('F2',self.spec.instance_id,tick,dict(source_case=self.source,**evidence_fields(self.spec),
            input_protocol_version=self.spec.config['input_protocol_version'],source_start_tick=self.start,
            source_origin_receipt=self.origin_receipt,initial=self.initial,
            measurements=self.rows.receipt(),terminal_tick=self.terminal_tick))

    def is_complete(self):
        machine = getattr(self, 'machine', None)
        return (machine is not None and not machine.pending_release and
                (machine.status == 'success' or len(machine.attempts) >= self.c['attempt_budget']))

    def health(self):
        from math import isfinite
        return Health(all(isfinite(float(v)) for body in (*self.bodies.values(),self.cap)
                          for v in body.get_world_pose()[0]),'F2 native body pose must be finite')

    def result_metadata(self):
        return dict(source_id=SOURCE_ID,rubric=VERSION,input_protocol_version=self.spec.config['input_protocol_version'],
            source_start_tick=self.start,source_last_tick=self.start+self.rows.count,
            source_machine_tick=self.machine.tick,native_physics_ticks=self.rows.count,
            native_machine_physics_ticks=self.machine.physics_samples,terminal_tick=self.terminal_tick,
            original_attempt_budget=self.source['attempt_budget'],attempt_budget=self.c['attempt_budget'],
            valid_submissions=len(self.machine.attempts),pending_confirm_release=self.machine.pending_release,
            native_status=self.machine.status,reason=self.machine.reason,
            source_origin_receipt=self.origin_receipt,measurements=self.rows.receipt(),initialization_physics_ticks=0)

    def events_complete(self) -> bool:
        return self.is_complete()

    def irreversible_failure(self) -> str | None:
        row = getattr(self, 'latest', None)
        if row is None or row['physics_tick'] == 0 or self.machine.status == 'success':
            return None
        floor_z = min(region['floor_z'] for region in self.c['regions'].values())
        for raw in row['f2_raw']['bodies'].values():
            block = Body.parse(raw)
            if block.center[2] < floor_z - .40 and block.released:
                return 'block_fell_off_station'
        return None

    def close(self):
        self.rows.close()
        self.contacts.close()


def build(spec,ctx):
    source,c,start,receipt=validate(spec,ctx)
    if receipt is not None:
        # Validation remains read-only. Archive the exact small source receipt
        # once in this private run, leaving the asset library unchanged.
        from mind2act_eval.sim.source_assets import asset_bytes
        raw=asset_bytes(ctx.assets, 'source_receipt')
        path=Path(ctx.output_dir).resolve()/'private'/'F2'/'source-origin.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.read_bytes()!=raw:
                raise ValueError('F2 private source receipt differs from the selected origin')
        else:
            with path.open('xb') as stream:stream.write(raw)
        receipt=dict(receipt,path=str(path))
    from isaacsim.core.prims import SingleArticulation,SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    import numpy as np
    stage=ctx.sim.stage
    for path in (CAP,BUTTON_ROOT,*(OBJECT_ROOT+name for name in c['objects'])):
        if not stage.GetPrimAtPath(path):
            raise ValueError('F2 prepared source scene is missing '+path)
    ctx.robot.prepare_stage(stage)
    prepare_task_contacts(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world=ctx.world
    bodies={name:world.scene.add(SingleRigidPrim(prim_path=OBJECT_ROOT+name,name='F2_'+name)) for name in c['objects']}
    button=world.scene.add(SingleArticulation(prim_path=BUTTON_ROOT,name='F2ConfirmMechanism'))
    cap=world.scene.add(SingleRigidPrim(prim_path=CAP,name='F2ConfirmCap'))
    ctx.robot.initialize(world)
    for body in bodies.values():
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
    home=np.asarray(c['button']['initial_joint_positions_m'])
    button.set_joint_positions(home)
    button.set_joint_velocities(np.zeros_like(home))
    button.apply_action(ArticulationAction(joint_positions=home))  # task-owned spring, never an X5 write
    contacts=Manifolds()
    contacts.subscribe()
    try:
        return ExperimentScene(spec,ctx,source,c,start,receipt,bodies,button,cap,contacts)
    except BaseException:
        contacts.close()
        raise
