"""Selected PR5 prepared scene on the framework X5 and simulator services.

The CLI opens assets['station'] through its hash-pinned relocation seam before
build. Geometry, camera, contact material and physics opinions stay authored.
"""
import hashlib
import json
from importlib import resources

from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, Observation, to_json
from .evidence import Recorder, STAMP
from .instances import SOURCE_REVISION, validate
from .scorer import RULE_VERSION


def checked_inputs(spec, assets):
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    case = validate(spec)
    paths = {}
    for asset_id in ('station', 'scene_config', 'authoring_receipt', 'ink_clean', 'ink_smeared'):
        from mind2act_eval.sim.source_assets import read_source_bytes, verify_source_file
        path = assets.resolve(asset_id)
        expected = spec.config['asset_sha256'][asset_id]
        if path.suffix == '.json':
            read_source_bytes(assets,asset_id,expected,error="PR5 source asset digest mismatch: "+asset_id)
        else:
            verify_source_file(assets,asset_id,expected,error="PR5 source asset digest mismatch: "+asset_id)
        paths[asset_id] = path
    authored = json.loads(read_source_bytes(assets,'scene_config',spec.config['asset_sha256']['scene_config']))
    receipt = json.loads(read_source_bytes(assets,'authoring_receipt',spec.config['asset_sha256']['authoring_receipt']))
    pin = json.loads(resources.files(__package__).joinpath('config/source.json').read_text())
    if (authored['seed'] != to_json(case) or authored['face_side_m'] != .050
            or authored['plate_top_m'] != .772 or authored['card_top_m'] != .7768
            or receipt['source_revision'] != SOURCE_REVISION
            or receipt['scene_author_sha256'] != pin['scene_author_sha256']
            or receipt['common_sha256'] != pin['common_sha256']):
        raise ValueError('PR5 prepared scene/config does not match the selected source')
    return paths


def _stamp_body(world):
    from isaacsim.core.prims import SingleRigidPrim
    return world.scene.add(SingleRigidPrim(STAMP, name='PR5StampA'))


def _tcp_positions(stage):
    """Read native link6 frames, with the source URDF FK's +145mm X TCP offset."""
    from pxr import Gf, UsdGeom
    cache = UsdGeom.XformCache()
    result = {}
    for side in ('Left', 'Right'):
        prim = stage.GetPrimAtPath('/World/Robot' + side + '/Asset/link6')
        if not prim:
            raise ValueError('PR5 missing measured link6 frame: ' + side)
        result[side] = list(cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(.145, 0., 0.)))
    return result


class StampScene:
    def __init__(self, spec, ctx, body, recorder, contacts, subscription, ink):
        self.spec, self.ctx, self.body, self.recorder = spec, ctx, body, recorder
        self.contacts, self.subscription, self.ink = contacts, subscription, ink
        self._health = Health(True)

    def before_step(self, tick):
        self.contacts.clear()

    def after_step(self, tick):
        p, q = self.body.get_world_pose()
        try:
            emitted = self.recorder.record(tick, p, q, _tcp_positions(self.ctx.sim.stage), self.contacts.snapshot())
        except ValueError as error:
            self._health = Health(False, str(error))
            raise
        self.ink.add(emitted)

    def observe(self, tick):
        return Observation(tick=tick, sim_time_s=tick/self.spec.clock.physics_hz,
            instruction=self.spec.config['instruction'], protocol_version=self.spec.protocol_version,
            **dict(self.ctx.robot.observe()))

    def evidence(self, tick):
        return self.recorder.evidence(tick)

    def health(self):
        return self._health

    def is_success(self):
        return bool(self.recorder.frame_count and self.recorder.facts()['success'])

    def is_complete(self):
        return self.is_success()

    def result_metadata(self):
        return dict(source_id=self.spec.source_id, source_revision=SOURCE_REVISION,
                    source_config_sha256=self.spec.config['source_config_sha256'],
                    source_asset_sha256=dict(self.spec.config['asset_sha256']),
                    task_protocol_version=RULE_VERSION, scoring_version=RULE_VERSION,
                    bootstrap_physics_ticks=120, tcp_measurement='native_link6_plus_145mm_x')

    def close(self):
        unsubscribe = getattr(self.subscription, 'unsubscribe', None)
        try:
            if callable(unsubscribe):
                unsubscribe()
        finally:
            self.subscription = None
            self.recorder.close()


def build(spec: InstanceSpec, ctx: BuildContext):
    from .mechanism.contacts import ContactReadings, subscribe
    from .mechanism.visuals import Ink
    paths = checked_inputs(spec, ctx.assets)
    stage = ctx.sim.stage
    for path in (STAMP, '/World/Task/CardA', '/World/Task/HolderA', '/World/Task/Board'):
        if not stage.GetPrimAtPath(path):
            raise ValueError('PR5 selected source station is not open: ' + path)
    ctx.robot.prepare_stage(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    body = _stamp_body(ctx.world)
    ctx.robot.initialize(ctx.world)
    body.initialize()
    case = to_json(spec.config['source_case'])
    holder = case['stamps'][0]['holder_xy']
    body.set_world_pose([*holder, .772], [1., 0., 0., 0.])
    body.set_linear_velocity([0., 0., 0.])
    body.set_angular_velocity([0., 0., 0.])
    # Original source reset settles for 120 real ticks before logical tick 1.
    # This is distinct from the framework's render-only camera warmup.
    for _ in range(120):
        ctx.sim.step(render=False)
    contacts = ContactReadings()
    subscription = subscribe(contacts)
    try:
        ink = Ink(stage, dict(clean=paths['ink_clean'], smeared=paths['ink_smeared']), case['stamps'][0]['card_xy'])
        recorder = Recorder(spec.task_id, spec.instance_id, case,
                            trace_path=ctx.output_dir / 'PR5' / 'raw-trace.jsonl')
        return StampScene(spec, ctx, body, recorder, contacts, subscription, ink)
    except BaseException:
        subscription.unsubscribe()
        raise
