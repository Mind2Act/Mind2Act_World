"""F1 build(spec, ctx): prepared F scene (rail/grid) with spring buttons and an environment-driven slider."""
import math

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, FrozenHistory, Health, InstanceSpec, NativeEvidence, Observation

from .evidence import ContactTable, measure_row
from .instances import layout, require_baseline_clock
from .judge import Judge, plain
from .mechanism import Slider, Timeline


class SliderScene:
    termination_kind = 'result'

    def __init__(self, spec, ctx, parts):
        self.spec, self._ctx = spec, ctx
        self.__dict__.update(parts)
        self.instruction = public_instruction(spec)
        self.rows, self._grade = FrozenHistory(), dict(task_success=False, protocol_violations=[])

    def before_step(self, tick: int) -> None:
        pass

    def after_step(self, tick: int) -> None:
        from pxr import Gf, UsdGeom
        robot = self._ctx.robot.observe()
        q = list(robot['left_arm']) + list(robot['right_arm'])
        case, lay = self.case, self.lay
        depths = {b: self.initial[b] - float(a.get_joint_positions()[0]) for b, a in self.buttons.items()}
        contacts = {b: self.contacts.cap_contact(b) for b in self.buttons}
        arm_pairs = {b: {s: self.contacts.finger_pairs(b, s) for s in ('Left', 'Right')} for b in self.buttons}
        events, failures = self.slider.update(tick, depths, contacts, arm_pairs)
        x, y = self.slider.xy_world()
        self.motion.Set(Gf.Vec3d(x, y, self.slider_z))
        visible = self.goal.ComputeVisibility() != UsdGeom.Tokens.invisible
        row = measure_row(tick, self.slider.stage, q, self.slider, depths, contacts, visible, events, failures)
        self._grade = self.judge.sample(row)
        self.rows = self.rows.append(row)
        released = all(depths[b] < self.protocol['release_threshold_m'] for b in self.buttons)
        self.timeline.advance(tick, self.slider.ready_to_reset(released), self.goal.MakeVisible)

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz, instruction=self.instruction,
                           protocol_version=self.spec.protocol_version, **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        from mind2act_eval.gap_composition_inputs import evidence_fields
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
                              dict(case=self.case, protocol=self.protocol, rows=self.rows, **evidence_fields(self.spec),
                                   contact_summary=[dict(pair=list(p), **r) for p, r in self.contacts.pairs.items()]))

    def health(self) -> Health:
        if self.rows and not all(math.isfinite(d) for d in self.rows[-1]['button_depths'].values()):
            return Health(False, 'button depression is not finite')
        return Health(True)

    def is_complete(self) -> bool:
        return bool(self._grade['task_success'])

    def is_success(self) -> bool:
        return self.is_complete()

    def close(self) -> None:
        unsubscribe = getattr(self.subscription, 'unsubscribe', None)
        if callable(unsubscribe):
            unsubscribe()


def build(spec: InstanceSpec, ctx: BuildContext) -> SliderScene:
    if spec.instance_id.startswith('VAL-F1-reuse-') or spec.config.get('gap_composition_authority')=='gap-composition-v1':
        from mind2act_eval.gap_composition_inputs import validate as validate_gap
        validate_gap(spec, ctx.assets)
    if spec.instance_id.startswith('VAL-F1-gap-') or 'clean_validation_authority' in spec.config:
        from .validation_inputs import validate
        validate(spec, ctx.assets)
    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from pxr import Gf, UsdGeom
    from . import mechanism
    lay = layout()
    require_baseline_clock(spec.clock)
    cfg = spec.config
    case, protocol = plain(cfg['case']), plain(cfg['protocol'])
    if spec.task_id != 'F1' or len(case['buttons']) != (2 if case['space']['kind'] == 'line' else 4):
        raise ValueError('not a consistent F1 spec')
    stage, support_z = ctx.sim.stage, float(cfg['support_z'])
    ctx.robot.restore_asset(stage)
    ctx.robot.attach_cameras(stage)
    mechanism.lift_easy_rail(stage, case['space'], support_z, lay)
    mechanism.author_buttons(stage, case, support_z, lay)
    mechanism.author_colliders(stage)
    motion = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Slider')).GetOrderedXformOps()[0]
    goal = UsdGeom.Imageable(stage.GetPrimAtPath('/World/Target'))
    goal.MakeInvisible()
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    buttons = {b: world.scene.add(SingleArticulation(prim_path='/World/Button%s/Asset' % b, name='Button' + b))
               for b in case['buttons']}
    ctx.robot.initialize(world)
    contacts = ContactTable(lay['finger_links'])
    subscription = mechanism.subscribe_contacts(contacts)
    from mind2act_eval.contracts import Action
    rest = [math.radians(d) for d in lay['home_joints_deg']]
    action = Action(rest, rest, 0., 0.)
    for _ in range(lay['warmup_ticks']):
        ctx.robot.apply(action)
        ctx.sim.step(render=False)
    initial = {b: float(a.get_joint_positions()[0]) for b, a in buttons.items()}
    contacts.pairs.clear()
    slider = Slider(case, lay, protocol['press_threshold_m'], protocol['release_threshold_m'])
    z = mechanism.slider_world_z(case, support_z, lay)
    from .space import slider_xy
    tx, ty = slider_xy(case['layout'], case['space'], case['execution']['target'], lay['slider_end_margin_m'])
    mechanism.align_target(stage, (tx, ty, z))
    x, y = slider.xy_world()
    motion.Set(Gf.Vec3d(x, y, z))
    hold = round(case['timing']['hold_seconds'] * 120)
    return SliderScene(spec, ctx, dict(
        case=case, protocol=protocol, lay=lay, buttons=buttons, initial=initial, contacts=contacts,
        subscription=subscription, slider=slider, motion=motion, goal=goal, slider_z=z,
        judge=Judge(case, protocol, lay), timeline=Timeline(slider, lay, hold)))
