"""DR2 build(spec, ctx): prepared FR2 panel scene on the shared X5; cue clock is environment-owned."""
import math

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation

from .event_clock import EventClock
from .evidence import ContactTable, Recorder
from .instances import layout, validate_case


def make_clock(spec, lay):
    c, hz = spec.config, spec.clock.physics_hz
    return EventClock(list(c['sequence']), round(c['press_window_s'] * hz), round(c['release_window_s'] * hz),
                      round(c['inter_event_gap_s'] * hz), lay['release_depression_m'], lay['press_depression_m'])


class ButtonScene:
    def __init__(self, spec, ctx, recorder, clock, contacts, subscription, stage, ring_paths, lay):
        self.spec, self._ctx, self._rec, self.clock = spec, ctx, recorder, clock
        self._contacts, self._sub, self._stage, self._rings, self._lay = contacts, subscription, stage, ring_paths, lay
        self.instruction = public_instruction(spec)
        self._lit = None

    def before_step(self, tick: int) -> None:
        pass

    def after_step(self, tick: int) -> None:
        from . import mechanism
        self._rec.record(tick)
        current = self.clock.current
        lit = current['button'] if current and current['press_tick'] is None else None
        if lit != self._lit or tick == 1:
            mechanism.set_rings(self._stage, self._rings, lit, self._lay)
            self._lit = lit

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz, instruction=self.instruction,
                           protocol_version=self.spec.protocol_version, **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        c, hz = self.spec.config, self.spec.clock.physics_hz
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick, dict(
            difficulty=self.spec.difficulty, sequence=list(c['sequence']), press_ticks=self.clock.press_ticks, release_ticks=self.clock.release_ticks,
            gap_ticks=self.clock.gap_ticks, release_depression_m=self._lay['release_depression_m'],
            press_depression_m=self._lay['press_depression_m'], ticks=self._rec.ticks,
            live_records=[{k: r[k] for k in ('index', 'button', 'cue_tick', 'press_tick', 'release_tick', 'status', 'reason')} for r in self.clock.records], live_events=[dict(e) for e in self.clock.events],
            unexpected_contacts=self._contacts.unexpected()))

    def health(self) -> Health:
        if self._rec.ticks and not all(math.isfinite(x) for x in self._rec.ticks[-1]['button_depression_m']):
            return Health(False, 'button depression is not finite')
        return Health(True)

    def events_complete(self) -> bool:
        """Finish the ten-cue sequence even when some/all responses were wrong."""
        return bool(self.clock.done)

    def is_complete(self) -> bool:
        return self.events_complete()

    def close(self) -> None:
        unsubscribe = getattr(self._sub, 'unsubscribe', None)
        if callable(unsubscribe):
            unsubscribe()


def build(spec: InstanceSpec, ctx: BuildContext) -> ButtonScene:
    from isaacsim.core.prims import SingleArticulation
    from . import mechanism
    lay = layout()
    cfg = spec.config
    if spec.task_id != 'DR2' or cfg['press_window_s'] != lay['press_window_s'][spec.difficulty]:
        raise ValueError('not a consistent DR2 spec')
    stage = ctx.sim.stage
    ctx.robot.restore_asset(stage)
    ctx.robot.configure_stage(stage)
    # station base poses go to the shared FixedBase joints; the legacy anchors are switched off (as the source did)
    ctx.robot.mount_bases(stage, [dict(r) for r in cfg['station_robots']], legacy_anchors=True)
    ctx.robot.attach_cameras(stage)  # official camera attachment is kept; no detached sensor bodies
    rings = mechanism.author_panel(stage, lay)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    buttons = [world.scene.add(SingleArticulation(prim_path='/World/CallPanel/Button%d/Asset' % i, name='Button%d' % i))
               for i in range(lay['buttons'])]
    ctx.robot.initialize(world)
    initial = [float(b.get_joint_positions()[0]) for b in buttons]
    contacts = ContactTable()
    clock = make_clock(spec, lay)
    body = lay['button_body']
    recorder = Recorder(
        lambda: [initial[i] - float(buttons[i].get_joint_positions()[0]) for i in range(len(buttons))],
        lambda: [any(contacts.touch('/World/CallPanel/Button%d/Asset/%s' % (i, body), robot_only=True, link=l)
                     for l in lay['finger_links']) for i in range(len(buttons))],
        clock)
    mechanism.set_rings(stage, rings, None, lay)
    sub = mechanism.subscribe_contacts(contacts)
    return ButtonScene(spec, ctx, recorder, clock, contacts, sub, stage, rings, lay)
