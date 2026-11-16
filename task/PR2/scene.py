"""PR2 build(spec, ctx): author the voltage regulator on ctx.sim.stage and return its Scene.

Only the task fixture is built here; table, lighting, cameras and robots belong to sim/ and robot/.
Policy name and consumer kind do not reach this module.
"""
import json
import math
import os
from functools import lru_cache
from importlib import resources
from pathlib import Path

from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation

from mind2act_eval.tasks._precision.stability import window_ticks
from .scorer import HOLD_SECONDS, TOLERANCES
from mind2act_eval.tasks._precision.textures import font_loader
from .evidence import Recorder, VoltageModel, yaw_deg

REQUIRED_CONFIG = ('initial_volt', 'target_volt', 'fixture_y_offset_m')


@lru_cache(maxsize=None)
def _config(name: str) -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.PR2').joinpath('config', name).read_text(encoding='utf-8'))


def _reader(stage, path: str):
    """Live knob yaw in degrees from its USD world transform (baseline knob_value readout)."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(path)

    def read():
        m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)  # row-vector convention: m[0][1] is R[1][0]
        return yaw_deg([[m[0][0], m[1][0]], [m[0][1], m[1][1]]])
    return read


def validate(spec: InstanceSpec) -> None:
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    levels, layout = _config('levels.json'), _config('layout.json')
    missing = [k for k in REQUIRED_CONFIG if k not in spec.config]
    if spec.task_id != 'PR2' or missing:
        raise ValueError(f'not a complete PR2 spec (missing {missing})')
    lo, hi = layout['voltage_range_v']
    for key in ('initial_volt', 'target_volt'):
        if not lo <= spec.config[key] <= hi:
            raise ValueError(f'{key} outside the {lo}-{hi} V scale')
    if spec.difficulty not in levels['levels']:
        raise ValueError('unknown difficulty')
    if spec.source_id != layout['source_ids'][spec.difficulty]:
        raise ValueError(f'source {spec.source_id} is not the {spec.difficulty} baseline')


class VoltageScene:
    def __init__(self, spec, ctx, recorder, instruction, colliders, read, model):
        self.spec, self._ctx, self._recorder = spec, ctx, recorder
        self.instruction, self.colliders = instruction, colliders
        self._read, self._model = read, model
        self._diagnostic = None

    def before_step(self, tick: int) -> None:
        if self._diagnostic is None and os.environ.get('M2A_PR2_CONTACT_DIAGNOSTIC') == '1':
            from .diagnostic import native_probe
            self._diagnostic = native_probe(self._ctx)  # after shared initialization/reset
        if self._diagnostic is not None:
            self._diagnostic.before_step(tick)

    def after_step(self, tick: int) -> None:
        self._recorder.record(tick)  # one reading per post-step physics tick, first tick = 1
        if self._diagnostic is not None:
            self._diagnostic.after_step(tick)

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz,
                           instruction=self.instruction, protocol_version=self.spec.protocol_version,
                           **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        evidence = self._recorder.evidence(tick)
        return NativeEvidence(evidence.task_id, evidence.instance_id, tick,
                              dict(evidence.values, difficulty=self.spec.difficulty))

    def health(self) -> Health:
        self._ctx.robot.verify_task_collision_overlay(self._ctx.sim.stage)
        angle = self._read()
        if not math.isfinite(angle):
            return Health(False, 'knob yaw is not finite')
        volts = self._model.volts(angle)
        if not -10. <= volts <= 270.:
            return Health(False, f'knob left its range ({volts:.1f} V)')
        return Health(True)

    def is_success(self) -> bool:
        """Check the latest measured reading without a release/dwell gate."""
        try:
            return abs(self._recorder.last - self.spec.config['target_volt']) <= TOLERANCES[self.spec.difficulty] + 1e-12
        except IndexError:
            return False

    def is_complete(self) -> bool:
        return self.is_success()

    def close(self) -> None:
        if self._diagnostic is not None:
            self._diagnostic.close()
        self._recorder.dump(Path(self._ctx.output_dir) / 'PR2' / 'measurements.json')


def build(spec: InstanceSpec, ctx: BuildContext) -> VoltageScene:
    from . import mechanism  # needs pxr; kept out of registration
    validate(spec)
    if spec.config.get('layout_split') == 'validation':
        from mind2act_eval.sim.source_assets import source_file
        source_file(ctx.assets, 'station')
    levels, layout = _config('levels.json'), _config('layout.json')
    cfg = spec.config
    z = layout['support_z_m']
    model = VoltageModel.from_config(layout['value_mapping'])
    shift = (0., layout['base_shift_y_m'] + cfg['fixture_y_offset_m'], 0.)
    textures = Path(ctx.output_dir) / 'PR2'
    mechanism.write_textures(textures, font_loader(ctx.assets, spec))
    stage = ctx.sim.stage
    # Restore/configure the official asset before the source contact exception.
    # Shared initialize then reuses this stage instead of replacing the bindings.
    ctx.robot.prepare_stage(stage)
    geometry = mechanism.author_geometry(stage, z, model.yaw_deg(cfg['initial_volt']), textures.resolve(), shift,
                                         layout['knob_diameter_mm'])
    colliders = mechanism.apply_physics(stage, geometry['dial_cy'], geometry['deck'], shift, layout['physics'],
                                        layout['joint_limit_deg'])
    from mind2act_eval.robot.pr2_source_contacts import expected_descriptor
    ctx.robot.apply_task_collision_overlay(stage, expected_descriptor(spec))
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    angle = _reader(stage, mechanism.KNOB)
    tolerance = levels['levels'][spec.difficulty]['tolerance_V']
    start = model.volts(angle())
    if abs(start - cfg['initial_volt']) > tolerance:
        raise ValueError(f'knob reads {start:.3f} V, instance asks for {cfg["initial_volt"]}')
    touching = lambda: ctx.sim.contacts.touch(mechanism.KNOB, robot_only=True)
    recorder = Recorder(lambda: model.volts(angle()), touching, spec.task_id, spec.instance_id, cfg['target_volt'],
                        spec.clock.physics_hz, window_ticks(spec.clock.physics_hz, HOLD_SECONDS))
    instruction = layout['instruction'].replace('{TARGET}', f'{cfg["target_volt"]:g}')
    return VoltageScene(spec, ctx, recorder, instruction, colliders, angle, model)
