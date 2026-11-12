"""PR1 build(spec, ctx): author the rheostat on ctx.sim.stage (opened before the world is created) and return its Scene.

Only the task fixture is built here. Table, lighting, cameras and robots belong to
sim/ and robot/; no robot parameter is touched. Policy name and consumer kind do not
reach this module.
"""
import json
import math
from functools import lru_cache
from importlib import resources
from pathlib import Path

from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation

from mind2act_eval.tasks._precision.stability import window_ticks
from .scorer import HOLD_SECONDS, TOLERANCES
from mind2act_eval.tasks._precision.textures import font_loader
from .evidence import Recorder, ResistanceModel

REQUIRED_CONFIG = ('initial_ohm', 'target_ohm', 'fixture_y_offset_m')


@lru_cache(maxsize=None)
def _config(name: str) -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.PR1').joinpath('config', name).read_text(encoding='utf-8'))


def _reader(stage, path: str):
    """Live pose of the carriage prim; the baseline PR2/PR1 readout also used the USD transform."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(path)

    def read():
        return tuple(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation())
    return read


def validate(spec: InstanceSpec) -> None:
    levels = _config('levels.json')
    missing = [k for k in REQUIRED_CONFIG if k not in spec.config]
    if spec.task_id != 'PR1' or missing:
        raise ValueError(f'not a complete PR1 spec (missing {missing})')
    lo, hi = levels['sampling']['initial_and_target_range_ohm']
    for key in ('initial_ohm', 'target_ohm'):
        if not lo <= spec.config[key] <= hi:
            raise ValueError(f'{key} outside the {lo}-{hi} ohm range')
    if spec.difficulty not in levels['levels']:
        raise ValueError('unknown difficulty')


class RheostatScene:
    def __init__(self, spec, ctx, recorder, instruction, colliders, read, model):
        self.spec, self._ctx, self._recorder = spec, ctx, recorder
        self.instruction, self.colliders = instruction, colliders
        self._read, self._model = read, model

    def before_step(self, tick: int) -> None:
        pass

    def after_step(self, tick: int) -> None:
        self._recorder.record(tick)  # one reading per post-step physics tick, first tick = 1

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz,
                           instruction=self.instruction, protocol_version=self.spec.protocol_version,
                           **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        evidence = self._recorder.evidence(tick)
        return NativeEvidence(evidence.task_id, evidence.instance_id, tick,
                              dict(evidence.values, difficulty=self.spec.difficulty))

    def health(self) -> Health:
        x = self._read()[0]
        if not math.isfinite(x):
            return Health(False, 'carriage position is not finite')
        ohm = self._model.ohm(x)
        if not -1. <= ohm <= 41.:
            return Health(False, f'carriage left the guide bar ({ohm:.2f} ohm)')
        return Health(True)

    def is_success(self) -> bool:
        """Check the latest measured reading at a model-action boundary."""
        try:
            return abs(self._recorder.last - self.spec.config['target_ohm']) <= TOLERANCES[self.spec.difficulty] + 1e-12
        except IndexError:
            return False

    def is_complete(self) -> bool:
        return self.is_success()

    def close(self) -> None:
        self._recorder.dump(Path(self._ctx.output_dir) / 'PR1' / 'measurements.json')


def build(spec: InstanceSpec, ctx: BuildContext) -> RheostatScene:
    from . import mechanism  # needs pxr; kept out of registration
    validate(spec)
    levels, layout = _config('levels.json'), _config('layout.json')
    cfg = spec.config
    z = layout['support_z_m']
    model = ResistanceModel.from_config(levels)
    shift = (0., layout['base_shift_y_m'] + cfg['fixture_y_offset_m'], 0.)
    texture = Path(ctx.output_dir) / 'PR1' / 'ruler_ohm.png'
    mechanism.write_scale_texture(texture, font_loader(ctx.assets, spec))
    stage = ctx.sim.stage
    mechanism.author_geometry(stage, z, model.position_m(cfg['initial_ohm']), texture.resolve(), shift)
    colliders = mechanism.apply_physics(stage, z, shift, layout['physics'])
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    read = _reader(stage, mechanism.CARRIAGE)
    tolerance = levels['levels'][spec.difficulty]['tolerance_ohm']
    start = model.ohm(read()[0])
    if abs(start - cfg['initial_ohm']) > tolerance:
        raise ValueError(f'carriage reads {start:.3f} ohm, instance asks for {cfg["initial_ohm"]}')
    touching = lambda: ctx.sim.contacts.touch(mechanism.CARRIAGE, robot_only=True)
    recorder = Recorder(lambda: model.ohm(read()[0]), touching, spec.task_id, spec.instance_id, cfg['target_ohm'],
                        spec.clock.physics_hz, window_ticks(spec.clock.physics_hz, HOLD_SECONDS))
    instruction = layout['instruction'].replace('{TARGET}', f'{cfg["target_ohm"]:g}')
    return RheostatScene(spec, ctx, recorder, instruction, colliders, read, model)
