"""PR3 build(spec, ctx): the prepared scene asset plus the baseline marker/nib/board physics.

Baseline source_id pr3-original-source-mainline-pro-sourcepath. The prepared scene.usda is an instance
asset resolved through ctx.assets; nothing is rebuilt from builder source (no AST slicing, no geometry
recomputation) and no CurrentTrainingCase.dependencies is assumed. Robot setup is ctx.robot's.
"""
import json
import math
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import BuildContext, Health, InstanceSpec, NativeEvidence, Observation

from .evidence import TipRecorder

PATH_KEYS = ('level', 'mirror', 'angle_delta_deg', 'offset_xy_mm')
REQUIRED = PATH_KEYS + ('direction_gate',)


@lru_cache(maxsize=None)
def _layout() -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.PR3').joinpath('config', 'layout.json').read_text(encoding='utf-8'))


def validate(spec: InstanceSpec) -> None:
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    missing = [k for k in REQUIRED if k not in spec.config]
    if spec.task_id != 'PR3' or missing:
        raise ValueError(f'not a complete PR3 spec (missing {missing})')
    if type(spec.config['direction_gate']) is not bool:
        raise ValueError('direction_gate must be an explicit boolean')
    if spec.config['level'] != spec.difficulty:
        raise ValueError('PR3 path level does not match the requested difficulty')
    if spec.source_id != _layout()['source_id']:
        raise ValueError(f'{spec.source_id} is not the PR3 baseline source')
    if 'scene' not in spec.assets:
        raise ValueError('PR3 spec must declare the prepared scene asset')
    if spec.clock.physics_hz != _layout()['tick_hz']:
        raise ValueError(f"PR3 is authored at {_layout()['tick_hz']} Hz physics")


def _quat_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _isaac_bodies(world, paths):
    """Native rigid-body handles (registered before the shared robot resets the world)."""
    from isaacsim.core.prims import SingleRigidPrim
    return {path: SingleRigidPrim(prim_path=path, name=path.strip('/').replace('/', '_')) for path in paths}


class TraceScene:
    def __init__(self, spec, ctx, recorder, bodies, ink, tip, board, layout):
        self._settle = layout['normal_completion_settle_ticks']
        self._paper_top_z = layout['paper']['top_z_m']
        self.spec, self._ctx, self._recorder = spec, ctx, recorder
        self._bodies, self._ink, self._tip, self._board = bodies, ink, tip, board
        self._health = Health(True)
        self._ink_ticks = 0
        self.instruction = public_instruction(spec)

    def before_step(self, tick: int) -> None:
        pass

    def after_step(self, tick: int) -> None:
        nib, _ = self._bodies['nib'].get_world_pose()
        nib = np.asarray(nib, dtype=float)
        marker, _ = self._bodies['marker'].get_world_pose()
        if not (np.isfinite(nib).all() and np.isfinite(np.asarray(marker, dtype=float)).all()):
            self._health = Health(False, 'marker/nib pose is not finite')
            return
        point = nib - np.array([0., 0., .0025])
        board = bool(self._ctx.sim.contacts.touch(self._tip, other=self._board))
        valid = self._recorder.sample(tick, board, point)
        self._ink.sample([float(point[0]), float(point[1]), self._paper_top_z], valid)
        if tick % 10 == 0:
            self._ink.flush()

    def observe(self, tick: int) -> Observation:
        return Observation(tick=tick, sim_time_s=tick / self.spec.clock.physics_hz, instruction=self.instruction,
                           protocol_version=self.spec.protocol_version, **dict(self._ctx.robot.observe()))

    def evidence(self, tick: int) -> NativeEvidence:
        return self._recorder.evidence(tick)

    def health(self) -> Health:
        return self._health

    def is_success(self) -> bool:
        """Actual ink must meet both metrics; lifting is not a completion gate."""
        tick = self._recorder._measured
        if not tick:
            return False
        if getattr(self, '_success_check_tick', None) != tick:
            from .scorer import score
            self._success_at_tick = score(self._recorder.evidence(tick)).value == 100.
            self._success_check_tick = tick
        return bool(self._success_at_tick)

    def is_complete(self) -> bool:
        return self.is_success()

    def result_metadata(self) -> dict:
        """Scene settings that change what the score means; the runner stores them in Result.task_metadata."""
        return dict(direction_gate=self.spec.config['direction_gate'],
                    normal_completion_settle_ticks=self._settle)

    def close(self) -> None:
        self._ink.flush()


def build(spec: InstanceSpec, ctx: BuildContext) -> TraceScene:
    from .mechanism import marker, paths
    validate(spec)
    layout = _layout()
    cfg = spec.config
    parameters = {k: cfg[k] for k in PATH_KEYS}
    if spec.config.get('layout_split') == 'validation':
        from mind2act_eval.sim.source_assets import verify_source_file
        verify_source_file(ctx.assets, 'scene', spec.config['scene_sha256'])
    scene_path = Path(ctx.assets.resolve('scene')).resolve()
    ctx.sim.open_stage(scene_path)
    stage = ctx.sim.stage
    selected = marker.select_level(stage, spec.difficulty)
    nodes = marker.author_marker(stage, selected, layout)
    reference = paths.reference(dict(parameters, offset_xy_mm=list(parameters['offset_xy_mm'])))
    texture = marker.write_reference_texture(Path(ctx.output_dir) / 'PR3' / 'reference_faint.png', reference)
    marker.bind_reference_texture(stage, selected, spec.difficulty, texture.resolve())
    marker.add_station_physics(stage, selected, layout)
    ink = marker.add_ink(stage, layout['paper']['top_z_m'], layout['ink']['width_m'])
    ctx.robot.prepare_stage(stage)  # pinned X5 under the station robot roots; official configuration
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    bodies = _isaac_bodies(ctx.world, [marker.MARKER, nodes['nib_body']])
    ctx.robot.initialize(ctx.world)  # registers the arms, resets the world, rest pose, native verification
    for body in bodies.values():
        body.initialize()
    recorder = TipRecorder(spec.task_id, spec.instance_id, parameters,
                           dict(top_z_m=layout['paper']['top_z_m'], center_y_m=layout['paper']['center_xy_m'][1]),
                           layout['ink'], cfg['direction_gate'], layout['normal_completion_settle_ticks'])
    return TraceScene(spec, ctx, recorder, dict(marker=bodies[marker.MARKER], nib=bodies[nodes['nib_body']]),
                      ink, nodes['tip'], marker.BOARD, layout)
