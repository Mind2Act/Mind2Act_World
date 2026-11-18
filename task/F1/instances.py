"""Map a baseline per-episode spec (`collection-pilots/F__<tier>__episode_NNNN/spec.json`) to an InstanceSpec.

The case block (mapping, execution start/target) is evaluation truth for the scene and scorer only; it is
never part of an observation. The instance-recorded expert exploration order is dropped.
"""
import json
import math
from functools import lru_cache
from importlib import resources

from mind2act_eval.contracts import ClockSpec, InstanceSpec


@lru_cache(maxsize=None)
def layout() -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.F1').joinpath('config', 'layout.json').read_text(encoding='utf-8'))


def require_baseline_clock(clock) -> None:
    """Feedback, hold, reset and reveal are 120 Hz tick counts; reject any other physics rate."""
    if clock.physics_hz != 120:
        raise ValueError('F1 timing is bound to the 120 Hz baseline physics clock')


def spec_from_row(row: dict, *, asset_version: str, protocol_version: str = None,
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    lay = layout()
    require_baseline_clock(clock)
    cfg = row['config']
    case = dict(cfg['native_inputs']['config'])
    if row['task_id'] != 'F' or row['split'] != 'train' or row['difficulty'] != case['difficulty'].lower():
        raise ValueError('not a consistent F/F1 train instance')
    if sorted(case['mapping']) != sorted(case['buttons']) or case['exploration'] and set(case['exploration']) != set(case['buttons']):
        raise ValueError('F button set differs from mapping')
    protocol = {k: cfg['protocol'][k] for k in ('press_threshold_m', 'release_threshold_m')}
    protocol['parking_error'] = cfg['protocol']['quality']['parking_error']
    keep = ('space', 'buttons', 'mapping', 'execution', 'timing', 'layout')
    config = dict(case={k: case[k] for k in keep}, protocol=protocol, instruction=row['instruction'],
                  physics_budget=cfg['budget_ticks'],
                  support_z=cfg['native_inputs']['manifest']['station']['world_support_z'],
                  robot_rest=[math.radians(d) for d in lay['home_joints_deg']] + [lay['rest_finger_m']] * 2)
    return InstanceSpec(
        task_id='F1', instance_id=row['case_id'], difficulty=row['difficulty'], source_id=lay['source_id'],
        config_version=lay['config_version'], asset_version=asset_version,
        protocol_version=protocol_version or lay['protocol_version'], clock=clock, config=config,
        assets={'station': f"F1/{row['difficulty']}/scene.usda", 'x5': 'robot/x5/ARX.usd'})
