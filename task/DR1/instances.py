"""Map a baseline per-episode spec (source jobs `pilots-v2/<tier>/episode_NNNN/spec.json`) to an InstanceSpec.

Expert-only fields (closed_joint_target_m) are not copied. Scoring needs the slot cells
and the arm owning each slot; both come from the retained placement_slots rule.
"""
import json
import math
from functools import lru_cache
from importlib import resources

from mind2act_eval.contracts import ClockSpec, InstanceSpec


@lru_cache(maxsize=None)
def layout() -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.DR1').joinpath('config', 'layout.json').read_text(encoding='utf-8'))


def require_baseline_clock(clock) -> None:
    """Parcel stability windows and the warmup are tick counts of the 120 Hz baseline; reject any other physics rate."""
    if clock.physics_hz != 120:
        raise ValueError('DR1 timing is bound to the 120 Hz baseline physics clock')


def spec_from_row(row: dict, *, asset_version: str, protocol_version: str = None,
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    lay = layout()
    require_baseline_clock(clock)
    cfg = row['config']
    native = cfg['native_inputs']
    count = cfg['parameters']['count']
    if row['task_id'] != 'FR1' or row['split'] != 'train' or count != lay['count_by_difficulty'][row['difficulty']]:
        raise ValueError('not a consistent FR1/DR1 train instance')
    items = [{k: v for k, v in item.items() if k != 'closed_joint_target_m'} for item in cfg['source_recipe']['items']]
    goods = {g['identity']: g for g in native['geometry_manifest']['stream_goods']}
    stream = [dict(identity=g['identity'], size_m=g['size_m'], label=g['label']) for g in native['manifest']['stream_goods']]
    for g in stream:
        if g['identity'] not in goods:
            raise ValueError('stream good has no geometry: ' + g['identity'])
    config = dict(
        parameters=cfg['parameters'],
        source_recipe=dict(cfg['source_recipe'], items=items),
        belt_bounds_m=native['manifest']['layout']['belt_bounds_m'],
        stream_goods=stream,
        geometry_paths={k: v['path'] for k, v in goods.items()},
        tray_bounds_m=native['manifest']['layout']['inspection_tray_bounds_m'],
        instruction=row['instruction'],
        scene_sha256=native['scene_sha256'],
        station_robots=native['station']['robots'],
        physics_budget=cfg['physics_budget'],
        robot_rest=[math.radians(d) for d in lay['rest_joints_deg']] + [lay['rest_finger_m']] * 2)
    return InstanceSpec(
        task_id='DR1', instance_id=row['case_id'], difficulty=row['difficulty'], source_id=lay['source_id'],
        config_version=lay['config_version'], asset_version=asset_version,
        protocol_version=protocol_version or lay['protocol_version'], clock=clock, config=config,
        assets={'station': f"DR1/{row['difficulty']}/scene.usda", 'x5': 'robot/x5/ARX.usd'})
