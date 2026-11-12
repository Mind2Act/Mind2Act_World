"""Map a baseline per-episode spec (`b-mainline-20261025-v1/specs/<tier>/episode_NNNN.json`) to an InstanceSpec.

Expert-only data (`expert_geometry` hover/press joint targets, plan files) is not copied.
"""
import json
import math
from functools import lru_cache
from importlib import resources

from mind2act_eval.contracts import ClockSpec, InstanceSpec


@lru_cache(maxsize=None)
def layout() -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.DR2').joinpath('config', 'layout.json').read_text(encoding='utf-8'))


def validate_case(case: dict, difficulty: str) -> None:
    lay = layout()
    if (case['press_window_s'] != lay['press_window_s'][difficulty] or case['release_window_s'] != lay['release_window_s']
            or case['inter_event_gap_s'] != lay['inter_event_gap_s'] or len(case['sequence']) != lay['cues']
            or not all(b in (0, 1, 2) for b in case['sequence'])):
        raise ValueError('FR2 protocol differs from the tier definition')


def spec_from_row(row: dict, *, asset_version: str, protocol_version: str = None,
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    lay = layout()
    if row['task_id'] != 'FR2' or row['difficulty'] not in lay['press_window_s']:
        raise ValueError('not an FR2/DR2 instance')
    case = row['config']['source_case']
    validate_case(case, row['difficulty'])
    if row['case_id'] != case['case_id']:
        raise ValueError('case id differs from source case')
    config = dict(
        sequence=case['sequence'], press_window_s=case['press_window_s'], release_window_s=case['release_window_s'],
        inter_event_gap_s=case['inter_event_gap_s'], button_centers_xy=case['layout']['button_centers_xy'],
        instance_sha256=case['instance_sha256'], instruction=row['instruction'],
        station_robots=row['config']['native_inputs']['manifest']['station']['robots'],
        robot_rest=[math.radians(d) for d in lay['rest_joints_deg']] + [lay['rest_finger_m']] * 2)
    return InstanceSpec(
        task_id='DR2', instance_id=row['case_id'], difficulty=row['difficulty'], source_id=lay['source_id'],
        config_version=lay['config_version'], asset_version=asset_version,
        protocol_version=protocol_version or lay['protocol_version'], clock=clock, config=config,
        assets={'station': 'DR2/source-scene.usda', 'x5': 'robot/x5/ARX.usd'})
