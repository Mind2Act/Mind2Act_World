"""Map an instance-table row of a PR2 baseline source to an InstanceSpec.

Easy/Medium come from the em-edge-side-grasp source, Hard from hard-directional-grasp (K0 decision
2026-11-01; the sources differ only in expert lineage, not in the scene). Expert-only fields
(grasp yaw, contact tilt, arm) are not copied.
"""
import json
from importlib import resources

from mind2act_eval.contracts import ClockSpec, InstanceSpec

# Same pinned fonts as PR1 (asset-gap list in docs/tasks/PR1.md).
FONT_ASSETS = {'font_sans': 'fonts/DejaVuSans.ttf', 'font_sans_bold': 'fonts/DejaVuSans-Bold.ttf'}
CONFIG_VERSION = 'pr2-measured-stability-20260930-v2'


def source_id(difficulty: str) -> str:
    text = resources.files('mind2act_eval.tasks.PR2').joinpath('config', 'layout.json').read_text(encoding='utf-8')
    return json.loads(text)['source_ids'][difficulty]


def spec_from_row(row: dict, *, asset_version: str, protocol_version: str = 'v1',
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    case = row['source_case']
    if row['task'] != 'PR2' or case['task'] != 'PR2' or case['parameters']['difficulty'] != row['difficulty']:
        raise ValueError('row is not a consistent PR2 instance')
    return InstanceSpec(
        task_id='PR2', instance_id=case['case_id'], difficulty=row['difficulty'],
        source_id=source_id(row['difficulty']), config_version=CONFIG_VERSION, asset_version=asset_version,
        protocol_version=protocol_version, clock=clock, assets=FONT_ASSETS,
        config=dict(initial_volt=case['parameters']['initial'], target_volt=case['parameters']['target'],
                    fixture_y_offset_m=row['source_protocol']['fixture_y_offset_m']))
