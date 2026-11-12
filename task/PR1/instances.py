"""Map an instance-table row of the baseline source to an InstanceSpec.

Row layout is the baseline `fixed150.json` (source_id pr1-original-source-uniform-run925de9).
Expert-only fields (grasp yaw, contact tilt, arm choice) are deliberately not copied.
"""
from mind2act_eval.contracts import ClockSpec, InstanceSpec

SOURCE_ID = 'pr1-original-source-uniform-run925de9'
# Public ids -> relative paths below the asset root; the pinned DejaVu files are an asset-gap-list entry.
FONT_ASSETS = {'font_sans': 'fonts/DejaVuSans.ttf', 'font_sans_bold': 'fonts/DejaVuSans-Bold.ttf'}
CONFIG_VERSION = 'pr1-measured-stability-20260930-v2'


def spec_from_row(row: dict, *, asset_version: str, protocol_version: str = 'v1',
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    case = row['source_case']
    if row['task'] != 'PR1' or case['task'] != 'PR1' or case['parameters']['difficulty'] != row['difficulty']:
        raise ValueError('row is not a consistent PR1 instance')
    return InstanceSpec(
        task_id='PR1', instance_id=case['case_id'], difficulty=row['difficulty'], source_id=SOURCE_ID,
        config_version=CONFIG_VERSION, asset_version=asset_version, protocol_version=protocol_version,
        clock=clock, assets=FONT_ASSETS,
        config=dict(initial_ohm=case['parameters']['initial'], target_ohm=case['parameters']['target'],
                    fixture_y_offset_m=row['source_protocol']['fixture_y_offset_m']))
