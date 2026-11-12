"""Map an instance-table row of the PR3 baseline to an InstanceSpec.

The scene is an instance asset (`scene`: prepared scene.usda, relocated offline from the original
accepted scene); the target path is derived from the four path parameters. Expert-only fields (arm, writing
yaw, grasp, lift goals) are not copied. `direction_gate` is explicit (no default): True only once the
direction cue is proven visible to models in all three difficulties (CommonInput PR3.md section 9).
"""
from mind2act_eval.contracts import ClockSpec, InstanceSpec

SOURCE_ID = 'pr3-original-source-mainline-pro-sourcepath'
CONFIG_VERSION = 'pr3-spec-20261030-v1'


def spec_from_row(row: dict, *, scene_path: str, direction_gate: bool, asset_version: str, protocol_version: str = 'v1',
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    case = row['source_case']
    p = case['parameters']
    if row['task'] != 'PR3' or case['task'] != 'PR3' or p['level'] != row['difficulty']:
        raise ValueError('row is not a consistent PR3 instance')
    return InstanceSpec(
        task_id='PR3', instance_id=case['case_id'], difficulty=row['difficulty'], source_id=SOURCE_ID,
        config_version=CONFIG_VERSION, asset_version=asset_version, protocol_version=protocol_version, clock=clock,
        assets={'scene': scene_path},
        config=dict(level=p['level'], mirror=bool(p['mirror']), angle_delta_deg=p['angle_delta_deg'],
                    offset_xy_mm=list(p['offset_xy_mm']), direction_gate=direction_gate))
