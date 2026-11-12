from mind2act_eval.scoring import SCORING_VERSION
"""Bind a selected PR5 training/evaluation source row without regenerating its seed."""
import hashlib
import json
import math
from importlib import resources
from pathlib import Path

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json

SOURCE_ID = 'pr5-eval-controller-v3-lower-lift'
SOURCE_REVISION = 'b6432b408607981059acf36ee668b731e36d70a0'
CONFIG_VERSION = 'single-square-head-arrow-tight-box-20261022'
REQUIRED_ASSETS = frozenset(('station', 'x5', 'scene_config', 'authoring_receipt', 'ink_clean', 'ink_smeared'))


def digest(value):
    return hashlib.sha256((json.dumps(to_json(value), sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest()


def instruction(difficulty):
    rows = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    return next(row['en'] for row in rows['entries'] if difficulty in row['d'])


def goal_recipe(difficulty):
    rows = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    return rows['goal_recipe'][difficulty]


def source_rest():
    reset = json.loads(resources.files(__package__).joinpath('config/source.json').read_text())['source_reset_joint_state']
    return [math.radians(value) for value in reset['arm_degrees']] + reset['finger_positions_m']


def validate_case(case, difficulty):
    from math import isfinite
    if case['level'] != difficulty or case['design'] != CONFIG_VERSION:
        raise ValueError('PR5 source design/difficulty mismatch')
    if case['split'] not in ('train', 'eval') or type(case['index']) is not int or case['index'] < 0:
        raise ValueError('PR5 requires an explicit original source split/index')
    if len(case['stamps']) != 1 or case['stamps'][0]['id'] != 'A':
        raise ValueError('PR5 requires the source square stamp A')
    stamp = case['stamps'][0]
    for key in ('card_xy', 'holder_xy', 'box_wh'):
        if len(stamp[key]) != 2 or not all(type(v) in (int, float) and isfinite(v) for v in stamp[key]):
            raise ValueError('invalid PR5 ' + key)
    expected = {'easy': (.100, 5.), 'medium': (.080, 2.), 'hard': (.074, 1.)}[difficulty]
    if tuple(stamp['box_wh']) != (expected[0], expected[0]) or stamp['yaw_tol'] != expected[1]:
        raise ValueError('PR5 geometry/tolerance differs from selected training source')
    if type(stamp['target_yaw']) not in (int, float) or not isfinite(stamp['target_yaw']):
        raise ValueError('invalid PR5 target heading')
    return case


def spec_from_source(row, *, assets, asset_version, protocol_version='v1', station_relocation_asset_id=None):
    """Original paths/annotations/expert parameters never enter the new spec."""
    if row['task_id'] != 'PR5' or row['binding']['source_revision'] != SOURCE_REVISION:
        raise ValueError('wrong PR5 source identity/revision')
    difficulty = row['difficulty']
    case = to_json(validate_case(row['config']['source_case'], difficulty))
    binding = row['binding']
    if (binding['config_sha256'] != digest(case)
            or binding['source_seed'] != dict(split=case['split'], index=case['index'])
            or binding['declared_version'] != CONFIG_VERSION or binding['native_task_id'] != 'PR5'):
        raise ValueError('PR5 source configuration differs from its original binding')
    if not REQUIRED_ASSETS <= assets.keys():
        raise ValueError('PR5 missing declared task/scene asset IDs')
    if any(Path(value).is_absolute() or '..' in Path(value).parts for value in assets.values()):
        raise ValueError('PR5 assets must be paths below the asset-library root')
    names = dict(scene_config='scene-config.json', authoring_receipt='source-authoring.json',
                 ink_clean='ink_clean.png', ink_smeared='ink_smeared.png')
    closure = row['config']['native_inputs']['manifest']['asset_closure']
    pins = {'station': row['config']['scene_sha256']}
    for asset_id, name in names.items():
        matches = {member['sha256'] for member in closure if Path(member['path']).name == name}
        if len(matches) != 1:
            raise ValueError('PR5 source closure lacks a unique digest for ' + asset_id)
        pins[asset_id] = matches.pop()
    config = dict(source_case=case, source_config_sha256=digest(case), asset_sha256=pins,
                  instruction=instruction(difficulty), goal_recipe=goal_recipe(difficulty),
                  scoring_version=SCORING_VERSION, robot_rest=source_rest())
    if station_relocation_asset_id is not None:
        if station_relocation_asset_id not in assets:
            raise ValueError('PR5 station relocation manifest must be a declared asset ID')
        config['station_relocation_manifest'] = station_relocation_asset_id
    return InstanceSpec(task_id='PR5', instance_id=row['case_id'], difficulty=difficulty,
        source_id=SOURCE_ID, config_version=CONFIG_VERSION, asset_version=asset_version,
        protocol_version=protocol_version, seed=case['index'], clock=ClockSpec(120, 30, 30), assets=assets,
        config=config)


def validate(spec):
    if (spec.task_id != 'PR5' or spec.source_id != SOURCE_ID or spec.config_version != CONFIG_VERSION
            or spec.clock.physics_hz != 120 or not REQUIRED_ASSETS <= spec.assets.keys()):
        raise ValueError('not a complete PR5 source binding')
    case = validate_case(spec.config['source_case'], spec.difficulty)
    if spec.seed != case['index'] or spec.config['source_config_sha256'] != digest(case):
        raise ValueError('PR5 source config/index changed')
    if tuple(spec.config['robot_rest']) != tuple(source_rest()):
        raise ValueError('PR5 reset joint state differs from selected source')
    if spec.config['instruction'] != instruction(spec.difficulty):
        raise ValueError('PR5 instruction differs from the delivered instruction')
    if (spec.config['goal_recipe'] != goal_recipe(spec.difficulty)
            or spec.config['scoring_version'] != SCORING_VERSION):
        raise ValueError('PR5 input/scoring rule binding mismatch')
    return case


def main():
    """Produce a portable spec from one explicitly selected immutable source row."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--source-spec', type=Path, required=True)
    parser.add_argument('--source-spec-sha256', required=True)
    parser.add_argument('--asset-entries', type=Path, required=True)
    parser.add_argument('--asset-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_spec.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.source_spec_sha256:
        raise ValueError('PR5 source spec SHA256 mismatch')
    entries = json.loads(args.asset_entries.read_bytes())
    spec = spec_from_source(json.loads(raw), assets=entries['assets'], asset_version=args.asset_version,
        station_relocation_asset_id=entries.get('station_relocation_asset_id'))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(to_json(spec), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
