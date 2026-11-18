"""DR4 public ID with unchanged PR7 case IDs and frozen task conditions."""
import hashlib
import json
import math
from pathlib import Path
from importlib import resources

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json

SOURCE_ID = 'pr7-short-tail-v2'
CONFIG_VERSION = 'printed-guide-30-25-20-v1'


def digest(value):
    return hashlib.sha256(json.dumps(to_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def public_inputs(difficulty):
    data = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    instruction = next(r['en'] for r in data['entries'] if difficulty in r['d'])
    return instruction, data['goal_recipe'][difficulty]


def source_rest():
    reset = json.loads(resources.files(__package__).joinpath('config/source.json').read_text())['source_reset_joint_state']
    return [math.radians(value) for value in reset['arm_degrees']] + reset['finger_positions_m']


def validate_case(case, condition, difficulty):
    if (set(case) != {'level', 'radius_mm', 'ball_count', 'source_seed', 'task_condition_sha256'}
            or case['level'] != difficulty or case['radius_mm'] != {'easy':30., 'medium':25., 'hard':20.}[difficulty]
            or type(case['ball_count']) is not int or case['ball_count'] < 1
            or (difficulty in ('easy', 'hard') and case['ball_count'] != {'easy':2, 'hard':4}[difficulty])):
        raise ValueError('DR4 source ball count/radius/difficulty mismatch')
    if (condition['generator_version'] != 'pr7-task-seed-v1' or condition['task_seed'] != case['source_seed']
            or type(condition['task_seed']) is not int or not isinstance(case['task_condition_sha256'], str)
            or len(case['task_condition_sha256']) != 64):
        raise ValueError('DR4 source task condition/seed changed')


def spec_from_source(raw, *, source_spec_sha256, assets, asset_version, station_relocation_asset_id=None):
    # The legacy condition hash has no canonical-encoding definition in this
    # handoff. Preserve it as provenance; pin the WHOLE original source bytes
    # before parsing, then separately bind the portable condition content.
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != source_spec_sha256:
        raise ValueError('DR4 original source spec SHA256 mismatch')
    row = json.loads(raw)
    if row['task_id'] != 'PR7' or row['binding']['declared_version'] != 'pr7-task-seeded-v1':
        raise ValueError('not a selected DR4/PR7 source row')
    c = row['config']
    case, condition = to_json(c['source_case']), to_json(c['task_condition'])
    validate_case(case, condition, row['difficulty'])
    if digest(case) != row['binding']['config_sha256'] or row['binding']['task_definition_version'] != CONFIG_VERSION:
        raise ValueError('DR4 source config binding mismatch')
    if not {'station', 'x5'} <= assets.keys() or any(Path(v).is_absolute() or '..' in Path(v).parts for v in assets.values()):
        raise ValueError('DR4 assets must resolve below the asset-library root')
    instruction, recipe = public_inputs(row['difficulty'])
    config = dict(source_case=case, task_condition=condition, source_config_sha256=digest(case),
        source_spec_sha256=source_spec_sha256, task_condition_content_sha256=digest(condition),
        scene_sha256=c['scene_sha256'], source_support_z_m=c['native_inputs']['manifest']['station']['world_support_z'],
        instruction=instruction, goal_recipe=recipe, scoring_version='assigned7-scoring-20261029-v4')
    config['robot_rest'] = source_rest()
    if station_relocation_asset_id is not None:
        if station_relocation_asset_id not in assets:
            raise ValueError('DR4 relocation manifest must be a declared asset ID')
        config['station_relocation_manifest'] = station_relocation_asset_id
    return InstanceSpec('DR4', row['case_id'], row['difficulty'], SOURCE_ID, CONFIG_VERSION, asset_version,
                        'v1', seed=case['source_seed'], clock=ClockSpec(120,30,30), assets=assets, config=config)


def validate(spec):
    c = spec.config
    if spec.task_id != 'DR4' or spec.source_id != SOURCE_ID or spec.config_version != CONFIG_VERSION or spec.clock.physics_hz != 120:
        raise ValueError('DR4 source/protocol binding mismatch')
    validate_case(c['source_case'], c['task_condition'], spec.difficulty)
    if c['source_config_sha256'] != digest(c['source_case']) or spec.seed != c['source_case']['source_seed']:
        raise ValueError('DR4 frozen instance/seed changed')
    if c['task_condition_content_sha256'] != digest(c['task_condition']):
        raise ValueError('DR4 portable task condition changed')
    if tuple(c['robot_rest']) != tuple(source_rest()):
        raise ValueError('DR4 reset joint state differs from selected source')
    instruction, recipe = public_inputs(spec.difficulty)
    if (c['instruction'] != instruction or c['goal_recipe'] != recipe
            or c['scoring_version'] != 'assigned7-scoring-20261029-v4' or c['source_support_z_m'] != .768):
        raise ValueError('DR4 input/scoring/station version mismatch')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bind one explicit DR4/PR7 source instance to asset-library IDs')
    parser.add_argument('--source-spec', type=Path, required=True)
    parser.add_argument('--source-spec-sha256', required=True)
    parser.add_argument('--asset-entries', type=Path, required=True)
    parser.add_argument('--asset-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_spec.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.source_spec_sha256:
        raise ValueError('DR4 source spec SHA256 mismatch')
    entries = json.loads(args.asset_entries.read_bytes())
    spec = spec_from_source(raw, source_spec_sha256=args.source_spec_sha256, assets=entries['assets'], asset_version=args.asset_version,
                            station_relocation_asset_id=entries.get('station_relocation_asset_id'))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(to_json(spec), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
