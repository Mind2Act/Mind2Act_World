"""Bind exact selected PR8 source bytes; public task ID is PR6."""
import hashlib
import json
from mind2act_eval.scoring import SCORING_VERSION
import math
from pathlib import Path
from importlib import resources
import numpy as np
from mind2act_eval.contracts import InstanceSpec, ClockSpec, to_json

SOURCE_ID = CONFIG_VERSION = 'pr8-four-case-native-v4'
RULE_VERSION = SCORING_VERSION


def digest(value):
    return hashlib.sha256(json.dumps(to_json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_rest():
    return [math.radians(v) for v in (0,65,100,-65,0,0)]+[.035,.035]


def public_inputs(difficulty):
    row = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    return next(v['en'] for v in row['entries'] if difficulty in v['d']), row['goal_recipe'][difficulty]


def validate_case(case, difficulty, *, strict=False):
    c = to_json(case)
    if (c['task'] != 'CP11' or c['public_task_id'] != 'PR8' or c['version'] != CONFIG_VERSION
            or c['difficulty'] != difficulty or c['physics_hz'] != 120 or c['control_hz'] != 30
            or c['push_budget'] != 8 or c['stall_ticks'] != 360 or c['stable_ticks'] != 60
            or c['cue_ticks'] != 600 or c['scene_profile'] != 'cp-aligned'):
        raise ValueError('PR6 selected source identity/rule parameters differ')
    if strict and digest({k:v for k,v in c.items() if k != 'case_sha256'}) != c['case_sha256']:
        raise ValueError('PR6 frozen source case SHA mismatch')
    names = {'domino'+str(i+1) for i in range({'easy':5,'medium':7,'hard':9}[difficulty])}
    if set(c['objects']) != names | {'bell','clapper'}:
        raise ValueError('PR6 selected domino count differs')
    if (c['bell']['mechanism'] != 'hollow_shell_passive_clapper_v1'
            or c['prop_solver'] != dict(position_iterations=128,velocity_iterations=8)):
        raise ValueError('PR6 original physical mechanism differs')
    points = np.asarray(c['route_region']['centerline'],dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all() or (np.linalg.norm(np.diff(points,axis=0),axis=1) == 0).any()):
        raise ValueError('PR6 missing original continuous curve band')


def spec_from_source(raw, *, source_spec_sha256, assets, asset_version, station_relocation_asset_id=None):
    if not isinstance(raw,bytes) or hashlib.sha256(raw).hexdigest() != source_spec_sha256:
        raise ValueError('PR6 original source spec SHA mismatch')
    row = json.loads(raw)
    if row['task_id'] != 'PR8' or row['binding']['declared_version'] != CONFIG_VERSION:
        raise ValueError('not a selected PR6/PR8 source spec')
    c = row['config']; case = c['source_case']
    validate_case(case,row['difficulty'])
    if row['binding']['source_case_sha256'] != case['case_sha256'] or row['binding']['seed'] != case['seed']:
        raise ValueError('PR6 source binding differs')
    if not {'station','x5'} <= assets.keys() or any(Path(v).is_absolute() or '..' in Path(v).parts for v in assets.values()):
        raise ValueError('PR6 requires asset-library relative IDs')
    instruction, recipe = public_inputs(row['difficulty'])
    config = dict(source_case=case,source_config_sha256=digest(case),source_spec_sha256=source_spec_sha256,
        source_config_bytes_sha256=row['binding']['source_config_bytes_sha256'],
        scene_sha256=c['native_inputs']['manifest']['scene_sha256'],instruction=instruction,goal_recipe=recipe,
        scoring_version=RULE_VERSION,robot_rest=source_rest())
    if station_relocation_asset_id is not None:
        if station_relocation_asset_id not in assets:
            raise ValueError('PR6 relocation manifest must be a declared asset ID')
        config['station_relocation_manifest'] = station_relocation_asset_id
    return InstanceSpec('PR6',row['case_id'],row['difficulty'],SOURCE_ID,CONFIG_VERSION,asset_version,'v1',
                        seed=case['seed'],clock=ClockSpec(120,30,30),assets=assets,config=config)


def validate(spec):
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    c = spec.config
    if (spec.task_id != 'PR6' or spec.source_id != SOURCE_ID or spec.config_version != CONFIG_VERSION
            or spec.clock != ClockSpec(120,30,30) or spec.seed != c['source_case']['seed']):
        raise ValueError('PR6 source/protocol binding mismatch')
    validate_case(c['source_case'],spec.difficulty)
    if c['source_config_sha256'] != digest(c['source_case']) or tuple(c['robot_rest']) != tuple(source_rest()):
        raise ValueError('PR6 frozen config/reset changed')
    instruction, recipe = public_inputs(spec.difficulty)
    if c['instruction'] != instruction or c['goal_recipe'] != recipe or c['scoring_version'] != RULE_VERSION:
        raise ValueError('PR6 public input/scoring version mismatch')


def main():
    import argparse
    p = argparse.ArgumentParser(description='Bind one explicit selected PR6/PR8 source instance')
    p.add_argument('--source-spec',type=Path,required=True)
    p.add_argument('--source-spec-sha256',required=True)
    p.add_argument('--asset-entries',type=Path,required=True)
    p.add_argument('--asset-version',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); entries = json.loads(a.asset_entries.read_bytes())
    spec = spec_from_source(a.source_spec.read_bytes(),source_spec_sha256=a.source_spec_sha256,
        assets=entries['assets'],asset_version=a.asset_version,station_relocation_asset_id=entries.get('station_relocation_asset_id'))
    with a.output.open('x',encoding='utf-8') as stream:
        json.dump(to_json(spec),stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
