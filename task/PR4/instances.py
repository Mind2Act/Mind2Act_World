"""Exact source-byte binding with all physical conditions preserved."""
from mind2act_eval.scoring import SCORING_VERSION
import hashlib
import json
import math
from pathlib import Path
from importlib import resources
from mind2act_eval.contracts import InstanceSpec,ClockSpec,to_json

SOURCE_ID='pr4-original-time-v3'
CONFIG_VERSION='height-study-raised20mm-seeded-translation-v1'
RULE_VERSION=SCORING_VERSION


def digest(value):
    return hashlib.sha256(json.dumps(to_json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_rest():
    return [math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044]


def public_inputs(difficulty):
    r=json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    return next(v['en'] for v in r['entries'] if difficulty in v['d']),r['goal_recipe'][difficulty]


def portable_case(value):
    c=to_json(value)
    # These strings are historical provenance, never runtime asset locations.
    # Their original full-spec and config SHA remain bound separately.
    for key in ('historical_source','panel_source'):
        c['origin'].pop(key,None)
    return c


def validate_case(c,difficulty):
    o=c['origin']
    if (c['level']!=difficulty or o['grid'] not in ([81,81],(81,81)) or o['particles']!=6561
            or o['physical_model']!='native_surface_FEM_cloth' or o['attachments'] or o['kinematic_nodes']
            or o['adhesion']!=0 or o['youngs_modulus_pa']!=1e7 or o['dynamic_friction']!=1.
            or o['body_path']!='/World/DustCloth' or 'historical_source' in o or 'panel_source' in o):
        raise ValueError('PR4 original free native FEM physical model differs')
    if not isinstance(c['source_seed'],int) or c['initial_points_sha256']!=o['initial_points_sha256']:
        raise ValueError('PR4 selected initial geometry/seed missing')


def spec_from_source(raw,*,source_spec_sha256,assets,asset_version,station_relocation_asset_id=None):
    if not isinstance(raw,bytes) or hashlib.sha256(raw).hexdigest()!=source_spec_sha256:
        raise ValueError('PR4 original source spec SHA mismatch')
    row=json.loads(raw);c=row['config'];original=c['source_case']
    if (row['task_id']!='PR4' or row['binding']['declared_version']!='pr4-seeded-translation-v1'
            or row['binding']['task_definition_version']!=CONFIG_VERSION or digest(original)!=row['binding']['config_sha256']):
        raise ValueError('PR4 source identity/config binding mismatch')
    case=portable_case(original);validate_case(case,row['difficulty'])
    if not {'station','x5'}<=assets.keys() or any(Path(v).is_absolute() or '..' in Path(v).parts for v in assets.values()):
        raise ValueError('PR4 requires asset-library relative IDs')
    instruction,recipe=public_inputs(row['difficulty'])
    config=dict(source_case=case,portable_config_sha256=digest(case),source_config_sha256=digest(original),
        source_spec_sha256=source_spec_sha256,scene_sha256=c['scene_sha256'],instruction=instruction,
        goal_recipe=recipe,robot_rest=source_rest(),scoring_version=RULE_VERSION)
    if station_relocation_asset_id is not None:
        if station_relocation_asset_id not in assets:raise ValueError('PR4 undeclared relocation manifest')
        config['station_relocation_manifest']=station_relocation_asset_id
    return InstanceSpec('PR4',row['case_id'],row['difficulty'],SOURCE_ID,CONFIG_VERSION,asset_version,'v1',
        seed=case['source_seed'],clock=ClockSpec(120,30,30),config=config,assets=assets)


def validate(spec):
    c=spec.config
    if (spec.task_id!='PR4' or spec.source_id!=SOURCE_ID or spec.config_version!=CONFIG_VERSION
            or spec.clock!=ClockSpec(120,30,30) or spec.seed!=c['source_case']['source_seed']):
        raise ValueError('PR4 source/protocol binding mismatch')
    validate_case(c['source_case'],spec.difficulty)
    if c['portable_config_sha256']!=digest(c['source_case']) or tuple(c['robot_rest'])!=tuple(source_rest()):
        raise ValueError('PR4 frozen config/reset changed')
    instruction,recipe=public_inputs(spec.difficulty)
    if c['instruction']!=instruction or c['goal_recipe']!=recipe or c['scoring_version']!=RULE_VERSION:
        raise ValueError('PR4 current public input/scoring mismatch')


def main():
    import argparse
    p=argparse.ArgumentParser(description='Bind one selected PR4 source spec, preserving its conditions')
    p.add_argument('--source-spec',type=Path,required=True);p.add_argument('--source-spec-sha256',required=True)
    p.add_argument('--asset-entries',type=Path,required=True);p.add_argument('--asset-version',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();entries=json.loads(a.asset_entries.read_bytes())
    spec=spec_from_source(a.source_spec.read_bytes(),source_spec_sha256=a.source_spec_sha256,assets=entries['assets'],
        asset_version=a.asset_version,station_relocation_asset_id=entries.get('station_relocation_asset_id'))
    with a.output.open('x',encoding='utf-8') as stream:
        json.dump(to_json(spec),stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
