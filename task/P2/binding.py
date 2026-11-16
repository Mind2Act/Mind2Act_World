"""Frozen task conditions and prepared-scene identity, never an episode generator."""
import hashlib
import json
from pathlib import Path
from mind2act_eval.contracts import to_json

VERSION='p2-spec-20261030-v1'
from mind2act_eval.scoring import SCORING_VERSION
SOURCE_ID='p-steamer-wrist-continuous'
SOURCE_VERSION='p-steamer-original-order-v1'
DISMANTLING_RULE='p2-table-board-dismantling-20261101-2330-v1'
EVALUATION_MODE='formal'
COUNTS={'easy':(3,0),'medium':(4,0),'hard':(4,2)}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def instruction(difficulty):
    data=json.loads((Path(__file__).with_name('config')/'instructions.json').read_text())
    return next(row['en'] for row in data['entries'] if difficulty in row['d'])


def geometry(row):
    """Static source annotate.runtime_overrides geometry; no pose or physics writes."""
    n=len(row['target_foods_bottom_to_top'])
    return dict(stove_xyz=[-.35,-.06,.768],board_xyz=[.20,0,0],
        pot_geometry_scale=[1.5,1.5,.065/.085],preplaced_tier_translate=[0,0,.065],
        target_initial_xyz={f'Tier{i+1}':[.26,-.12,.784+i*.04] for i in range(n)},
        lid_initial_xyz=[.26,-.12,.784+n*.04],decoy_xyz=row['decoy_xyz'],
        final_xyz={**{f'Tier{i+1}':[-.35,-.06,.873+i*.04] for i in range(n)},
                   'Lid':[-.35,-.06,.873+n*.04]})


def validate_source(row):
    n,decoys=COUNTS[row['difficulty']]
    if (type(row['seed']) is not int or len(row['target_foods_bottom_to_top'])!=n
            or len(row['decoy_xyz'])!=decoys or len(row['decoy_foods'])!=decoys):
        raise ValueError('P2 source tier/object binding mismatch')
    return geometry(row)


def validate(spec,ctx):
    if spec.task_id!='P2' or spec.source_id!=SOURCE_ID or spec.config_version!=VERSION:
        raise ValueError('P2 public/source/rule identity mismatch')
    if spec.clock.physics_hz!=120 or spec.clock.control_hz not in (30,120) or spec.clock.rgb_hz!=30:
        raise ValueError('P2 requires120Hz physics,30/120control and30Hz RGB')
    cfg=spec.config;row=to_json(cfg['source_instance'])
    if cfg.get('dismantling_rule_version')!=DISMANTLING_RULE:
        raise ValueError('P2 requires explicit current table/board dismantling rule')
    if (row['difficulty']!=spec.difficulty or row['seed']!=spec.seed
            or digest(row)!=cfg['source_record_sha256']):
        raise ValueError('P2 frozen source record differs from the requested case')
    if cfg['instruction']!=instruction(spec.difficulty):
        raise ValueError('P2 instruction differs from canonical delivery')
    if cfg['input_protocol_version'] not in (VERSION,SOURCE_VERSION):
        raise ValueError('P2 input protocol must be explicit')
    from mind2act_eval.sim.source_assets import source_file
    source_file(ctx.assets, 'station')
    source_file(ctx.assets, 'task_scene')
    for food in set(row['target_foods_bottom_to_top']+row['decoy_foods']):
        source_file(ctx.assets, 'food_'+food)
    return row,validate_source(row)
