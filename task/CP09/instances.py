"""Exact selected acquisition cohorts and canonical delivered instructions."""
import hashlib,json,math
from importlib.resources import files
from pathlib import PurePosixPath
from mind2act_eval.contracts import InstanceSpec,to_json
from mind2act_eval.robot.task_collision_overlay import DECISION,is_geometry_decision

CONFIG_VERSION='cp09-acquisition-geometry-v1'

def config(name):return json.loads(files(__package__).joinpath('config/'+name+'.json').read_text())
def signature(value):return hashlib.sha256(json.dumps(to_json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def instruction(difficulty,language):
    rows=[r for r in config('instructions') if difficulty in r['d']]
    if language not in ('en','zh') or len(rows)!=1:raise ValueError('CP09 delivered instruction selection differs')
    return rows[0][language]

def validate_case_source(spec):
    """Case admission is separate from instance-independent robot geometry."""
    if spec.task_id!='CP09' or spec.config_version!=CONFIG_VERSION:
        raise ValueError('CP09 task/version differs')
    if 'test_source' in spec.config:
        from mind2act_eval.sim.new15_source_bindings import matches_original_test_binding
        if not matches_original_test_binding(spec):
            raise ValueError('CP09 original test case source binding differs')
    else:
        row=config('selected')[spec.difficulty];c=spec.config
        if (spec.instance_id!=row['instance_id'] or spec.seed!=row['seed']
                or spec.source_id!=row['source_id'] or c.get('h5_sha256')!=row['h5_sha256']
                or c.get('source_case_sha256')!=row['source_case_sha256']):
            raise ValueError('CP09 selected acquisition binding differs')


def validate(spec):
    validate_case_source(spec)
    if spec.task_id!='CP09' or spec.config_version!=CONFIG_VERSION:raise ValueError('CP09 task/version differs')
    row=config('selected')[spec.difficulty];c=spec.config
    if 'test_source' in c:
        from mind2act_eval.original_test_inputs import source_identity
        source_identity(spec)
        if 'h5_sha256' in c:
            raise ValueError('CP09 original test input has no selected training H5')
        if 'station_relocation_manifest' in c or 'station_material_relocation_manifest' in c:
            raise ValueError('CP09 original test opens the exact already-portable acquisition station')
    elif (spec.instance_id!=row['instance_id'] or spec.seed!=row['seed'] or spec.source_id!=row['source_id']
          or c['h5_sha256']!=row['h5_sha256'] or c['physics_budget']!=50400
          or c['source_case_sha256']!=row['source_case_sha256']):
        raise ValueError('CP09 selected acquisition binding differs')
    if (spec.clock.physics_hz!=120 or spec.clock.control_hz!=30 or spec.clock.rgb_hz!=30
        or not is_geometry_decision(c['geometry_decision']) or c['station_config_sha256']!=signature(config('station'))):
        raise ValueError('CP09 selected acquisition binding differs')
    if c['instruction'] not in (instruction(spec.difficulty,'en'),instruction(spec.difficulty,'zh')):
        raise ValueError('CP09 must use delivered difficulty instruction')
    if (list(c['robot_rest'])!=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044]
        or c['dock_rule_status']!='provisional_source_retrieval_facts'
        or ('test_source' not in c and c.get('station_relocation_manifest')!='station_relocation')):
        raise ValueError('CP09 source rest/retrieval condition differs')
    for asset in spec.assets.values():
        path=PurePosixPath(asset)
        if not asset or path.is_absolute() or '..' in path.parts or '\\' in asset:raise ValueError('CP09 requires library asset ids')
    return row


def original_case(spec, assets):
    from mind2act_eval.original_test_inputs import read
    raw=read(spec,assets)
    if (raw['task_id']!='CP09' or raw['difficulty']!=spec.difficulty or raw['split']!='eval'
            or raw['binding']['declared_version']!=spec.source_id
            or raw['config']['seed']!=spec.seed):
        raise ValueError('CP09 original test source identity differs')
    case=raw['config']['source_case']
    if signature(case)!=spec.config['source_case_sha256']:
        raise ValueError('CP09 original test geometry differs')
    return case

def spec_for_selected(difficulty,*,instance_id,station_asset,x5_asset,station_proof_asset,case_asset,station_relocation_asset,asset_version,language='en',protocol_version='v1'):
    row=config('selected')[difficulty]
    if instance_id!=row['instance_id']:raise ValueError('CP09 requires explicit selected training instance')
    spec=InstanceSpec(task_id='CP09',instance_id=instance_id,difficulty=difficulty,source_id=row['source_id'],
        config_version=CONFIG_VERSION,asset_version=asset_version,protocol_version=protocol_version,seed=row['seed'],
        assets=dict(station=station_asset,x5=x5_asset,station_proof=station_proof_asset,source_case=case_asset,station_relocation=station_relocation_asset),
        config=dict(instruction=instruction(difficulty,language),station_relocation_manifest='station_relocation',physics_budget=50400,h5_sha256=row['h5_sha256'],
            source_case_sha256=row['source_case_sha256'],station_config_sha256=signature(config('station')),
            geometry_decision=DECISION,dock_rule_status='provisional_source_retrieval_facts',
            robot_rest=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044]))
    validate(spec);return spec
