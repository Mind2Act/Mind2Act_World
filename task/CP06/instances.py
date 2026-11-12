"""CP06 original selected H5/spec whitelist; portable source hashes stay separate."""
import hashlib
import json
from importlib import resources
from mind2act_eval.contracts import to_json

FRAMEWORK='413b3d3a73de257aad6e5a034c3b947dcc52f91d'
INSTRUCTION_SHA='9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae'


def signature(value):
    return hashlib.sha256(json.dumps(to_json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def selected(level):
    return json.loads(resources.files(__package__).joinpath('config/selected.json').read_text())[level]


def instruction(level, language='en'):
    if language not in ('en','zh'):raise ValueError('CP06 delivered language required')
    d=json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    if d['source_sha256']!=INSTRUCTION_SHA or signature(d['instructions'])!='5db7da2d97ed45b992c709b681d2a815af127e90c74bc1804a7695a71940f160':
        raise ValueError('CP06 delivered instruction source changed')
    rows=[r for r in d['instructions'] if level in r['d']]
    if len(rows)!=1:raise ValueError('CP06 difficulty instruction not unique')
    return rows[0][language]


def parse_original(data, level):
    expected=selected(level)
    if hashlib.sha256(data).hexdigest()!=expected['source_spec_sha256']:
        raise ValueError('CP06 original selected H5 instance SHA mismatch')
    row=json.loads(data);cfg=row['config']['task']
    if (row['task_id']!='CP06' or row['case_id']!=expected['instance_id'] or row['split']!='train'
            or row['difficulty']!=level or cfg['candidate']['seed']!=expected['seed']
            or cfg['framework_commit']!=FRAMEWORK or signature(cfg)!=expected['declared_task_sha256']
            or row['binding']['config_sha256']!=signature(cfg)):
        raise ValueError('CP06 original identity/seed/spec binding differs')
    return row


def validate_core(core, level):
    if signature(core)!=signature(selected(level)['source_task']):
        raise ValueError('CP06 selected physical task core changed')
    if core['difficulty']!=level or core['candidate']['seed']!=selected(level)['seed']:
        raise ValueError('CP06 selected instance condition changed')
    if core['framework_commit']!=FRAMEWORK or core['physics_hz']!=120:
        raise ValueError('CP06 source framework/clock differs')
    return core

SOURCE_ID='cp06-train-supplement-v1'
CONFIG_VERSION='cp06-original-resource-projection-v1'


def original_core(raw):
    return {k:v for k,v in raw['config']['task'].items() if k not in
        ('materialization','jobs','nominal_mass_combinations_not_physical_proof','source_hold_binding')}


def validate(spec):
    import math
    from pathlib import PurePosixPath
    row=selected(spec.difficulty);c=to_json(spec.config)
    test='test_source' in c
    if (spec.task_id!='CP06' or spec.config_version!=CONFIG_VERSION
            or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30):
        raise ValueError('CP06 selected original instance/seed/clock differs')
    if test:
        from mind2act_eval.original_test_inputs import source_identity
        if c['source_spec_sha256']!=source_identity(spec)['source_configuration_sha256']:
            raise ValueError('CP06 original test spec SHA differs')
        if (c['source_task']['difficulty']!=spec.difficulty or c['source_task']['candidate']['seed']!=spec.seed
                or c['source_task']['framework_commit']!=FRAMEWORK or c['source_task']['physics_hz']!=120):
            raise ValueError('CP06 original test core identity differs')
    else:
        if spec.instance_id!=row['instance_id'] or spec.seed!=row['seed'] or spec.source_id!=SOURCE_ID:
            raise ValueError('CP06 selected original instance/seed/clock differs')
        validate_core(c['source_task'],spec.difficulty)
        for key in ('source_spec_sha256','declared_task_sha256','original_task_layer_sha256',
                'derived_task_layer_sha256','prepared_scene_sha256','stand_sha256',
                'composition_receipt_sha256','resource_projection_proof_sha256'):
            if c[key]!=row[key]:raise ValueError('CP06 original/resource projection binding differs: '+key)
    if (c['instruction'] not in (instruction(spec.difficulty),instruction(spec.difficulty,'zh'))
            or c['instruction_source_sha256']!=INSTRUCTION_SHA):
        raise ValueError('CP06 requires delivered tier instruction, not an external old page')
    if c['public_parameters']!=public_parameters(c['source_task']):
        raise ValueError('CP06 public inventory/green interval differs from selected source')
    rest=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044]
    if c['robot_rest']!=rest:raise ValueError('CP06 selected source robot rest differs')
    if not test and c['station_relocation_manifest']!='station_mapping':
        raise ValueError('CP06 explicit asset-library station projection required')
    required=({'station','task_source','test_source_request','test_source_raw','source_geometry_proof','x5'}
              if test else {'station','task_source','original_spec','station_mapping','projection_proof','x5'})
    if not required<=set(spec.assets):
        raise ValueError('CP06 original/portable resources incomplete')
    for value in spec.assets.values():
        path=PurePosixPath(value)
        if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
            raise ValueError('CP06 runtime assets must use relative library identifiers')
    return row


def spec_from_original(data, level, *, station_asset, task_asset, original_spec_asset,
                       x5_asset, relocation_asset, projection_asset, asset_version, language='en'):
    import math
    from mind2act_eval.contracts import InstanceSpec
    original=parse_original(data,level);row=selected(level)
    core={k:v for k,v in original['config']['task'].items() if k not in
          ('materialization','jobs','nominal_mass_combinations_not_physical_proof','source_hold_binding')}
    validate_core(core,level)
    config={key:row[key] for key in ('source_spec_sha256','declared_task_sha256',
        'original_task_layer_sha256','derived_task_layer_sha256','prepared_scene_sha256',
        'stand_sha256','composition_receipt_sha256','resource_projection_proof_sha256')}
    config.update(source_task=core,instruction=instruction(level,language),public_parameters=public_parameters(core),
        instruction_source_sha256=INSTRUCTION_SHA,station_relocation_manifest='station_mapping',
        robot_rest=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044])
    spec=InstanceSpec(task_id='CP06',instance_id=row['instance_id'],difficulty=level,
        source_id=SOURCE_ID,config_version=CONFIG_VERSION,asset_version=asset_version,
        protocol_version='v1',seed=row['seed'],config=config,assets=dict(station=station_asset,
        task_source=task_asset,original_spec=original_spec_asset,station_mapping=relocation_asset,projection_proof=projection_asset,x5=x5_asset))
    validate(spec)
    return spec


def original_case(spec,assets):
    from mind2act_eval.original_test_inputs import read
    raw=read(spec,assets);task=raw['config']['task']
    if (raw['task_id']!='CP06' or raw['difficulty']!=spec.difficulty or raw['split']!='eval'
            or raw['binding']['declared_version']!=spec.source_id or task['candidate']['seed']!=spec.seed
            or signature(original_core(raw))!=signature(spec.config['source_task'])
            or raw['binding']['config_sha256']!=signature(task)
            or spec.config['declared_task_sha256']!=raw['binding']['config_sha256']):
        raise ValueError('CP06 original test core differs')
    return raw


def validate_assets(spec, assets):
    from mind2act_eval.sim.source_assets import read_source_bytes, source_layer, verify_source_file
    row=validate(spec)
    if 'test_source' in spec.config:
        from mind2act_eval.original_test_inputs import geometry
        raw=original_case(spec,assets)
        proof=geometry(spec,assets,original_scene_sha256=raw['config']['scene_sha256'],available=False,
            method='authored_original_parameters',roots=['/World/Task'])
        if (proof['compiled_scene']['sha256']!=spec.config['prepared_scene_sha256']
                or proof['actual_composed_task']['sha256']!=spec.config['derived_task_layer_sha256']
                or proof['original_scene']['sha256']!=spec.config['original_task_layer_sha256']):
            raise ValueError('CP06 original versus reauthored geometry binding differs')
        from .original_geometry import validate_geometry
        validate_geometry(spec, assets, raw, proof)
        return dict(station_robots=raw['config']['backend']['station_robots'])
    original=parse_original(read_source_bytes(assets,'original_spec',row['source_spec_sha256']),spec.difficulty)
    for name,key in (('station','prepared_scene_sha256'),('task_source','derived_task_layer_sha256')):
        verify_source_file(assets,name,row[key],error='CP06 exact source-derived resource differs: '+name)
    from .projection import validate_manifest,load_provenance
    layer=source_layer(assets,'station',row['prepared_scene_sha256'])
    if layer is None:raise ValueError('CP06 prepared source station layer missing')
    provenance=load_provenance(read_source_bytes(assets,'projection_proof',row['resource_projection_proof_sha256']),row)
    validate_manifest(json.loads(read_source_bytes(assets,'station_mapping')),row,original,layer,provenance)
    return row


def public_parameters(core):
    """Public instance parameters; excludes fruit masses/answers/private truth."""
    result=dict(green_angle_interval_deg=list(core['green_angle_interval_deg']),
                weight_inventory_g=list(core['candidate']['weight_inventory_g']))
    if core['difficulty']=='hard':result['rider_target_m']=.09
    return result
