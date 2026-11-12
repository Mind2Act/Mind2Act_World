"""Selected training-source and delivered-instruction whitelist for CP08."""
import hashlib
import json
from importlib import resources
from pathlib import PurePosixPath
from mind2act_eval.contracts import to_json
from .projection import validate as validate_projection

SOURCE_ID = 'fr3-replay-new15-20261031'
SOURCE_REVISION = 'b6432b408607981059acf36ee668b731e36d70a0'


def signature(value):
    return hashlib.sha256(json.dumps(to_json(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def selected(difficulty):
    return json.loads(resources.files(__package__).joinpath('config/selected.json').read_text())[difficulty]


def instructions(difficulty):
    data = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    if (data['source_sha256'] != '9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae'
            or signature(data['instructions']) != '7a3542dc5b635313e9692a1db301801e8911e75cc91f81f02f71e3ab90228cf5'):
        raise ValueError('CP08 delivered instruction provenance differs')
    rows = [r for r in data['instructions'] if difficulty in r['d']]
    if len(rows) != 1:
        raise ValueError('CP08 instruction difficulty is ambiguous')
    return rows[0]


def validate(spec):
    test='test_source' in spec.config
    if spec.task_id != 'CP08' or (not test and spec.source_id != SOURCE_ID) or spec.config_version != 'cp08-source-projection-v1':
        raise ValueError('CP08 source identity differs')
    if spec.clock.physics_hz != 120 or spec.clock.control_hz != 30 or (not test and spec.seed != 0):
        raise ValueError('CP08 selected clock/seed differs')
    row = selected(spec.difficulty)
    c = spec.config
    if test:
        from mind2act_eval.original_test_inputs import source_identity
        source=source_identity(spec)
        if c['source_spec_sha256']!=source['source_configuration_sha256']:
            raise ValueError('CP08 original test spec SHA differs')
        validate_projection(to_json(c['source_projection']),spec.difficulty,original_case=to_json(c['source_case']))
    elif (spec.instance_id != row['instance_id'] or signature(c['source_case']) != signature(row['source_case'])
            or c['scene_sha256'] != row['scene_sha256'] or c['physics_budget'] != row['physics_budget']
            or c['source_spec_sha256'] != row['source_spec_sha256']):
        raise ValueError('CP08 selected training instance differs')
    if c['instruction'] not in (instructions(spec.difficulty)['en'], instructions(spec.difficulty)['zh']):
        raise ValueError('CP08 requires delivered difficulty instruction')
    if not test:validate_projection(to_json(c['source_projection']), spec.difficulty)
    for value in spec.assets.values():
        path = PurePosixPath(value)
        if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
            raise ValueError('CP08 asset must resolve through relative library path')
    return row


def original_case(spec,assets):
    from mind2act_eval.original_test_inputs import read
    raw=read(spec,assets)
    if (raw['task_id']!='CP08' or raw['difficulty']!=spec.difficulty or raw['split']!='eval'
            or raw['binding']['declared_version']!=spec.source_id
            or raw['binding']['source_seed']['index']!=spec.seed
            or signature(raw['config']['source_case'])!=signature(spec.config['source_case'])
            or raw['config']['physics_budget']!=spec.config['physics_budget']):
        raise ValueError('CP08 original test source condition differs')
    return raw


def validate_assets(spec,assets):
    row=validate(spec)
    if 'test_source' not in spec.config:return row
    raw=original_case(spec,assets)
    from mind2act_eval.original_test_inputs import geometry
    proof=geometry(spec,assets,original_scene_sha256=raw['config']['scene_sha256'],available=True,
        method='composed_source_task',roots=['/World/CP11','/World/CP11Looks'])
    if proof['compiled_scene']['sha256']!=spec.config['scene_sha256']:
        raise ValueError('CP08 compiled scene binding differs')
    closure=raw['config']['native_inputs']['manifest']['asset_closure']
    textures={name:[r['sha256'] for r in closure if PurePosixPath(r['path']).name==name]
              for name in ('plan.png','screen-idle.png')}
    if any(len(pins)!=1 for pins in textures.values()):
        raise ValueError('CP08 original cue texture provenance ambiguous')
    from mind2act_eval.sim.source_assets import verify_source_file
    for role,name in (('cue_plan','plan.png'),('cue_idle','screen-idle.png')):
        verify_source_file(assets,role,textures[name][0],error='CP08 original test cue texture differs')
    from .original_geometry import validate_geometry
    validate_geometry(spec, assets, raw, proof)
    return dict(scene_sha256=proof['compiled_scene']['sha256'],
                cue_texture_sha256={name:pins[0] for name,pins in textures.items()})


def spec_for_selected(difficulty, *, instance_id, scene_asset, x5_asset, plan_asset,
                      idle_asset, asset_version, language='en', protocol_version='v1'):
    from mind2act_eval.contracts import InstanceSpec
    import math
    row=selected(difficulty)
    if instance_id != row['instance_id'] or language not in ('en','zh'):
        raise ValueError('CP08 requires explicit selected instance and delivered language')
    from .projection import H5_SHA256
    spec=InstanceSpec(task_id='CP08',instance_id=instance_id,difficulty=difficulty,
        source_id=SOURCE_ID,config_version='cp08-source-projection-v1',asset_version=asset_version,
        protocol_version=protocol_version,seed=0,assets=dict(station=scene_asset,x5=x5_asset,
        cue_plan=plan_asset,cue_idle=idle_asset),config=dict(
        source_case=row['source_case'],scene_sha256=row['scene_sha256'],
        source_spec_sha256=row['source_spec_sha256'],physics_budget=row['physics_budget'],
        instruction=instructions(difficulty)[language],
        robot_rest=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044],
        source_projection=dict(source_physics_hz=120,source_state_tick=72,source_action_tick=76,
            source_h5_row=18,source_cue_hide_tick=360,h5_sha256=H5_SHA256[difficulty])))
    validate(spec)
    return spec
