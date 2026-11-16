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
    raise ValueError('CP08 legacy prototype is unavailable in the public clean subset; use an explicitly admitted instance')


def instructions(difficulty, *, strict=False):
    data = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    if strict and (data['source_sha256'] != '9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae'
            or signature(data['instructions']) != '7a3542dc5b635313e9692a1db301801e8911e75cc91f81f02f71e3ab90228cf5'):
        raise ValueError('CP08 delivered instruction provenance differs')
    rows = [r for r in data['instructions'] if difficulty in r['d']]
    if len(rows) != 1:
        raise ValueError('CP08 instruction difficulty is ambiguous')
    return rows[0]


def validate(spec):
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    if spec.config.get('layout_split') == 'validation':
        from .validation import validate as validate_validation
        return validate_validation(spec)
    training="train_source" in spec.config
    if training:
        from mind2act_eval.cp08_training_inputs import validate as training_validate
        row=training_validate(spec)
    test='test_source' in spec.config
    if spec.task_id != 'CP08' or (not (test or training) and spec.source_id != SOURCE_ID) or spec.config_version != 'cp08-source-projection-v1':
        raise ValueError('CP08 source identity differs')
    if spec.clock.physics_hz != 120 or spec.clock.control_hz != 30 or (not (test or training) and spec.seed != 0):
        raise ValueError('CP08 selected clock/seed differs')
    if not training:
        row = selected(spec.difficulty)
    c = spec.config
    if test:
        validate_projection(to_json(c['source_projection']),spec.difficulty,original_case=to_json(c['source_case']))
    elif not training and (spec.instance_id != row['instance_id'] or signature(c['source_case']) != signature(row['source_case'])
            or c['scene_sha256'] != row['scene_sha256'] or c['physics_budget'] != row['physics_budget']
            or c['source_spec_sha256'] != row['source_spec_sha256']):
        raise ValueError('CP08 selected training instance differs')
    if c['instruction'] not in (instructions(spec.difficulty)['en'], instructions(spec.difficulty)['zh']):
        raise ValueError('CP08 requires delivered difficulty instruction')
    if training:
        from .projection import validate_training
        validate_training(to_json(c['source_projection']),spec.difficulty,to_json(c['source_case']))
    if not (test or training):validate_projection(to_json(c['source_projection']), spec.difficulty)
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
    if spec.config.get('layout_split') == 'validation':
        from .validation import validate_assets as validate_validation_assets
        return validate_validation_assets(spec,assets)
    if "train_source" in spec.config:
        validate(spec)
        from mind2act_eval.cp08_training_inputs import validate_assets as training_validate_assets
        row=training_validate_assets(spec,assets)
        from mind2act_eval.sim.source_assets import asset_bytes,source_file,asset_layer
        raw=json.loads(asset_bytes(assets, 'train_source_raw'))
        original={k:v for k,v in raw['config']['source_case'].items() if k!='expert_staging'}
        if (original!=to_json(spec.config['source_case'])
                or raw['binding']['declared_version']!=spec.source_id
                or raw['binding']['source_seed']['index']!=spec.seed
                or raw['config']['physics_budget']!=spec.config['physics_budget']):
            raise ValueError('CP08 original training source condition differs')
        source_file(assets, 'station')
        if asset_layer(assets, 'station') is None:
            raise ValueError('CP08 original training scene proof missing')
        for role in ('cue_plan', 'cue_idle'):
            source_file(assets, role)
        return row
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
    from mind2act_eval.sim.source_assets import source_file
    for role in ('cue_plan', 'cue_idle'):
        source_file(assets, role)
    from .original_geometry import validate_geometry
    validate_geometry(spec, assets, raw, proof)
    return dict(scene_sha256=proof['compiled_scene']['sha256'],
                cue_texture_sha256={name:pins[0] for name,pins in textures.items()})


def spec_for_selected(difficulty, *, instance_id, scene_asset, x5_asset, plan_asset,
                      idle_asset, asset_version, language='en', protocol_version='v1'):
    raise ValueError('CP08 legacy prototype is unavailable in the public clean subset; use an explicitly admitted instance')
