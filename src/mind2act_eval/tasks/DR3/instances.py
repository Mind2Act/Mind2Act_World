"""Explicit source-condition binding for DR3 (native alias FR3).

Training and reserved evaluation conditions take the same validation path. A
caller must supply the selected condition record; no first seed or tier fallback.
Native absolute paths are consumed only during conversion, never copied into the
runtime InstanceSpec. Scene bytes resolve through the shared asset library.
"""
import hashlib
import json
import math
from importlib import resources
from pathlib import PurePosixPath

from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json

SOURCE_ID = 'fr3-replay-new15-20261031'
SOURCE_REVISION = 'b6432b408607981059acf36ee668b731e36d70a0'
CONFIG_VERSION = 'dr3-initial-phase-source-v1'
TASK_INSTRUCTIONS_SHA256 = 'b6f2ae1a40576350bb32077d1f4f53c1406b5c8abf6a134fdb04eeeca6dff5ae'
INSTRUCTIONS_SHA256 = '9f0ca3f4890dc12b416d1ed738c0fa0adc30a90130ff6c0753dec2fb2167d4ae'
ROBOT_REST = tuple(math.radians(v) for v in (0, 65, 100, -65, 0, 0)) + (.044, .044)
CONDITION_FIELDS = ('instance_id', 'difficulty', 'source_revision', 'source_case_sha256',
                    'native_config_sha256', 'semantic_signature', 'scene_sha256', 'phase_ticks')


def signature(value):
    return hashlib.sha256(json.dumps(to_json(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def instructions(difficulty):
    catalog = json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())
    if (catalog['source_sha256'] != INSTRUCTIONS_SHA256 or catalog['task_id'] != 'DR3'
            or signature(catalog['instructions']) != TASK_INSTRUCTIONS_SHA256):
        raise ValueError('DR3 delivered instruction source differs')
    rows = [row for row in catalog['instructions'] if difficulty in row['d']]
    if len(rows) != 1:
        raise ValueError('DR3 instruction difficulty is not uniquely bound')
    return rows[0]


def _asset(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('DR3 assets must use relative asset-library paths')
    return value


def validate(spec):
    if spec.task_id != 'DR3' or spec.source_id != SOURCE_ID or spec.config_version != CONFIG_VERSION:
        raise ValueError('not a bound DR3 instance')
    if spec.clock.physics_hz != 120 or spec.clock.control_hz != 30:
        raise ValueError('DR3 source requires 120 Hz physics and 30 Hz control')
    c = spec.config
    source, condition = c['source_case'], c['condition']
    if source['difficulty'] != spec.difficulty or source['seed'] != 0:
        raise ValueError('DR3 source difficulty/seed changed')
    if (condition['instance_id'] != spec.instance_id or condition['difficulty'] != spec.difficulty
            or condition['source_revision'] != SOURCE_REVISION
            or condition['source_case_sha256'] != signature(source)
            or condition['scene_sha256'] != c['scene_sha256']
            or condition['phase_ticks'] != c['phase_ticks']):
        raise ValueError('DR3 source condition does not match the selected instance')
    if type(c['phase_ticks']) is not int or c['phase_ticks'] < 0:
        raise ValueError('DR3 initial mechanism phase must be explicit nonnegative ticks')
    if c['instruction'] not in (instructions(spec.difficulty)['en'], instructions(spec.difficulty)['zh']):
        raise ValueError('DR3 instruction is not the delivered text for this difficulty')
    if c['instruction_source_sha256'] != INSTRUCTIONS_SHA256:
        raise ValueError('DR3 instruction provenance changed')
    if type(c['physics_budget']) is not int or c['physics_budget'] <= 0:
        raise ValueError('DR3 physics budget must be positive ticks')
    for value in spec.assets.values():
        _asset(value)


def spec_from_row(row, *, condition, scene_asset, x5_asset, asset_version,
                  language='en', protocol_version='v1'):
    """Convert one explicitly selected original source/phase condition.

    `condition` is supplied by the selected source release, not synthesized by a
    task-name lookup. It binds full native config, source case and scene digests.
    """
    if row['task_id'] not in ('FR3', 'DR3') or row['split'] not in ('train', 'eval') or row['variant'] != 'clean':
        raise ValueError('DR3 requires an explicit clean training/evaluation condition')
    c, level, binding = row['config'], row['difficulty'], row['binding']
    if language not in ('en', 'zh'):
        raise ValueError('DR3 instruction language must be en or zh')
    if (condition['instance_id'] != row['case_id'] or condition['difficulty'] != level
            or condition['source_revision'] != binding['source_revision']
            or condition['native_config_sha256'] != signature(c)
            or condition['semantic_signature'] != binding['config_sha256']):
        raise ValueError('native DR3 spec is not the selected source condition')
    if binding['source_revision'] != SOURCE_REVISION:
        raise ValueError('DR3 source revision differs')
    if c['scene_sha256'] != condition['scene_sha256']:
        raise ValueError('DR3 prepared scene differs from the selected source')
    catalog = instructions(level)
    config = dict(source_case=to_json(c['source_case']),
                  condition={key: condition[key] for key in CONDITION_FIELDS},
                  phase_ticks=c['phase_ticks'], scene_sha256=c['scene_sha256'],
                  instruction=catalog[language], instruction_source_sha256=INSTRUCTIONS_SHA256,
                  physics_budget=c['physics_budget'], robot_rest=list(ROBOT_REST),
                  cameras={
                      'head': {'path': c['source_case']['cams']['head'], 'resolution': [640, 480]},
                      'left_wrist': {'path': '/World/RobotLeft/Asset/camera/PhysCoWristCamera', 'resolution': [640, 480]},
                      'right_wrist': {'path': '/World/RobotRight/Asset/camera/PhysCoWristCamera', 'resolution': [640, 480]},
                  })
    spec = InstanceSpec('DR3', row['case_id'], level, SOURCE_ID, CONFIG_VERSION, asset_version,
                        protocol_version, seed=c['source_case']['seed'], clock=ClockSpec(120, 30, 30),
                        config=config, assets={'station': _asset(scene_asset), 'x5': _asset(x5_asset)})
    validate(spec)
    return spec
