"""Private-source binding and independent audit substitution regressions."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind2act_eval import original_test_inputs as original
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.CP10 import original_inputs as inputs


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def document(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode()
    path.write_bytes(raw)
    return sha(raw)


def source_config(difficulty):
    shots, white, seconds, speed = {'easy': (5, 3, 4, .9),
        'medium': (8, 4, 3, 1.1), 'hard': (12, 6, 2, 1.3)}[difficulty]
    colors = ['white', 'red'] + ['white'] * white + ['red'] * (shots - white)
    objects = {'paddle': dict(position=[0., -.222, .8370000000000001],
        size=[.052, .014, .048], mass=.12, asset='anchored_slide_paddle')}
    for index, color in enumerate(colors):
        yaw, slide = float(index - 8), -.1
        dx, dy = math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))
        objects['ball' + str(index)] = dict(size=[.036] * 3, mass=.018,
            asset='authored_sphere', launch_index=index, initially_disabled=True,
            color=[.95] * 3 if color == 'white' else [.9, .025, .025],
            position=[round(slide + .36 * dx, 12), round(.755 + .36 * dy, 12),
                      .8310000000000001],
            launcher_yaw_deg=yaw, launcher_slide_x_m=slide,
            launch_velocity=[round(speed * dx, 12), round(speed * dy, 12), 0.])
    first = objects['ball0']
    arena = dict(court_center=[0., .15], court_size=[.65, .88],
        surface_z=.8130000000000001, wall_height_m=.06,
        ball_diameter_m=.036, paddle_width_m=.052, paddle_axis='X',
        paddle_travel_m=[-.295, .295], net_depth_m=.12,
        collector_length_m=dict(near=.075, far=.135),
        collector_profile=dict(near=dict(center_half_width_m=.17,
            wing_inner_floor_z=.779, wing_outer_floor_z=.785)),
        practice_shots=2, scored_shots=shots, net_settle_ticks=60,
        per_shot_timeout_ticks=2400, launch_speed_m_s=speed,
        emission='preauthored_disabled_balls_enabled_once_no_pose_reset',
        contact_law='PhysX_frictionless_restitution_one', shot_colors=colors,
        launcher=dict(base_xy=[0., .755], muzzle_distance_m=.36,
            yaw_limits_deg=[-58., 58.], slide_limits_m=[-.2, .2],
            vertical_drop_m=0., guide_length_m=.38, guide_half_width_m=.056,
            stowed_pitch_deg=-90.), countdown=dict(seconds=seconds),
        initial_slide_x_m=first['launcher_slide_x_m'],
        initial_yaw_deg=first['launcher_yaw_deg'], launch_origin_xy=first['position'][:2])
    return dict(task='CP10', difficulty=difficulty, seed=5001, physics_hz=120,
        max_ticks=180000, objects=objects, arena=arena,
        original_provenance=dict(parent='held-out-input', unknown_field=['preserve', 7]))


def projection(config):
    # Hand-authored USD text is independent of the validator's Sdf construction.
    objects, materials = [], []
    for name, obj in config['objects'].items():
        position = ', '.join(map(str, obj['position']))
        objects.append(f'over "{name}"\n{{\n double3 xformOp:translate = ({position})\n}}\n')
        if name.startswith('ball'):
            color = ', '.join(map(str, obj['color']))
            materials.append(f'over "Four_{name}"\n{{\n over "Shader"\n{{\n'
                             f'color3f inputs:diffuseColor = ({color})\n}}\n}}\n')
    slide, yaw = config['arena']['initial_slide_x_m'], config['arena']['initial_yaw_deg']
    return ('#usda 1.0\n(\ndefaultPrim="World"\nmetersPerUnit=1\ntimeCodesPerSecond=120\n'
        'upAxis="Z"\nsubLayers=[@pool.usda@]\n)\n'
        'over "World"\n{\nover "Task"\n{\nover "Objects"\n{\n' + ''.join(objects) +
        '}\nover "Design"\n{\nover "Emitter"\n{\n'
        f'over "Slide"\n{{\nfloat drive:linear:physics:targetPosition = {slide}\n}}\n'
        f'over "Yaw"\n{{\nfloat drive:angular:physics:targetPosition = {yaw}\n}}\n'
        '}\n}\n}\nover "Materials"\n{\n' + ''.join(materials) + '}\n}\n').encode()


@pytest.fixture(params=['easy', 'medium', 'hard'])
def bound(tmp_path, monkeypatch, request):
    difficulty = request.param
    config = source_config(difficulty)
    raw_sha = document(tmp_path / 'raw.json', config)
    case = 'CP10_' + difficulty + '_clean_000'
    row = dict(task_id='CP10', difficulty=difficulty, seed=5001, logical_slot=case,
        native_case_id='original-native-' + difficulty, source_configuration_sha256=raw_sha,
        source_index=dict(sha256='a' * 64))
    request_sha = document(tmp_path / 'request.json', dict(records=[row] +
        [dict(logical_slot='other-' + str(i)) for i in range(2249)]))
    monkeypatch.setattr(original, 'REQUEST_SHA256', request_sha)
    pool = b'#usda 1.0\n(defaultPrim="World")\ndef Xform "World" {}\n'
    (tmp_path / 'pool.usda').write_bytes(pool)
    monkeypatch.setattr(inputs, 'STATIC_POOL_SHA256', dict.fromkeys(
        ['easy', 'medium', 'hard'], sha(pool)))
    # This unit seam already substitutes code-owned pool/source authority.
    # Give its minimal pool the corresponding closed graph; real retained-pool
    # child mutation tests never replace production graph pins.
    for folder in ('sdk/mdl/core/Base', 'sdk/mdl/core/mdl', 'sdk/mdl/rtx'):
        (tmp_path/folder).mkdir(parents=True)
    node_sha = sha(json.dumps([sha(pool), []], separators=(',', ':')).encode())
    graph_sha = sha(json.dumps([node_sha, []], separators=(',', ':')).encode())
    monkeypatch.setattr(inputs, 'STATIC_POOL_GRAPH_SHA256', dict.fromkeys(
        ['easy', 'medium', 'hard'], graph_sha))
    scene = projection(config)
    (tmp_path / 'scene.usda').write_bytes(scene)
    proof = dict(schema=inputs.PROOF_SCHEMA, task_id='CP10', logical_slot=case,
        source_configuration_sha256=raw_sha, static_source_geometry_parameters_equal=True,
        original_author_bytes_available=False, original_scene_byte_identity_claimed=False,
        details=dict(initial_slide_x_m=config['arena']['initial_slide_x_m'],
                     initial_yaw_deg=config['arena']['initial_yaw_deg']),
        common_station=dict(path='pool.usda', sha256=sha(pool)),
        compiled_scene=dict(path='scene.usda', sha256=sha(scene)))
    proof_sha = document(tmp_path / 'proof.json', proof)
    source = dict(schema=original.SCHEMA, instance_set='test50-r5', split='test',
        logical_slot=case, native_case_id=row['native_case_id'], source_document_id=None,
        source_seed=5001, source_configuration_sha256=raw_sha, source_index_sha256='a' * 64,
        source_request_sha256=request_sha, snapshot_sha256=original.SNAPSHOT_SHA256)
    entries = dict(scene='scene.usda', station='scene.usda', source_geometry_proof='proof.json',
                   test_source_request='request.json', test_source_raw='raw.json')
    spec = SimpleNamespace(task_id='CP10', instance_id=case, difficulty=difficulty, seed=5001,
        assets=entries, config=dict(source_config=deepcopy(config), test_source=source,
            source_config_sha256=raw_sha, scene_sha256=sha(scene),
            source_geometry_proof_sha256=proof_sha))
    return spec, LocalAssets(tmp_path, entries), config


def repin_scene(spec, assets, raw):
    assets.resolve('scene').write_bytes(raw)
    spec.config['scene_sha256'] = sha(raw)
    proof_path = assets.resolve('source_geometry_proof')
    proof = json.loads(proof_path.read_text())
    proof['compiled_scene']['sha256'] = sha(raw)
    spec.config['source_geometry_proof_sha256'] = document(proof_path, proof)


def test_exact_private_case_and_independent_audit_preserve_all_source_fields(bound):
    spec, assets, config = bound
    assert inputs.validate_binding(spec, assets) == config
    declaration = inputs.audit_source(spec, assets)
    assert inputs.read_audit(spec, declaration) == config
    assert declaration['test_source']['native_case_id'] != spec.instance_id
    assert declaration['test_source']['source_document_id'] is None
    changed = inputs.read_audit(spec, declaration)
    changed['max_ticks'] = 1
    assert inputs.read_audit(spec, declaration)['max_ticks'] == 180000


@pytest.mark.parametrize('part', ['request', 'raw', 'scene', 'proof', 'pool'])
def test_independent_audit_reopens_and_rejects_substituted_bytes(bound, part):
    spec, assets, _ = bound
    declaration = inputs.audit_source(spec, assets)
    role = dict(request='test_source_request', raw='test_source_raw', scene='scene',
                proof='source_geometry_proof').get(part)
    path = assets.resolve(role) if role else assets.root / 'pool.usda'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        inputs.read_audit(spec, declaration)


@pytest.mark.parametrize('field,bad', [('native_case_id', 'another-native'),
    ('logical_slot', 'another-case'), ('source_document_id', 'invented-document'),
    ('source_seed', 5002), ('source_index_sha256', 'b' * 64), ('split', 'train')])
def test_distinct_original_identities_cannot_be_substituted(bound, field, bad):
    spec, assets, _ = bound
    spec.config['test_source'][field] = bad
    with pytest.raises(ValueError):
        inputs.validate_binding(spec, assets)


@pytest.mark.parametrize('change', ['budget', 'provenance', 'object', 'seed_zero', 'raw_repin'])
def test_complete_source_configuration_is_request_bound(bound, change):
    spec, assets, _ = bound
    config = spec.config['source_config']
    if change == 'budget':
        config['max_ticks'] = 1
    elif change == 'provenance':
        del config['original_provenance']
    elif change == 'object':
        config['objects']['ball0']['position'][0] += .01
    elif change == 'seed_zero':
        config['seed'] = spec.seed = spec.config['test_source']['source_seed'] = 0
    else:
        config['max_ticks'] = 1
        altered_sha = document(assets.resolve('test_source_raw'), config)
        spec.config['source_config_sha256'] = altered_sha
        spec.config['test_source']['source_configuration_sha256'] = altered_sha
    with pytest.raises(ValueError):
        inputs.validate_binding(spec, assets)


@pytest.mark.parametrize('change', ['position', 'color', 'yaw', 'extra_physics',
                                  'extra_prim', 'retime', 'new_pool'])
def test_repinning_scene_and_proof_cannot_authorize_changed_geometry(bound, change):
    spec, assets, _ = bound
    raw = assets.resolve('scene').read_bytes()
    if change == 'position':
        raw = raw.replace(b'0.8310000000000001', b'0.8410000000000001', 1)
    elif change == 'color':
        raw = raw.replace(b'(0.95, 0.95, 0.95)', b'(0.1, 0.1, 0.1)', 1)
    elif change == 'yaw':
        raw = raw.replace(b'targetPosition = -8.0', b'targetPosition = -7.0', 1)
    elif change == 'extra_physics':
        raw = raw.replace(b'over "paddle"\n{\n',
            b'over "paddle"\n{\nbool physics:rigidBodyEnabled = false\n', 1)
    elif change == 'extra_prim':
        raw += b'def Xform "Unexpected" {}\n'
    elif change == 'retime':
        raw = raw.replace(b'@pool.usda@', b'@pool.usda@ (offset = 1)')
    else:
        (assets.root / 'other.usda').write_bytes((assets.root / 'pool.usda').read_bytes())
        raw = raw.replace(b'@pool.usda@', b'@other.usda@')
    assert raw != assets.resolve('scene').read_bytes()
    repin_scene(spec, assets, raw)
    with pytest.raises(ValueError):
        inputs.validate_binding(spec, assets)


@pytest.mark.parametrize('change', ['source', 'case', 'scene_hash', 'scene_path',
                                  'pool_hash', 'escape', 'author_claim', 'initial_pose'])
def test_repinning_proof_cannot_substitute_its_bound_facts(bound, change):
    spec, assets, _ = bound
    path = assets.resolve('source_geometry_proof')
    proof = json.loads(path.read_text())
    if change == 'source':
        proof['source_configuration_sha256'] = 'b' * 64
    elif change == 'case':
        proof['logical_slot'] = 'other-case'
    elif change == 'scene_hash':
        proof['compiled_scene']['sha256'] = 'b' * 64
    elif change == 'scene_path':
        proof['compiled_scene']['path'] = 'pool.usda'
    elif change == 'pool_hash':
        proof['common_station']['sha256'] = 'b' * 64
    elif change == 'escape':
        proof['common_station']['path'] = '../pool.usda'
    elif change == 'author_claim':
        proof['original_scene_byte_identity_claimed'] = True
    else:
        proof['details']['initial_yaw_deg'] += 1
    spec.config['source_geometry_proof_sha256'] = document(path, proof)
    with pytest.raises(ValueError):
        inputs.validate_binding(spec, assets)


@pytest.mark.parametrize('field', ['test_source', 'scene', 'source_geometry_proof', 'asset_root'])
def test_audit_declaration_is_checked_against_spec_and_current_assets(bound, field):
    spec, assets, _ = bound
    declaration = inputs.audit_source(spec, assets)
    if field == 'test_source':
        declaration[field]['native_case_id'] = 'other-native'
    else:
        declaration[field] = str(assets.root / 'raw.json')
    with pytest.raises((ValueError, NotADirectoryError)):
        inputs.read_audit(spec, declaration)


@pytest.mark.parametrize('field', ['test_source_request_sha256', 'test_source_raw_sha256',
                                  'scene_sha256', 'source_geometry_proof_sha256'])
def test_audit_hashes_are_rechecked_against_the_authenticated_spec(bound, field):
    spec, assets, _ = bound
    declaration = inputs.audit_source(spec, assets)
    declaration[field] = 'b' * 64
    with pytest.raises(ValueError):
        inputs.read_audit(spec, declaration)


def test_loaded_station_must_be_the_validated_derived_scene(bound):
    spec, assets, _ = bound
    spec.assets['station'] = 'pool.usda'
    assets.entries['station'] = 'pool.usda'
    with pytest.raises(ValueError):
        inputs.validate_binding(spec, assets)


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_actual_original_layout_probe_when_explicitly_supplied(difficulty):
    path = os.environ.get('CP10_ORIGINAL_PROBE_ROOT')
    if path is None:
        pytest.skip('actual private probes must be supplied explicitly')
    root = Path(path)
    name = 'CP10_' + difficulty + '_clean_000'
    spec_path = root / 'test-cases' / 'CP10' / difficulty / name / 'spec.json'
    spec = InstanceSpec.from_dict(json.loads(spec_path.read_text()))
    assets = LocalAssets(root, spec.assets, spec=spec)
    actual = inputs.validate_binding(spec, assets)
    assert actual == json.loads(assets.resolve('test_source_raw').read_text())
    assert inputs.read_audit(spec, inputs.audit_source(spec, assets)) == actual
