"""Private, source-bound CP10 original inputs and derived layout validation.

This reader preserves the original case, including its budgets and provenance.
The derived scene uses the retained static pool; it is never represented as an
original author scene. No controller or native simulator is started here.
"""
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from mind2act_eval import original_test_inputs as original
from mind2act_eval.contracts import to_json
from mind2act_eval.verified_files import verified_bytes


PROOF_SCHEMA = 'mind2act.original_test_layout_projection.v1'
AUDIT_SCHEMA = 'mind2act.cp10_original_input_audit.v1'
# Exact retained, relocated static pools inspected with the three original
# probes. These are derived station bytes, not original-author byte identities.
STATIC_POOL_SHA256 = MappingProxyType({
    'easy': '568e624a045697e5547c988e027f25caa875a61f0fffb718bd971ef980e0db5c',
    'medium': '6f4645bd715aca1431431a35c7a6332975cd6a9d6b615ce324b430cf9600ee40',
    'hard': '9c03ceeabb4fcfe790bc8b970bdd7d863490566c83e277856bd380fddbf14d09',
})
# Full retained pool, USD/texture dependencies and renderer search-root files.
# These code-owned graph pins are independent of caller closure declarations.
STATIC_POOL_GRAPH_SHA256 = MappingProxyType({
    'easy': 'e06ecd2a72891d533c5e1a80c1271f79e213e712aa88ed6a2f6dfd7393313192',
    'medium': '257e404f4b8dd83ff606e89b63aead5fb7b1f672a50bac6df7c512d01fc79afd',
    'hard': '785954b0254382b948922768e5a95440291cd3fde649b837bffeea98156b4870',
})


def _encode(value):
    return json.dumps(to_json(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _require(value, expected, description):
    if _encode(value) != _encode(expected):
        raise ValueError('CP10 original ' + description + ' differs')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('CP10 original geometry requires finite numbers')
    return value


def _static_geometry(config):
    """Check the pool contract in code, independently of producer booleans."""
    difficulty = config['difficulty']
    if difficulty not in STATIC_POOL_SHA256:
        raise ValueError('CP10 original difficulty is unsupported')
    shots, whites, seconds, speed = {
        'easy': (5, 3, 4, .9), 'medium': (8, 4, 3, 1.1),
        'hard': (12, 6, 2, 1.3),
    }[difficulty]
    arena = config['arena']
    for key, expected in dict(
        court_center=[0., .15], court_size=[.65, .88],
        surface_z=.8130000000000001, wall_height_m=.06,
        ball_diameter_m=.036, paddle_width_m=.052, paddle_axis='X',
        paddle_travel_m=[-.295, .295], net_depth_m=.12,
        collector_length_m=dict(near=.075, far=.135),
        collector_profile=dict(near=dict(center_half_width_m=.17,
            wing_inner_floor_z=.779, wing_outer_floor_z=.785)),
        practice_shots=2, scored_shots=shots, net_settle_ticks=60,
        per_shot_timeout_ticks=2400, launch_speed_m_s=speed,
        emission='preauthored_disabled_balls_enabled_once_no_pose_reset',
        contact_law='PhysX_frictionless_restitution_one',
    ).items():
        _require(arena[key], expected, 'static arena ' + key)
    launcher = arena['launcher']
    for key, expected in dict(base_xy=[0., .755], muzzle_distance_m=.36,
            yaw_limits_deg=[-58., 58.], slide_limits_m=[-.2, .2],
            vertical_drop_m=0., guide_length_m=.38, guide_half_width_m=.056,
            stowed_pitch_deg=-90.).items():
        _require(launcher[key], expected, 'static launcher ' + key)
    _require(arena['countdown']['seconds'], seconds, 'countdown')
    _require(config['physics_hz'], 120, 'physics clock')
    colors = arena['shot_colors']
    if (not isinstance(colors, list) or len(colors) != shots + 2
            or colors[:2] != ['white', 'red']
            or colors[2:].count('white') != whites
            or colors[2:].count('red') != shots - whites):
        raise ValueError('CP10 original shot inventory differs')
    objects = config['objects']
    names = {'paddle'} | {'ball' + str(i) for i in range(shots + 2)}
    if set(objects) != names:
        raise ValueError('CP10 original object inventory differs')
    paddle = objects['paddle']
    for key, expected in dict(position=[0., -.222, .8370000000000001],
            size=[.052, .014, .048], mass=.12,
            asset='anchored_slide_paddle').items():
        _require(paddle[key], expected, 'paddle ' + key)
    for i, color in enumerate(colors):
        ball = objects['ball' + str(i)]
        for key, expected in dict(size=[.036] * 3, mass=.018,
                asset='authored_sphere', launch_index=i, initially_disabled=True,
                color=[.95] * 3 if color == 'white' else [.9, .025, .025]).items():
            _require(ball[key], expected, 'ball ' + key)
        yaw = _number(ball['launcher_yaw_deg'])
        slide = _number(ball['launcher_slide_x_m'])
        if not -58 <= yaw <= 58 or not -.2 <= slide <= .2:
            raise ValueError('CP10 original launcher pose is outside the static pool')
        direction = [math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))]
        expected_position = [slide + .36 * direction[0], .755 + .36 * direction[1],
                             .8310000000000001]
        for actual, expected in zip(ball['position'], expected_position, strict=True):
            if not math.isclose(_number(actual), expected, abs_tol=2e-12, rel_tol=0):
                raise ValueError('CP10 original ball launch geometry differs')
        for actual, expected in zip(ball['launch_velocity'],
                [speed * direction[0], speed * direction[1], 0.], strict=True):
            if not math.isclose(_number(actual), expected, abs_tol=2e-12, rel_tol=0):
                raise ValueError('CP10 original ball launch velocity differs')
    first = objects['ball0']
    _require(arena['initial_slide_x_m'], first['launcher_slide_x_m'], 'initial slide')
    _require(arena['initial_yaw_deg'], first['launcher_yaw_deg'], 'initial yaw')
    _require(arena['launch_origin_xy'], first['position'][:2], 'launch origin')


def _original_config(spec, assets):
    config = original.read(spec, assets)
    if (spec.task_id != 'CP10' or config.get('task') != 'CP10'
            or config.get('difficulty') != spec.difficulty
            or type(config.get('seed')) is not int or config['seed'] != spec.seed):
        raise ValueError('CP10 original task/difficulty/seed binding differs')
    _require(spec.config['source_config'], config, 'source configuration')
    _require(spec.config['source_config_sha256'],
             original.source_identity(spec)['source_configuration_sha256'],
             'source configuration hash')
    _static_geometry(config)
    return config


def _relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError('CP10 original proof requires a relative asset path')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('CP10 original proof asset path escapes its root')
    return path


def _layer_opinions(layer):
    """Include every authored field, including extra prims and properties."""
    opinions = {}
    def visit(path):
        item = layer.GetObjectAtPath(path)
        opinions[str(path)] = {key: (
            [(offset.offset, offset.scale) for offset in layer.subLayerOffsets]
            if key == 'subLayerOffsets' else item.GetInfo(key)
        ) for key in item.ListInfoKeys()}
    layer.Traverse('/', visit)
    return opinions


def _validate_projection(scene_bytes, scene_path, pool_path, config):
    # Sdf only parses CPU-side data; it does not load a stage, assets or Kit.
    # Anonymous layers avoid the global file-layer cache and always parse the
    # authenticated bytes from this call.
    from pxr import Gf, Sdf, Tf
    layer = Sdf.Layer.CreateAnonymous('cp10-original.usda')
    try:
        parsed = layer.ImportFromString(scene_bytes.decode('utf-8'))
    except (Tf.ErrorException, UnicodeError) as error:
        raise ValueError('CP10 original derived scene cannot be parsed') from error
    if not parsed:
        raise ValueError('CP10 original derived scene cannot be parsed')
    if len(layer.subLayerPaths) != 1:
        raise ValueError('CP10 original derived scene requires one static pool')
    actual_pool = (scene_path.parent / layer.subLayerPaths[0]).resolve(strict=True)
    if actual_pool != pool_path:
        raise ValueError('CP10 original derived scene static pool path differs')
    if any(offset.offset != 0 or offset.scale != 1 for offset in layer.subLayerOffsets):
        raise ValueError('CP10 original derived scene retimes its static pool')
    expected = Sdf.Layer.CreateAnonymous('cp10-expected.usda')
    expected.defaultPrim = 'World'
    expected.pseudoRoot.SetInfo('metersPerUnit', 1.)
    expected.timeCodesPerSecond = 120
    expected.pseudoRoot.SetInfo('upAxis', 'Z')
    expected.subLayerPaths = list(layer.subLayerPaths)
    def attribute(path, name, kind, value):
        prim = Sdf.CreatePrimInLayer(expected, path)
        Sdf.AttributeSpec(prim, name, kind).default = value
    for name, obj in config['objects'].items():
        attribute('/World/Task/Objects/' + name, 'xformOp:translate',
                  Sdf.ValueTypeNames.Double3, Gf.Vec3d(*obj['position']))
        if name.startswith('ball'):
            attribute('/World/Materials/Four_' + name + '/Shader',
                      'inputs:diffuseColor', Sdf.ValueTypeNames.Color3f,
                      Gf.Vec3f(*obj['color']))
    for joint, axis, field in [('Slide', 'linear', 'initial_slide_x_m'),
                               ('Yaw', 'angular', 'initial_yaw_deg')]:
        attribute('/World/Task/Design/Emitter/' + joint,
                  'drive:' + axis + ':physics:targetPosition',
                  Sdf.ValueTypeNames.Float, config['arena'][field])
    if _layer_opinions(layer) != _layer_opinions(expected):
        raise ValueError('CP10 original derived scene has different geometry or extra opinions')


def validate_binding(spec, assets):
    """Return the exact private source case after checking its derived layout.

    Shared asset closure and native scene/scorer integration remain the caller's
    responsibility. Producer geometry flags do not authorize any geometry.
    """
    config = _original_config(spec, assets)
    root = Path(assets.root).resolve(strict=True)
    scene_path = Path(assets.resolve('scene')).resolve(strict=True)
    proof_path = Path(assets.resolve('source_geometry_proof')).resolve(strict=True)
    if not scene_path.is_relative_to(root) or not proof_path.is_relative_to(root):
        raise ValueError('CP10 original scene/proof escapes the asset root')
    if ('station' in spec.assets
            and Path(assets.resolve('station')).resolve(strict=True) != scene_path):
        raise ValueError('CP10 original station and validated scene differ')
    scene_bytes = verified_bytes(scene_path, spec.config['scene_sha256'])
    proof = json.loads(verified_bytes(proof_path,
        spec.config['source_geometry_proof_sha256'], max_bytes=1024**2))
    if (not isinstance(proof, dict) or proof.get('schema') != PROOF_SCHEMA
            or proof.get('task_id') != 'CP10'
            or proof.get('logical_slot') != spec.instance_id
            or proof.get('source_configuration_sha256') !=
                original.source_identity(spec)['source_configuration_sha256']
            or proof.get('original_scene_byte_identity_claimed') is not False
            or proof.get('original_author_bytes_available') is not False):
        raise ValueError('CP10 original geometry proof source identity differs')
    compiled = proof['compiled_scene']
    if (root / _relative(compiled['path'])).resolve(strict=True) != scene_path:
        raise ValueError('CP10 original geometry proof compiled scene path differs')
    _require(compiled['sha256'], spec.config['scene_sha256'], 'proof scene hash')
    pool = proof['common_station']
    expected_pool_sha = STATIC_POOL_SHA256[spec.difficulty]
    _require(pool['sha256'], expected_pool_sha, 'static pool hash')
    pool_path = (root / _relative(pool['path'])).resolve(strict=True)
    if not pool_path.is_relative_to(root):
        raise ValueError('CP10 original static pool escapes the asset root')
    verified_bytes(pool_path, expected_pool_sha)
    from mind2act_eval.original_asset_graph import verify_graph
    verify_graph(pool_path, root, STATIC_POOL_GRAPH_SHA256[spec.difficulty])
    _require(proof['details']['initial_slide_x_m'],
             config['arena']['initial_slide_x_m'], 'proof initial slide')
    _require(proof['details']['initial_yaw_deg'],
             config['arena']['initial_yaw_deg'], 'proof initial yaw')
    _validate_projection(scene_bytes, scene_path, pool_path, config)
    return config


def audit_source(spec, assets):
    """Keep this declaration in private evidence, never model observations."""
    validate_binding(spec, assets)
    declaration = to_json(original.audit_source(spec, assets))
    declaration.update(schema=AUDIT_SCHEMA, asset_root=str(Path(assets.root).resolve(strict=True)),
        scene=str(Path(assets.resolve('scene')).resolve(strict=True)),
        source_geometry_proof=str(Path(assets.resolve('source_geometry_proof')).resolve(strict=True)),
        test_source_request_sha256=original.REQUEST_SHA256,
        test_source_raw_sha256=original.source_identity(spec)['source_configuration_sha256'],
        scene_sha256=spec.config['scene_sha256'],
        source_geometry_proof_sha256=spec.config['source_geometry_proof_sha256'])
    return declaration


def read_audit(spec, declaration):
    """Reopen source, proof, pool and scene independently for private scoring."""
    if (not isinstance(declaration, Mapping) or declaration.get('schema') != AUDIT_SCHEMA
            or to_json(original.source_identity(spec)) != declaration.get('test_source')):
        raise ValueError('CP10 original audit source identity differs')
    for key, expected in dict(test_source_request_sha256=original.REQUEST_SHA256,
            test_source_raw_sha256=original.source_identity(spec)['source_configuration_sha256'],
            scene_sha256=spec.config['scene_sha256'],
            source_geometry_proof_sha256=spec.config['source_geometry_proof_sha256']).items():
        _require(declaration.get(key), expected, 'audit ' + key)
    root = Path(declaration['asset_root']).resolve(strict=True)
    roles = ('test_source_request', 'test_source_raw', 'scene', 'source_geometry_proof')
    paths = {}
    for role in roles:
        path = Path(declaration[role]).resolve(strict=True)
        expected = (root / _relative(spec.assets[role])).resolve(strict=True)
        if path != expected or not path.is_relative_to(root):
            raise ValueError('CP10 original audit asset path differs')
        paths[role] = path
    if 'station' in spec.assets:
        paths['station'] = (root / _relative(spec.assets['station'])).resolve(strict=True)
    return validate_binding(spec, SimpleNamespace(root=root, resolve=paths.__getitem__))
