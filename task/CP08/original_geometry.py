"""Source-qualified physical primitives and original layout on retained station."""
from pathlib import Path

from mind2act_eval.source_geometry import opinions, fingerprint, validate_composition, finite_default

BASE_SHA256 = {
    'easy': 'd131bfe8e16d1b42180b74b515a9aac782b6d4ffb1802713750e7f6a3103ee9a',
    'medium': '6897b11f7eba793267a55ee24dff497eb4e01b6de517251c420433c5f113600e',
    'hard': '3117c7a2115f292721015c448e01636fb43ddf319ea630b279593ccf48a68e9d'}
BASE_GRAPH_SHA256 = {
    'easy': '167b1fb478618ce58378215ce09a6cf9ebedf4a046f0e7c9dfd43bd648ace411',
    'medium': '4195d6f82e68752d770397ea552d49ef7ed0f846e0d0b7635e4d84f4a716e2ba',
    'hard': 'a37edd33b19db2806493dc11cbdd5fa3a869eabf65bb5955833eda623fbcd755'}
# Primitive/fixture fingerprints, never a table of150 accepted case scenes.
PRIMITIVES = {
    ('cuboid', (.044,.034,.034)): '254a32237cd730e44632f9ae2b49a2191ab073de6a480cafeb023b46cd507c6b',
    ('cuboid', (.124,.034,.034)): 'e783753fb3abc4b469a867baffa0b07bdc0dae99b34f4c3218e5e072512837e7',
    ('cylinder', (.034,.034,.034)): 'a7b43d20f3cd5a691de28a67ea303a71a15a19f91c0dd1302c59b0f882cf469d',
    ('triangular_prism', (.124,.034,.034)): '05fdd239d45b1085dbc9a891a22a42da91261337512742cc64cf3f9f2f0bba53',
    ('cuboid', (.034,.034,.034)): 'f7872ceb69d0233fc880dcbfe2a7c61b0b69800eb90c7da6467537219f6ebcc2'}
MATERIAL = 'ef2c81207eb787c22ee77b0017406333cf2fd4fa2148fb81bb0bae6889b44ea5'
STATIC_FIXTURE = dict.fromkeys(('easy','medium','hard'),
    '86fa99eeef303367213f7699fbf2aac19552a8101145253fac8e5b77cede3ae8')


def projected_task(raw, layer, path, root, *, enforce=True):
    from pxr import Sdf, Gf, Vt
    case = raw['config']['source_case']
    data = opinions(layer)
    bodies, materials = {}, set()
    for block in [*case['blocks'], *case['distractors']]:
        prefix = '/World/CP11/Blocks/'+block['id']
        material = '/World/CP11Looks/'+block['id']
        body = opinions(layer, prefix)
        position = prefix+'.xformOp:translate'
        if finite_default(body, position, (3,)) != Gf.Vec3d(*block['initial']):
            raise ValueError('CP08 original physical initial position differs')
        body[position]['default'] = 'RAW_INITIAL'
        if block['shape'] == 'triangular_prism':
            key = prefix+'/Normalized/Asset.xformOp:transform'
            matrix = finite_default(body, key, (4, 4))
            # Original bounding-box normalization leaves <=3.5e-18m residuals.
            # Verify mathematical identity at sub-femtometre precision; retain
            # every mesh point, type, collider and other transform opinion.
            if any(abs(matrix[i][j]-(1. if i == j else 0.)) > 1e-15
                   for i in range(4) for j in range(4)):
                raise ValueError('CP08 original normalized prism transform differs')
            body[key]['default'] = Gf.Matrix4d(1.)
        key = (block['shape'], tuple(block['size']))
        value = fingerprint(body, path, root, {prefix:'/Object', material:'/Material'})
        if enforce and value != PRIMITIVES.get(key):
            raise ValueError('CP08 original primitive shape or physics differs')
        bodies[key] = value
        m = opinions(layer, material)
        color = material+'/Surface.inputs:diffuseColor'
        linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in block['color']]
        if finite_default(m, color, (3,)) != Gf.Vec3f(*linear):
            raise ValueError('CP08 original linear material color differs')
        m[color]['default'] = 'RAW_COLOR'
        value = fingerprint(m, path, root, {material:'/Material'})
        if enforce and value != MATERIAL:
            raise ValueError('CP08 original material or friction differs')
        materials.add(value)
        for name in list(data):
            if Sdf.Path(name).HasPrefix(prefix) or Sdf.Path(name).HasPrefix(material):
                del data[name]
    x, monitor_y = case['monitor_xy']
    monitor = '/World/CP11/GoalMonitor.xformOp:translate'
    if finite_default(data, monitor, (3,)) != Gf.Vec3d(x, monitor_y, .9229378304920354):
        raise ValueError('CP08 original monitor position differs')
    data[monitor]['default'] = 'RAW_MONITOR'
    y = monitor_y-.215
    z = case['table_z']+.0005
    points = [(x-.09,y-.05,z),(x+.09,y-.05,z),(x+.09,y+.05,z),
        (x-.09,y+.05,z),(x-.09,y-.05,z),(x-.09,y,z),(x+.09,y,z),
        (x,y-.008,z),(x,y+.008,z)]
    area = '/World/CP11/BuildArea.points'
    if finite_default(data, area, (9, 3)) != Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]):
        raise ValueError('CP08 original build-area geometry differs')
    data[area]['default'] = 'RAW_BUILD_AREA'
    closure = raw['config']['native_inputs']['manifest']['asset_closure']
    pins = {Path(v['path']).name:v['sha256'] for v in closure
            if Path(v['path']).name in ('plan.png','screen-idle.png')}
    markers = {pins['plan.png']:'SOURCE_PLAN',pins['screen-idle.png']:'SOURCE_IDLE'}
    static = fingerprint(data, path, root, asset_markers=markers)
    if enforce and static != STATIC_FIXTURE.get(raw['difficulty']):
        raise ValueError('CP08 original static fixture or resource closure differs')
    return bodies, materials, static


def validate_geometry(spec, assets, raw, proof):
    path, layer = validate_composition(spec, assets, proof,
        BASE_SHA256[spec.difficulty], BASE_GRAPH_SHA256[spec.difficulty],
        ['/World/CP11','/World/CP11Looks'])
    projected_task(raw, layer, path, Path(assets.root).resolve(strict=True))
