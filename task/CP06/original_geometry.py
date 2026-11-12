"""Exact original produce primitives, physical parameters and retained scale."""
from pathlib import Path

from mind2act_eval.source_geometry import (
    opinions, fingerprint, validate_composition, finite_default, finite_value)

BASE_SHA256 = {
    'easy': 'd8f5c53c714ebbb3f1dddb027a0456ee163eb1c0ba31ee1b85f61ae954321803',
    'medium': '884abd869e0c83aa44a3bc23edc77c5d824eee8183c46028900f58170d4a8edd',
    'hard': '4fef2f46a87b18a1d2fed6840abe5a3d52216135c8fc4982da199014bea1f066'}
BASE_GRAPH_SHA256 = {
    'easy': 'ea2269497c66e58386ef189a5784afd0e803fc1d17f0fd7e00d4aec1c34c1ebf',
    'medium': 'e9d287a97d93082d623cb7e1bd1c4469ff0e98596e219618309114e847f13f47',
    'hard': '9a686ecad759f721ecd2db9c071fe52229419e79a6ceabee21c0488256cf955c'}
# These42 component pins were independently reconstructed from the original
# CAD bytes identified by the authenticated source snapshot. They bind complete
# authored subtrees after checking only the four raw physical parameters below.
# No test-case scene digest or producer receipt supplies geometry authority.
PRODUCE = {'002c7cbf5a200b5cec72f33e451e73c1f9dd09eaa933abff20bfd9a105ffd3e7': 'cbfa39f1a76592d3e49a4c9476ad2b19c58f81bb3fafc98c2f00160ca2666f53',
 '08d08fdaf8951ee08f12229708b8a0586fc92970d1bc3be9a5140900b9ebb349': 'b36d00c2d0ba86b033bb3eca3357631cb9043c6d14122593d47dbf57e5938bdc',
 '13a43fbc882e1446ec91436c364863e442bbfa7276fe54dbca3e5b2966d2ae68': 'b15f79369e15e71db95708e47dcbd64d2044eb38ec4828a42bf6343b678fe3dd',
 '14eb8ed1f477cb4ae2bcbd41d602cc70de4a5d46aa37ae06b77aa354ce91393d': 'e07ff7233502663f7f933a668e707d641fa625c43a72c05430663a670c4e1793',
 '2be598f9c58c4b64e5a99e4989db0022e09bf165218b1f990e97c558ffb258c8': 'fdb54fe21853aba090e4036484fddd1efd5d5a6763a1d37b7efef4563df9e9a6',
 '2cdb5558f7bd737d3e0154c3873f4e7ed3d71d961914a74ffabd883104cb309a': 'c27e8a810c1abbdb78a3a2268c141f0c7506c1c17201e610369ce0784370e09a',
 '3ea1e341c4133a1785f524875046bf1f969781528841bc2e35109aa10d13635c': '9edcb1837eac7cbeb8d530f315fd7de03611cb71447588ab7f4b4e3ce57b777c',
 '44457c91db3bf03df0a6e8fcbaaef9f0e2f1ac228a7356d773e0ec83d1a88511': '3156757470255b61d9b9b37d2bf2a8bf063b02cdc55532bd091adaa9c7779bc0',
 '4ce4c85c8f8a574a40326c548dc559ee2a908eea46c445a6fa047a9e1c6f5782': '244a55a859b84c80f6427f80c98c67874c36ddbcad7585f34e421a4ae6b88646',
 '4d334f6f79e808e4f47e438ebd7387ae984813a7e9c23e82d2821e393b3e3dbc': '05b5bafc838a24427c599e9ac5d37b29fdbec80c26a29cdb2a5f1cfd020cb515',
 '4f408d0693abc5c4526097723c7dba17445b5ea444c6826e83023b44133dcaa3': 'f23872a016caca8762316300f0c1872326cbba797a827668f056798d9a3dc87d',
 '55e7e85052b64362ba4c03e27089c40af0c2235f78caaf933b710599ebede8ca': '7265ee383fb600d38e8a2dd8257e5d5eda886945ef032bd17038e0f967838ee3',
 '56181e555de615733be9786543a044a4ab5cefb96f00555688271f3a2971472d': '51a40169336416a145e442825bbd664ee7224626bedadcc853ee696c4da28e13',
 '606f5ee029c5a4dab037422864a868aec8e1623dc4f4f97870000aad2cf2ad1f': '37b135435597978c01c71db720ea174f6d2f042ad5584b23d08cc4950d1d8604',
 '6626f9852cb95be50078822d8fcdcc556135f35908ef8cf4b08edfbe200f930d': 'd6003a6938959011bcdfaaefb1bbb758be4e2fcaf53d299796116819f15867dc',
 '664112b74359763a778dc42ec0627f73709385bfd6340fb5d0ddbac665bb3cb3': '5c8ff1ae2c41be309420cf477a055ee83cd7387cad4c7bf08b6b8c00c71a063d',
 '699d4a060bffba568c044f97e1006f335288f54272dc911d22acfa892f8f42ec': '5548f94595d66254929b71c7c1b864c612a447d0314d9146d29a578f63bac6ec',
 '77154bec22a764f02e16aceb2271467e16995eac9153b9b2dd5b781ddaff5c01': '089db3e690c8bf38760c885e026395d1225cf95782c1e1570a51ff97a1dd97bb',
 '7bafaf0dfec3f48565ac408aa199e9e0545d710875f3057c574ee7681c7a6ce2': 'bd1b00ee87946ab64e9c6b13cebc2534d80a539d704e55d1d4ecd1677edbca11',
 '7ece02fe81e819bec808238601ea611c58e73355a7f8997b965d27f004b3275b': '85f14c750f24bff72483c084c7e5ebb5a6a57bf17d43eb00aeb03cd4e5d9af5b',
 '8b9d2fcfc57c17b183ab7c4d08b842b980140bb4224c8961e27daf04763fa0e7': '3fb05add7b29dfe7659d224a98a21bccc3ce579add5bddbab30ea81a46c6dd29',
 '8bf52ea5db0827aab52929a29422295d089b7afdfd4be7492ca15ca8d7d1e2fc': 'c0ef8f55d0484681b076d8c518f7a9560955e6693da0532530b1c7201fbc6ca6',
 '8ede7e8bfae452dfc4dff68df5ee2a9288538a8e11094b3484f0753deb97dab5': 'fe9885158e2a0095a62037d70d6974b126b5fea906e9c15827229186beee0360',
 '9262966bd64353a2e770f7947e8069b884da3188b5de33392fa3ac94c0924275': 'c3aa62fc805754e3c1bb5cebca42bd5625790474352ae9f8cfde8fed8b74a9a5',
 '9d98b5d173341e8c69562887a4f609d13a6379d51650e3f9003595de6246f820': 'c67e9c5b06d62382d7b035cdc67193a54f9bdebe97c269ab4662effab6e39bf5',
 '9ffc4e199e7c8ae2866abd7dc699723f8b719557fe53de81c5f5517f8d7428d2': '5d4188d730f0abfc3af2eb5ebd01dfc42afe3dcadd99b3c9eced7c627129c63e',
 'a811bb1f9450afaf2ff74cbb8877a342ebb95e58852f3189f3adae8013a6b2cc': 'b5b76f6b684d90557e99be7ef182d1a92082c8aa3c7965d0fc3636cedd4ceb4f',
 'b1065b9b7310a9ae49134fbdbe5a0deee9619e157bfe31fe7ca7f93c18139f1b': '60fee5d9d4b87a748d7d3be060375f4bb5706c6754e69fd3227b7e4ec956416c',
 'b8ddd326db744dd460e42dbd843084609a62b706a9968d8a77be0369634dc1eb': '1552d9d2f70304aa535f81ae80a4c0a1537e4ebcbe2e5f0d42df6d933fa869ef',
 'bcd9d7a83e5a23ebfe1c9a4959ab7b1738f610a4ea71fb74b5df55ee81dfbc6c': '3641351d01ffe2923d8d5492ee71b123e7f4068789837401e731bcda15e6158e',
 'bd6696a49636bab91e44a07de3d1ea1bae4e48335ff5511792b6654ded236f01': 'df95be9482c99b8a98b3d8e1a023e041497f16efdaa7b793eb7f95c2d600fe83',
 'c02dfec9cf08ed67c083657a8b810c3d610b54dfb5fe01f302e0692190a438da': '39bb7e764b7dc45eb30ad05d561816c0fa223f867fe4faf061626ec1fc710843',
 'd05164fbd08f38f095a2dfcd90c75db7972fa414b5227dd55dca3276635b8de7': '804d2a572d05139353917de236bee018cf9dd61778b010b820d47559775bff5d',
 'd13ec2e05f30646b5ad8484dc012e927badd68b313ceab29e2ad2b76575bd51c': '7d33cb3b25057db8acb337fed2c747f4a383ca281dfc25db6e08f01808036b58',
 'dbd862ef0cac816b29774d29209271a50af72d9c9bc91483bc06bf187ae5c1e1': '53fb915bba48600e9cf04eac6293b7e0235f5274ffe7f94c0e5473ff79aadf6c',
 'ddc692bf338122739cb2b9f441f086ab3e2992567e24690cc39f6dd6eeb63251': '5bc761e96e0708c043d295bbe58d10da6f2d3bbf91496c32b79c70cf0dc21148',
 'ddd32b4f9e5f5a6685d1df69c5424f7bf689ebf748b9875672e7e88e0efaa2ca': '66e35e3a319ed76ebe66587bc1b6dae515ed214da36d04ec2f9a5273273d1839',
 'e189f292b9d62be1cfc2e6c9c07aea6cccc623837adea6f21239dc0388fe25ec': '8ea2fb2279d54aa6131d380c3a6be7ebd7c2fbb5b472b1a82c2d90ebee2238f5',
 'e9585ec054971bc56c667d83a56cdd3c7d20429ace0d7661f4bf4f736747525d': 'b1ffb00b63fc6eca212b62fb4953211c058e30f4e327c9ec3fc91860a7a26371',
 'e96b4d44cd167c620a3c0999b618058f5c22f78ca8630e4cfc7c08f8b70608a3': '424ba95b661442393d895b34696edac42868c22a740263b2f0a8ec59454af26f',
 'f13c6b1a57c9c42129ba739bb018acd16ee3d1d3af1f2cb50114c9246ac58e5b': 'e997d9c38a3d8671b8d52a475f203c158dc1b84f5ca8864277c19d2aef6c466d',
 'f4908becc57edbbd796ad5c9fc7dd7f00a850bbda8221eaf1b65e021bd0cfc5e': '837c6b11998cf687f2bb667eb7f2d52c2cd7a2b6cb0c03f839a60e540fec6b95'}
# The three retained scale/weight fixtures have separate tier authority.
STATIC_FIXTURE = {'easy': 'fac4957a8d08955a2a764178def90d725005b95895272f4c54154abc3121c64f',
 'hard': '1ed8fe81057cf5c3b55de8df672237bcfe11a09d8e0baa51c7965d96ba53b154',
 'medium': 'be6f2c687f270a67e9db32a99617cf44cee7edaa2c009c99c8092c330ea25513'}
FIELDS = ('xformOp:transform', 'physics:mass', 'physics:centerOfMass', 'physics:diagonalInertia')


def mask_physical_parameters(data, stage, obj):
    from pxr import Usd, UsdGeom, Gf
    body = obj['path']
    for name, shape in zip(FIELDS, ((4, 4), (), (3,), (3,))):
        finite_default(data, body+'.'+name, shape)
    q = obj['initial_orientation_wxyz']
    world = Gf.Matrix4d(1.)
    world.SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    world.SetTranslateOnly(Gf.Vec3d(*obj['initial_position_m']))
    actual = UsdGeom.Xformable(stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    finite_value(world, (4, 4))
    finite_value(actual, (4, 4))
    if any(abs(world[i][j]-actual[i][j]) > 1e-12 for i in range(4) for j in range(4)):
        raise ValueError('CP06 original physical body world pose differs')
    expected = {
        'physics:mass': Gf.Vec3f(obj['grams']/1000,0,0)[0],
        'physics:centerOfMass': Gf.Vec3f(*obj['center_of_mass_m']),
        'physics:diagonalInertia': Gf.Vec3f(*obj['diagonal_inertia_kg_m2'])}
    for name, value in expected.items():
        if data[body+'.'+name]['default'] != value:
            raise ValueError('CP06 original body mass, COM or inertia differs')
    for name in FIELDS:
        data[body+'.'+name]['default'] = 'AUTHENTICATED_RAW_'+name


def projected_task(raw, layer, path, root, *, enforce=True):
    from pxr import Usd, Sdf
    from .instances import original_core, selected, signature
    core = original_core(raw)
    prior = selected(raw['difficulty'])['source_task']
    if signature({k:v for k,v in core.items() if k not in ('all_objects','candidate')}) != signature(
            {k:v for k,v in prior.items() if k not in ('all_objects','candidate')}):
        raise ValueError('CP06 original static fixture parameters differ')
    weights = lambda value: {o['path']:o['grams'] for o in value['all_objects'] if o['kind']=='weight'}
    if weights(core) != weights(prior):
        raise ValueError('CP06 original marked weight inventory differs')
    data = opinions(layer)
    stage = Usd.Stage.Open(layer)
    for obj in core['all_objects']:
        mask_physical_parameters(data, stage, obj)
    fruits = [o for o in core['all_objects'] if o['kind']=='fruit']
    sources = raw['config']['task']['materialization']['objects']
    if len(fruits) != len(sources):
        raise ValueError('CP06 original fruit inventory differs')
    result = {}
    for obj, source in zip(fruits, sources):
        if obj['path'] != '/World/Task'+source['body_path'].removeprefix('/World'):
            raise ValueError('CP06 original produce/body source mapping differs')
        prefix = str(Sdf.Path(obj['path']).GetParentPath())
        selected_data = {k:v for k,v in data.items() if Sdf.Path(k).HasPrefix(prefix)}
        value = fingerprint(selected_data, path, root, {prefix:'/Object'})
        asset = source['asset_sha256']
        if enforce and value != PRODUCE.get(asset):
            raise ValueError('CP06 original produce shape, material or physics differs')
        result[asset] = value
        for key in list(data):
            if Sdf.Path(key).HasPrefix(prefix):
                del data[key]
    static = fingerprint(data, path, root)
    if enforce and static != STATIC_FIXTURE.get(raw['difficulty']):
        raise ValueError('CP06 original static scale, weights or resource closure differs')
    return result, static


def validate_geometry(spec, assets, raw, proof):
    from .instances import selected
    from pxr import Sdf
    roots = [str(Sdf.Path(o['path']).GetParentPath())
        for o in selected(spec.difficulty)['source_task']['all_objects'] if o['kind']=='fruit']
    path, layer = validate_composition(spec, assets, proof,
        BASE_SHA256[spec.difficulty], BASE_GRAPH_SHA256[spec.difficulty], roots,
        skip_retired_children=True)
    projected_task(raw, layer, path, Path(assets.root).resolve(strict=True))
