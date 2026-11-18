"""Exact original deck/offset projection over code-owned Memory control pools."""
from functools import lru_cache
from pathlib import Path

from mind2act_eval.original_asset_graph import layer_bytes
from mind2act_eval.source_geometry import opinions, expected_wrapper

BASE_SHA256 = {
    'easy': '4b8aba0840f7704a1c3ec1f369ffb03908d7f0f9833a5ffc6ec3c83437be3bc7',
    'medium': 'd0d6254b0907fc566ae1880192f95d74e9bbd88126b36146fa73594afc3f7be8',
    'hard': '1cf0b4ae5caaa816e669af9e10e84103b299d813f5e48ed7ede392f57172a721'}
BASE_GRAPH_SHA256 = {
    'easy': 'ad2620db23198c69999d785267bdb107b8a3998d783516e1564594cbed559f1a',
    'medium': 'c01da9c86aa034b83620ccb9c72056a0aeb1a99cefa30db59db7d1e4bb13ad41',
    'hard': '43810f7b31093fa82d160b5e934b94869637537760b1cdfdf007867441a23e67'}


@lru_cache(maxsize=16)
def _anchors(base, code_owned_sha, count):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(base)
    return tuple(tuple(tuple(UsdPhysics.Joint(stage.GetPrimAtPath(
        '/World/Memory/'+folder+'/'+name+str(i))).GetLocalPos0Attr().Get())
        for folder, name in [('Joints', 'Card'), ('ButtonJoints', 'Key')])
        for i in range(count))


def validate_geometry(spec, assets, binding, raw):
    from pxr import Usd, Sdf, Gf
    root = Path(assets.root).resolve(strict=True)
    scene = Path(assets.resolve('station')).resolve(strict=True)
    if not scene.is_relative_to(root) or any(k in spec.config for k in
            ('station_relocation_manifest', 'station_material_relocation_manifest')):
        raise ValueError('M2 original scene must open its authenticated library layer')
    _, layer = layer_bytes(scene)
    if len(layer.subLayerPaths) != 1:
        raise ValueError('M2 original projection requires one retained source pool')
    base = (scene.parent/layer.subLayerPaths[0]).resolve(strict=True)
    if not base.is_relative_to(root):
        raise ValueError('M2 original retained pool escapes root')

    expected = expected_wrapper(layer, standard_clock=True)
    memory = Sdf.CreatePrimInLayer(expected, '/World/Memory')
    memory.specifier = Sdf.SpecifierDef
    memory.typeName = 'Xform'
    delta = raw['board_offset_xy']  # The pinned source control boards use offset0.
    def attribute(prim, name, kind, value, variability=Sdf.VariabilityVarying):
        Sdf.AttributeSpec(prim, name, kind, variability, False).default = value
    attribute(memory, 'xformOp:translate:fixed50Offset', Sdf.ValueTypeNames.Double3,
              Gf.Vec3d(*delta, 0))
    attribute(memory, 'xformOpOrder', Sdf.ValueTypeNames.TokenArray,
              ['xformOp:translate:fixed50Offset'], Sdf.VariabilityUniform)
    anchors = _anchors(str(base), BASE_SHA256[spec.difficulty], len(binding['card_identities']))
    for i, kind in enumerate(raw['layout']['card_types']):
        face = Sdf.CreatePrimInLayer(expected, '/World/Memory/Cards/Card'+str(i)+'/Face')
        api = Sdf.TokenListOp()
        api.prependedItems = ['MaterialBindingAPI']
        face.SetInfo('apiSchemas', api)
        relationship = Sdf.RelationshipSpec(face, 'material:binding', False)
        relationship.targetPathList.explicitItems = [Sdf.Path('/World/Memory/Looks/'+kind)]
        for index, (folder, name) in enumerate([('Joints', 'Card'), ('ButtonJoints', 'Key')]):
            joint = Sdf.CreatePrimInLayer(expected, '/World/Memory/'+folder+'/'+name+str(i))
            p = anchors[i][index]
            attribute(joint, 'physics:localPos0', Sdf.ValueTypeNames.Point3f,
                      Gf.Vec3f(p[0]+delta[0], p[1]+delta[1], p[2]))
    if opinions(layer) != opinions(expected):
        raise ValueError('M2 original scene has unauthorized geometry, physics or offset opinions')
