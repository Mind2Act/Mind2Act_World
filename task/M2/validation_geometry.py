"""Exact clean-VAL-only public picture relocation; original train/test pool authority is unchanged."""
from pathlib import Path
from mind2act_eval.original_asset_graph import layer_bytes
from mind2act_eval.source_geometry import opinions, expected_wrapper
from .original_geometry import _anchors

BASE_SHA256 = {'easy': '05c89a01e0a847c0009d2eec2fdfe427776094be806d16c144ebce12f5ce9673', 'medium': 'b78435c6eef6a828c01f4a0b5b209053daba843041090e9d36058fe7b1442d6d', 'hard': '925daf93fd1a09c767e21ab040b40bbf4366cc0a9ac55aae33bd93bd410f77b0'}
BASE_GRAPH_SHA256 = {'easy': '147f7eb75ff4a284d1c3a3594fd0c0b072976aed35d8cc8d5c04c27625faf253', 'medium': 'bb8bc0ca1aed86b17284f1b6a48365883ab24a2b7748fb2eaf2a1166364bcf52', 'hard': 'f3d9d4fa9eabe8b170c08e2383f98f402f66f0fbf7360a5cd3e81c1c69bda956'}
MDL_IMAGE_SHA256 = {'files/eval-resources/assets-final-20261029/shared/materials/nvidia/2023_1/Base/Metals/Aluminum_Anodized/Aluminum_Anodized_BaseColor.png': 'd1d042502d7d94bca13cee10c63ab5b3801fb0a46e26d79169e13f5b9c7b5a31', 'files/eval-resources/assets-final-20261029/shared/materials/nvidia/2023_1/Base/Metals/Aluminum_Anodized/Aluminum_Anodized_ORM.png': '768f2dbb4f702a9624b912b431efd1a6a8e0ff3e93744cf54f3866ef8f7986e9', 'files/eval-resources/assets-final-20261029/shared/materials/nvidia/2023_1/Base/Metals/Aluminum_Anodized/Aluminum_Anodized_Normal.png': '6dc1cb1b23a9abd766188a85ccbad1a2639d0a9a334f284e359c6c5d4438608e', 'files/eval-resources/assets-final-20261029/shared/materials/nvidia/2023_1/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png': '03ab7494d37e1f6551a26894628c5fe03a65b4d85a09ed11337e184852a580b3', 'files/eval-resources/assets-final-20261029/shared/materials/nvidia/2023_1/Base/Metals/Aluminum_Cast/Aluminum_Cast_ORM.png': '571eba5a99da444b681609a722a29914490c6dae45d230dac471a45875211a4c', 'files/eval-resources/assets-final-20261029/shared/materials/nvidia/2023_1/Base/Metals/Aluminum_Cast/Aluminum_Cast_N.png': '812c0161ea14abc581cb35c47b33dc6005c089f8e82dca6cb4dd4832deb0556c'}

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

    for relative in MDL_IMAGE_SHA256:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
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
