"""Original PR5 visual-only ink quads; no physical body or scoring state is moved."""


class Ink:
    def __init__(self, stage, textures, card_xy):
        from pxr import Sdf, UsdShade
        self.stage, self.card_xy, self.count = stage, card_xy, 0
        self.materials = {}
        for name, texture in textures.items():
            path = '/World/Looks/PR5Ink' + name
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + '/S')
            shader.CreateIdAttr('UsdPreviewSurface')
            tex = UsdShade.Shader.Define(stage, path + '/T')
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture))
            tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            for axis in ('S', 'T'):
                tex.CreateInput('wrap'+axis, Sdf.ValueTypeNames.Token).Set('clamp')
            reader = UsdShade.Shader.Define(stage, path + '/UV')
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.5)
            shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), 'a')
            shader.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(.05)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            self.materials[name] = material

    def add(self, marks):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        for mark in marks:
            self.count += 1
            mesh = UsdGeom.Mesh.Define(self.stage, '/World/Task/Marks/Ink' + str(self.count))
            mesh.CreatePointsAttr([(-.025, -.025, 0.), (.025, -.025, 0.), (.025, .025, 0.), (-.025, .025, 0.)])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
            mesh.AddTranslateOp().Set(Gf.Vec3d(self.card_xy[0] + mark.x, self.card_xy[1] + mark.y,
                                             .768 + .008 + .0008 + .0002 + .00005*self.count))
            mesh.AddRotateZOp().Set(mark.yaw_deg)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials['smeared' if mark.smeared else 'clean'])
