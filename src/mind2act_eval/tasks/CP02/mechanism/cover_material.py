# Ported from cases/CP/CP02/runtime/cover_material.py; SHA256 11bb808a9c63077a134b4c2d27b2222631caafef8910a96e348e7142ae248ec5
"""Opaque, matte plastic finish for the CP02 cover; no geometry/physics changes."""

def apply_opaque_cover(stage):
    from pxr import Gf,Sdf,UsdShade
    root='/World/CP02V1/Cover'
    if not stage.GetPrimAtPath(root):return
    def material(name,color,roughness):
        mat=UsdShade.Material.Define(stage,'/World/Materials/CP02V1_'+name)
        shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1.)
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        return mat
    shell=material('OpaqueCover',(.78,.81,.79),.65)
    rim=material('CoverRim',(.32,.36,.35),.72)
    handle=material('CoverHandle',(.035,.039,.044),.72)
    for name,mat in [('Shell',shell),('Rim',rim),('Handle',handle),('HandleStem',handle)]:
        prim=stage.GetPrimAtPath(root+'/'+name)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    stage.GetPrimAtPath(root+'/Shell').CreateAttribute('primvars:doNotCastShadows',Sdf.ValueTypeNames.Bool).Set(False)
