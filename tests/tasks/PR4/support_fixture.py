"""Explicit synthetic collision surfaces; never a native-run claim."""
from pxr import Usd, UsdGeom, UsdPhysics, Sdf
from mind2act_eval.tasks.PR4.mechanism.support import capture_support

PATH = '/World/PR4/Support/Normalized/Asset'


def author_support(stage, top=.8, rigid_rest=0.):
    cloth = stage.GetPrimAtPath('/World/DustCloth')
    if not cloth:cloth=UsdGeom.Mesh.Define(stage,'/World/DustCloth').GetPrim()
    UsdPhysics.CollisionAPI.Apply(cloth).CreateCollisionEnabledAttr(True)
    cloth.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(.0006)
    support=UsdGeom.Mesh.Define(stage,PATH)
    support.CreatePointsAttr([(-.5,-.5,top),(.5,-.5,top),(.5,.5,top),(-.5,.5,top)])
    support.CreateFaceVertexCountsAttr([3,3]);support.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
    support.CreateSubdivisionSchemeAttr('none')
    UsdPhysics.CollisionAPI.Apply(support.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(support.GetPrim()).CreateApproximationAttr('none')
    support.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(rigid_rest)
    return support


def support_geometry(top=.8, rigid_rest=0.):
    stage=Usd.Stage.CreateInMemory();author_support(stage,top,rigid_rest)
    return capture_support(stage,dict(body_path='/World/DustCloth',panel_prim=PATH))
