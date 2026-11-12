"""Actual authored rack bounds exclude the moving tool itself."""
from pxr import Usd,UsdGeom,UsdPhysics
from mind2act_eval.tasks.CP05.mechanism.rack import RackGeometry


def test_actual_support_footprint_has_no_ten_mm_gate_or_tool_expansion():
    stage=Usd.Stage.CreateInMemory();root='/World/CP12/Tools/Left/'
    def cube(path,pos,size):
        p=UsdGeom.Cube.Define(stage,path);p.CreateSizeAttr(1)
        p.AddTranslateOp().Set(tuple(pos));p.AddScaleOp().Set(tuple(size))
        UsdPhysics.CollisionAPI.Apply(p.GetPrim())
    cube(root+'Cradle',[0,0,1],[.08,.04,.01])
    cube(root+'Mallet',[.3,0,1],[.02,.02,.05])
    rack=RackGeometry(stage,['Left'])
    # The root mallet transform is supplied separately as its measured pose.
    assert rack.contains('Left',[.02,0,1],[1,0,0,0])
    assert not rack.contains('Left',[.05,0,1],[1,0,0,0])
    assert not rack.contains('Left',[.3,0,1],[1,0,0,0])
