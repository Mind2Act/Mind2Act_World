"""CPU USD composition tests; no native loading or physics claim."""
from types import SimpleNamespace as NS
import pytest
from pxr import Gf,Sdf,Usd,UsdGeom
from .test_p2_measurements import source
from mind2act_eval.tasks.P2.binding import geometry
from mind2act_eval.tasks.P2.scene import author_task


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_exact_source_task_overlays_stay_in_task_namespace(tmp_path,level):
    row=source(level);g=geometry(row)
    path=tmp_path/'task.usda';asset=Usd.Stage.CreateNew(str(path))
    root=UsdGeom.Xform.Define(asset,'/Task');asset.SetDefaultPrim(root.GetPrim())
    for name in ('Stove','Board','Stove/PreplacedTier',*g['decoy_xyz']):
        UsdGeom.Xform.Define(asset,'/Task/'+name).AddTranslateOp().Set(Gf.Vec3d(0))
    UsdGeom.Xform.Define(asset,'/Task/Stove/PotGeometry').AddScaleOp().Set(Gf.Vec3f(1))
    for name in g['target_initial_xyz']:
        UsdGeom.Xform.Define(asset,'/Task/'+name).AddTranslateOp().Set(Gf.Vec3d(*g['target_initial_xyz'][name]))
    for name in [*g['target_initial_xyz'],*g['decoy_xyz']]:
        UsdGeom.Xform.Define(asset,'/Task/'+name+'/Food')
    asset.GetRootLayer().Save();original=path.read_bytes()
    food_path=tmp_path/'food.usda';food=Usd.Stage.CreateNew(str(food_path))
    food.SetDefaultPrim(UsdGeom.Xform.Define(food,'/Food').GetPrim());food.GetRootLayer().Save()
    stage=Usd.Stage.CreateInMemory()
    robot=UsdGeom.Xform.Define(stage,'/World/RobotLeft').GetPrim()
    robot.CreateAttribute('sourceMarker',Sdf.ValueTypeNames.String).Set('shared-owned')
    author_task(stage,row,g,NS(resolve=lambda name:path if name=='task_scene' else food_path))
    assert list(stage.GetPrimAtPath('/World/Task/Stove').GetAttribute('xformOp:translate').Get())==g['stove_xyz']
    assert list(stage.GetPrimAtPath('/World/Task/Board').GetAttribute('xformOp:translate').Get())==g['board_xyz']
    assert robot.GetAttribute('sourceMarker').Get()=='shared-owned'
    assert not robot.GetAttribute('xformOp:translate')
    assert path.read_bytes()==original
    for name,position in g['target_initial_xyz'].items():
        assert list(stage.GetPrimAtPath('/World/Task/'+name).GetAttribute('xformOp:translate').Get())==position
    for name,position in g['decoy_xyz'].items():
        assert list(stage.GetPrimAtPath('/World/Task/'+name).GetAttribute('xformOp:translate').Get())==position
    with pytest.raises(ValueError,match='clean station'):
        author_task(stage,row,g,NS(resolve=lambda name:path))
