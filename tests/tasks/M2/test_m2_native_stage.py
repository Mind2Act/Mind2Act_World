"""Read-only USD IO double checks; actual prepared layers are private diagnostics."""
import sys
from types import SimpleNamespace as NS

import pytest

from mind2act_eval.tasks.M2.native_stage import check_stage


def stage_double(monkeypatch, count):
    class Rigid:pass
    class Joint:
        def __init__(self,prim):self.prim=prim
        def GetAxisAttr(self):return NS(Get=lambda:self.prim.axis)
        def GetBody0Rel(self):return NS(GetTargets=lambda:self.prim.body0)
        def GetBody1Rel(self):return NS(GetTargets=lambda:self.prim.body1)
    class Revolute(Joint):pass
    class Prismatic(Joint):pass
    prims={}
    for i in range(count):
        for kind,root,jroot,schema,axis in [('Card','Cards','Joints',Revolute,'Y'),('Key','Buttons','ButtonJoints',Prismatic,'Z')]:
            path=f'/World/Memory/{root}/{kind}{i}'
            prims[path]=NS(HasAPI=lambda api:api is Rigid)
            prims[f'/World/Memory/{jroot}/{kind}{i}']=NS(IsA=lambda cls,schema=schema:cls is schema,axis=axis,body0=[],body1=[path])
        prims[f'/World/Memory/Paired{i}']=NS()
    monkeypatch.setitem(sys.modules,'pxr',NS(UsdPhysics=NS(RigidBodyAPI=Rigid,RevoluteJoint=Revolute,PrismaticJoint=Prismatic),
        UsdGeom=NS(GetStageMetersPerUnit=lambda stage:stage.units,GetStageUpAxis=lambda stage:stage.up)))
    return NS(GetPrimAtPath=lambda path:prims.get(path),units=1.,up='Z'),prims


@pytest.mark.parametrize('count',[4,8,16])
def test_source_relationships_and_units_read_only(monkeypatch,count):
    stage,prims=stage_double(monkeypatch,count)
    assert check_stage(stage,count)['cards']==count
    # This double deliberately offers no mutation methods.
    assert all(not hasattr(prim,'Set') for prim in prims.values())


@pytest.mark.parametrize('damage',['units','axis','world_anchor','wrong_body','missing_body','wrong_type'])
def test_mismatched_prepared_stage_fails_before_robot_setup(monkeypatch,damage):
    stage,prims=stage_double(monkeypatch,4)
    joint=prims['/World/Memory/Joints/Card0']
    if damage=='units':stage.units=.01
    if damage=='axis':joint.axis='X'
    if damage=='world_anchor':joint.body0=['/World/RobotLeft']
    if damage=='wrong_body':joint.body1=['/World/Memory/Cards/Card1']
    if damage=='missing_body':del prims['/World/Memory/Buttons/Key0']
    if damage=='wrong_type':joint.IsA=lambda cls:False
    with pytest.raises(ValueError):check_stage(stage,4)
