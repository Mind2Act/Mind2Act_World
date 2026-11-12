"""Actual mesh planes and upward support, without an Isaac runtime."""
import numpy as np
from pxr import Usd, UsdGeom

from mind2act_eval.tasks.CP04.mechanism.containment import Containers, support_reaches


def test_authored_sloping_cavity_does_not_become_a_fixed_small_cylinder():
    stage=Usd.Stage.CreateInMemory();root='/World/Cooking/'
    def cube(path,position,size):
        p=UsdGeom.Cube.Define(stage,path);p.CreateSizeAttr(1)
        p.AddTranslateOp().Set(tuple(position));p.AddScaleOp().Set(tuple(size))
    cube(root+'PotCollision/Bottom',[0,0,.79],[.2,.2,.02])
    # A four-sided test vessel, growing from 80 to 100mm internally.
    points=np.array([[.08,-.15,.8],[.12,-.15,.8],[.12,.15,.8],[.08,.15,.8],
                     [.10,-.15,1.],[.14,-.15,1.],[.14,.15,1.],[.10,.15,1.]])
    faces=[0,1,2,3,4,7,6,5,0,4,5,1,1,5,6,2,2,6,7,3,3,7,4,0]
    for i in range(4):
        angle=i*np.pi/2;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        p=points.copy();p[:,:2]=p[:,:2]@R.T
        mesh=UsdGeom.Mesh.Define(stage,root+f'PotCollision/Wall{i}')
        mesh.CreatePointsAttr(p.tolist());mesh.CreateFaceVertexCountsAttr([4]*6);mesh.CreateFaceVertexIndicesAttr(faces)
    for name,position,size in [('LeftRim',[-.2,0,0],[.01,.4,.01]),('RightRim',[.2,0,0],[.01,.4,.01]),
                               ('FrontRim',[0,-.2,0],[.4,.01,.01]),('BackRim',[0,.2,0],[.4,.01,.01])]:
        cube(root+'CapTray/'+name,position,size)
    c=Containers(stage)
    assert c.pot_contains([.095,0,.99])
    assert not c.pot_contains([.095,0,.81])
    assert not c.pot_contains([.2,0,.99])
    assert not c.pot_contains([0,0,.7])


def test_support_chain_must_reach_real_container_not_a_robot():
    edges={('/food/top','/food/bottom'),('/food/bottom','/pot/floor')}
    assert support_reaches('/food/top','/pot',edges)
    assert not support_reaches('/food/top','/tray',edges)
    assert not support_reaches('/food/top','/pot',{('/food/top','/World/RobotLeft/link7')})


def test_drawer_return_uses_authored_cavity_and_assigned_slot_not_fifteen_mm():
    stage=Usd.Stage.CreateInMemory();root='/World/Cooking/MovingDrawer/'
    for name,p,size in [('Side0',[-.2,0,0],[.01,.4,.05]),('Side2',[.2,0,0],[.01,.4,.05]),
                        ('Front',[0,-.2,0],[.4,.01,.05]),('Back',[0,.2,0],[.4,.01,.05])]:
        shape=UsdGeom.Cube.Define(stage,root+name);shape.CreateSizeAttr(1)
        shape.AddTranslateOp().Set(tuple(p));shape.AddScaleOp().Set(tuple(size))
    slots={'left':[-.1,0,0],'right':[.1,0,0]}
    from mind2act_eval.tasks.CP04.mechanism.containment import DrawerSlots
    drawer=DrawerSlots(stage,slots,[0,0,0])
    assert drawer.contains('left',[-.14,0,0],[0,0,0])
    assert not drawer.contains('left',[.1,0,0],[0,0,0])
    assert not drawer.contains('left',[-.2,0,0],[0,0,0])
    assert drawer.contains('left',[-.14,-.3,0],[0,-.3,0])


def test_speculative_zero_force_contact_cannot_support_a_poured_portion():
    from mind2act_eval.tasks.CP04.mechanism.containment import support_directions
    edge=support_directions('/food','/pot',[0,0,1],[0,0,0],.001)
    assert not support_reaches('/food','/pot',edge)
    assert support_reaches('/food','/pot',support_directions('/food','/pot',[0,0,1],[0,0,.1],.001))
    assert support_reaches('/food','/pot',support_directions('/food','/pot',[0,0,1],[0,0,0],0.))
    assert support_reaches('/food','/pot',support_directions('/pot','/food',[0,0,-1],[0,0,0],-.001))
