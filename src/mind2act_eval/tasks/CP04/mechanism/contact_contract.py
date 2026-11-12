"""CP04 cap/body and visible-ground contact repair; no object motion or grasp aid."""
from pxr import Usd,Sdf,UsdGeom,UsdPhysics,UsdShade,PhysxSchema

def apply(stage):
    scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1:raise ValueError("Exactly one authored physics scene is required")
    scenes[0].CreateAttribute("physxScene:solveArticulationContactLast",Sdf.ValueTypeNames.Bool).Set(True)
    pairs=[]
    for i in range(4):
        body='/World/Cooking/'+('Jar0' if i==3 else 'Condiment'+str(i))
        lid='/World/Cooking/'+('TrialLid' if i==3 else 'CondimentCap'+str(i))
        joint='/World/Cooking/'+('ScrewGuide' if i==3 else 'CondimentScrew'+str(i))
        for root,other in ((body,lid),(lid,body)):
            prim=stage.GetPrimAtPath(root)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError(root)
            # The primary Jar0/TrialLid template omitted CCD; all four groups
            # need swept collision at120Hz, including fast passive drops.
            rigid=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid.CreateEnableCCDAttr(True)
            rigid.CreateEnableSpeculativeCCDAttr(True)
            rigid.CreateMaxDepenetrationVelocityAttr(1.)
            rigid.CreateSolverPositionIterationCountAttr(255)
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                rel=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
                rel.SetTargets([p for p in rel.GetTargets() if str(p)!=other])
        guide=UsdPhysics.Joint(stage.GetPrimAtPath(joint))
        if not guide:raise ValueError(joint)
        # Keep the authored screw-connected collision rule. Release enables
        # own-body/cap collision before the next physical solve. Permanent
        # filteredPairs are still removed, so the loose cap cannot ghost through.
        guide.CreateCollisionEnabledAttr(not bool(guide.GetJointEnabledAttr().Get()))
        pairs.append(dict(body=body,cap=lid,joint=joint))
    floor=stage.GetPrimAtPath('/World/Floor')
    if not floor or not floor.IsA(UsdGeom.Mesh):raise ValueError('Visible station floor missing')
    if floor.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('Station floor must remain static')
    # Collide with the existing visible closed mesh, including its exact0.05m
    # top. Do not introduce a catch plane at tabletop height or move dropped items.
    UsdPhysics.CollisionAPI.Apply(floor).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(floor).CreateApproximationAttr('none')
    contact=PhysxSchema.PhysxCollisionAPI.Apply(floor)
    contact.CreateContactOffsetAttr(.001);contact.CreateRestOffsetAttr(0.)
    material=UsdShade.Material(stage.GetPrimAtPath('/World/Cooking/Contact'))
    if not material:raise ValueError('Existing physical contact material missing')
    UsdShade.MaterialBindingAPI.Apply(floor).Bind(material,materialPurpose='physics')
    PhysxSchema.PhysxContactReportAPI.Apply(floor).CreateThresholdAttr(0.)
    return verify(stage)

def verify(stage):
    scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1 or scenes[0].GetAttribute("physxScene:solveArticulationContactLast").Get() is not True:raise ValueError("Dynamic articulation contact ordering absent")
    rows=[]
    for i in range(4):
        b='/World/Cooking/'+('Jar0' if i==3 else 'Condiment'+str(i))
        l='/World/Cooking/'+('TrialLid' if i==3 else 'CondimentCap'+str(i))
        j='/World/Cooking/'+('ScrewGuide' if i==3 else 'CondimentScrew'+str(i))
        for root,other in ((b,l),(l,b)):
            p=stage.GetPrimAtPath(root)
            rb=PhysxSchema.PhysxRigidBodyAPI(p)
            if not rb.GetEnableCCDAttr().Get() or not rb.GetEnableSpeculativeCCDAttr().Get():raise ValueError('CCD disabled: '+root)
            if rb.GetSolverPositionIterationCountAttr().Get()!=255:raise ValueError('Position solver setting mismatch: '+root)
            if abs(rb.GetMaxDepenetrationVelocityAttr().Get()-1.)>1e-8:raise ValueError('Depenetration setting mismatch: '+root)
            targets=UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets() if p.HasAPI(UsdPhysics.FilteredPairsAPI) else []
            if any(str(t)==other for t in targets):raise ValueError('Own cap collision still filtered: '+root)
        joint=UsdPhysics.Joint(stage.GetPrimAtPath(j));connected=bool(joint.GetJointEnabledAttr().Get());collides=bool(joint.GetCollisionEnabledAttr().Get())
        if collides==connected:raise ValueError('Joint contact phase mismatch: '+j)
        rows.append(dict(body=b,cap=l,body_cap_collision_enabled=collides,joint_collision_enabled=collides,screw_connected=connected,collision_after_release=True,body_and_cap_ccd_enabled=True,speculative_ccd_enabled=True,max_depenetration_velocity_m_s=1.,solver_position_iterations=255))
    floor=stage.GetPrimAtPath('/World/Floor')
    if not UsdPhysics.CollisionAPI(floor).GetCollisionEnabledAttr().Get():raise ValueError('Ground collision disabled')
    box=UsdGeom.BBoxCache(0,['default','render','proxy','guide'],False,True).ComputeWorldBound(floor).ComputeAlignedRange()
    return dict(passed=True,pairs=rows,floor_path=str(floor.GetPath()),floor_bounds=[list(box.GetMin()),list(box.GetMax())],geometry_mass_inertia_drives_unchanged=True,object_pose_writes=False)


def sync_release_contacts(stage, jars):
    """Respect existing actual screw release; never change pose or release threshold."""
    for item in jars.items.values():
        joint=item['guide'];expected=not bool(joint.GetJointEnabledAttr().Get())
        attr=joint.GetCollisionEnabledAttr()
        if bool(attr.Get())!=expected:joint.CreateCollisionEnabledAttr(expected)
