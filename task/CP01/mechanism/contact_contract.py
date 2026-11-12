"""CP01 contact asset checks shared by every difficulty and fixed seed.

Keep physically open work trays and food-specific shapes. Shape checks do not
replace native grasp, release, contact or transport qualification.
"""
from pathlib import Path
import json

def load_contract():
    return json.loads(Path(__file__).with_suffix('.json').read_text())

def classify(actor):
    for name,kind in [('BurgerBox','burger'),('Fry20','fries'),('CartonFloor','nuggets'),('Sleeve','pie'),('Floor','worktray')]:
        if actor.GetChild(name):return kind
    raise RuntimeError('Unknown CP01 contact asset: '+str(actor.GetPath()))

def audit_stage(stage):
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
    spec=load_contract();cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_],False,True)
    rows=[];transforms=UsdGeom.XformCache()
    for actor in stage.GetPrimAtPath('/World/MealTrays').GetChildren():
        kind=classify(actor)
        shapes=[p for p in Usd.PrimRange(actor) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        expected=spec['tray'] if kind=='worktray' else spec['foods'][kind]
        if len(shapes)!=expected['shape_count']:raise RuntimeError('CP01 collider count changed: '+str(actor.GetPath()))
        mass=float(UsdPhysics.MassAPI(actor).GetMassAttr().Get())
        if abs(mass-expected['mass_kg'])>1e-7:raise RuntimeError('CP01 mass changed: '+str(actor.GetPath()))
        shape_rows=[];all_local_points=[];inverse=transforms.GetLocalToWorldTransform(actor).GetInverse()
        for p in shapes:
            if not p.IsA(UsdGeom.Mesh):raise RuntimeError('Dynamic CP01 contact must be explicit mesh: '+str(p.GetPath()))
            mesh=UsdGeom.Mesh(p);points=np.asarray(mesh.GetPointsAttr().Get(),float)
            ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
            if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or len(points)<4 or len(ids)!=sum(counts) or ids.min()<0 or ids.max()>=len(points):
                raise RuntimeError('Invalid CP01 mesh: '+str(p.GetPath()))
            transform=transforms.GetLocalToWorldTransform(p)*inverse
            local_points=np.asarray([list(transform.Transform(Gf.Vec3d(*point))) for point in points])
            all_local_points.extend(local_points.tolist())
            local_size=np.ptp(local_points,axis=0)
            if kind=='worktray' and np.max(abs(local_size-np.asarray(spec['tray']['part_sizes_m'][p.GetName()])))>1e-6:raise RuntimeError('Worktray part dimension mismatch: '+str(p.GetPath()))
            bounds=cache.ComputeWorldBound(p).ComputeAlignedRange();size=np.asarray(bounds.GetSize(),float)
            if not np.isfinite(size).all() or np.any(size<=0):raise RuntimeError('Invalid CP01 shape bounds: '+str(p.GetPath()))
            approximation=UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
            if approximation!='convexHull':raise RuntimeError('Unexpected dynamic CP01 approximation: '+str(p.GetPath()))
            api=PhysxSchema.PhysxCollisionAPI(p)
            contact=float(api.GetContactOffsetAttr().Get());rest=float(api.GetRestOffsetAttr().Get())
            if not np.isfinite([contact,rest]).all() or contact<rest or rest!=0:raise RuntimeError('Invalid CP01 collision offsets: '+str(p.GetPath()))
            shape_rows.append({'path':str(p.GetPath()),'world_size_m':size.tolist(),'approximation':str(approximation),'contact_offset_m':contact,'rest_offset_m':rest})
        if kind!='worktray' and np.max(abs(np.ptp(np.asarray(all_local_points),axis=0)-np.asarray(expected['contact_envelope_size_m'])))>1e-6:raise RuntimeError('Food contact envelope dimensions changed: '+str(actor.GetPath()))
        if kind=='worktray':
            # An open tray must keep separate floor, rims, feet and grip pieces.
            names={p.GetName() for p in shapes}
            required={'Floor','X0','X2','Y0','Y2','Foot00','Foot01','Foot10','Foot11','GripBridgeCenter','GripCenter'}
            if names!=required:raise RuntimeError('Worktray hollow-shell topology changed')
            floor=np.asarray(UsdGeom.Mesh(actor.GetChild('Floor')).GetPointsAttr().Get(),float)
            if abs(np.ptp(floor[:,2])-.005)>1e-6:raise RuntimeError('Worktray floor thickness mismatch')
        rows.append({'root':str(actor.GetPath()),'kind':kind,'mass_kg':mass,'shapes':shape_rows})
    fixture_rows=[]
    for path in [spec['conveyor']['moving_surface'],*spec['conveyor']['finite_rail_paths'],'/World/Worktable']:
        root=stage.GetPrimAtPath(path)
        if not root:raise RuntimeError('Missing CP01 fixture: '+path)
        active=[p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        if not active:raise RuntimeError('CP01 fixture has no active contacts: '+path)
        for p in active:
            bounds=cache.ComputeWorldBound(p).ComputeAlignedRange();size=np.asarray(bounds.GetSize(),float)
            if not np.isfinite(size).all() or np.any(size<=0):raise RuntimeError('Invalid CP01 fixture bounds: '+str(p.GetPath()))
            fixture_rows.append({'path':str(p.GetPath()),'world_min_m':list(bounds.GetMin()),'world_max_m':list(bounds.GetMax()),'world_size_m':size.tolist()})
        if path==spec['conveyor']['moving_surface']:
            b=cache.ComputeWorldBound(root).ComputeAlignedRange()
            if abs(b.GetSize()[1]-.18)>1e-6 or abs(b.GetMidpoint()[1]+.025)>1e-6 or abs(b.GetMax()[2]-.858)>1e-6:raise RuntimeError('CP01 functional belt geometry changed')
    refs=stage.GetPrimAtPath('/World/ReferenceOrders')
    if refs and any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(refs)):raise RuntimeError('Observation-only order became a physical obstacle')
    return {'passed':True,'actors':rows,'fixtures':fixture_rows,'reference_orders_nonphysical':True,'native_dynamic_qualification':False}

def audit_native(w):
    import numpy as np
    spec=load_contract();rows=[]
    for path,body in zip(w.paths,w.bodies):
        kind=classify(w.stage.GetPrimAtPath(path));ref=spec['tray'] if kind=='worktray' else spec['foods'][kind]
        view=body._rigid_prim_view._physics_view
        mass=float(np.asarray(view.get_masses()).reshape(-1)[0]);com=np.asarray(view.get_coms()).reshape(-1,7)[0];inertia=np.asarray(view.get_inertias()).reshape(-1,9)[0]
        material=np.asarray(view.get_material_properties()).reshape(-1,3);contact=np.asarray(view.get_contact_offsets()).reshape(-1);rest=np.asarray(view.get_rest_offsets()).reshape(-1)
        if abs(mass-ref['mass_kg'])>1e-7 or len(material)!=ref['shape_count'] or np.max(abs(material-[.9,.75,0.]))>1e-6 or np.max(abs(contact-.002))>1e-7 or np.max(abs(rest))>1e-7:raise RuntimeError('Native CP01 property mismatch: '+path)
        if not np.isfinite(np.r_[com,inertia]).all() or np.min(np.linalg.eigvalsh(inertia.reshape(3,3)))<=0:raise RuntimeError('Invalid native CP01 inertia: '+path)
        if kind!='worktray' and (np.max(abs(com[:3]-ref['COM_local_xyz']))>1e-7 or np.max(abs(inertia-ref['inertia_body_column_major']))>5e-10):raise RuntimeError('Native CP01 food COM/inertia changed: '+path)
        rows.append({'path':path,'kind':kind,'mass_kg':mass,'COM_local_xyz':com[:3].tolist(),'inertia_body_column_major':inertia.tolist(),'native_shapes':len(material),'native_material':material.tolist(),'contact_offsets_m':contact.tolist(),'rest_offsets_m':rest.tolist()})
    return {'passed':True,'rows':rows,'shape_slot_to_USD_mapping':'not established','not_physical_task_qualification':True}
