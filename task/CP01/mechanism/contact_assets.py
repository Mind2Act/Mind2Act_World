"""CP01 task-owned collision shapes. Shared X5 configuration is unchanged."""
import math

def ellipsoid_mesh_points(radii, center, latitudes=10, longitudes=12):
    # 110 points; explicit coordinates avoid unsupported analytic sphere scaling.
    points = [(center[0], center[1], center[2]+radii[2])]
    for i in range(1, latitudes):
        polar=math.pi*i/latitudes
        for j in range(longitudes):
            azimuth=2*math.pi*j/longitudes
            points.append((center[0]+radii[0]*math.sin(polar)*math.cos(azimuth),
                           center[1]+radii[1]*math.sin(polar)*math.sin(azimuth),
                           center[2]+radii[2]*math.cos(polar)))
    points.append((center[0],center[1],center[2]-radii[2]))
    from scipy.spatial import ConvexHull
    hull=ConvexHull(points)
    return points, hull.simplices.tolist()

def repair_foods(stage, roots):
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf,UsdShade
    changes=[]
    for root in roots:
        for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
            name=prim.GetName()
            # Artwork remains renderable. Thin paint/stickers do not carry loads.
            if name in ('FrontBand','LidMark','Stripe','Band','Window') or name.startswith('CrustScore'):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    changes.append({'path':str(prim.GetPath()),'repair':'visual_detail_collision_removed'})
            if prim.IsA(UsdGeom.Sphere) and name.startswith('Nugget'):
                shape=UsdGeom.Sphere(prim);radius=float(shape.GetRadiusAttr().Get())
                ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
                center=next(list(op.Get()) for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeTranslate)
                radii=next([radius*float(v) for v in op.Get()] for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeScale)
                path=root+'/'+name+'_Contact'
                points,faces=ellipsoid_mesh_points(radii,center)
                mesh=UsdGeom.Mesh.Define(stage,path)
                mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
                mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
                mesh.CreateSubdivisionSchemeAttr('none');UsdGeom.Imageable(mesh).CreateVisibilityAttr('invisible')
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
                collision=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                collision.CreateContactOffsetAttr(.002);collision.CreateRestOffsetAttr(0.)
                # Explicitly bounded hull; visual sphere stays byte-identical in stage.
                if hasattr(PhysxSchema,'PhysxConvexHullCollisionAPI'):
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(128)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath('/World/TrayGripMaterial')),materialPurpose='physics')
                if prim.HasAPI(UsdPhysics.CollisionAPI):prim.RemoveAPI(UsdPhysics.CollisionAPI)
                changes.append({'path':str(prim.GetPath()),'repair':'analytic_nonuniform_sphere_to_explicit_convex_mesh','collision_path':path,'radii_m':radii,'vertices':len(points),'visual_unchanged':True})
    from mind2act_eval.tasks.CP01.mechanism.fries_envelope import replace_welded_fries_contact
    import json
    from pathlib import Path
    properties=json.loads(Path(__file__).with_name('cp01_fries_mass_properties.json').read_text())
    changes += replace_welded_fries_contact(stage, roots, properties)
    return changes

def repair_fixture_primitives(stage):
    """Keep conveyor visuals; give stretched analytic cylinders exact mesh inputs."""
    import numpy as np
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,UsdShade,Usd
    changes=[]
    for prim in list(stage.Traverse()):
        path=str(prim.GetPath())
        if not path.startswith('/World/CP01/Conveyor/') or '/OriginalBelt/' in path:continue
        if not prim.IsA(UsdGeom.Cylinder) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        transform=UsdGeom.Xformable(prim)
        world=np.asarray(transform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world_scale=np.linalg.norm(world[:3,:3],axis=1)
        if max(world_scale)-min(world_scale)<1e-6:continue
        cylinder=UsdGeom.Cylinder(prim);radius=float(cylinder.GetRadiusAttr().Get());height=float(cylinder.GetHeightAttr().Get());axis=str(cylinder.GetAxisAttr().Get());count=64
        vertices=[]
        for sign in [-1.,1.]:
            for i in range(count):
                theta=2*math.pi*i/count;p=(radius*math.cos(theta),radius*math.sin(theta),sign*height/2)
                vertices.append((p[2],p[0],p[1]) if axis=='X' else (p[0],p[2],p[1]) if axis=='Y' else p)
        vertices+=([( -height/2,0.,0.),(height/2,0.,0.)] if axis=='X' else [(0.,-height/2,0.),(0.,height/2,0.)] if axis=='Y' else [(0.,0.,-height/2),(0.,0.,height/2)])
        triangles=[]
        for i in range(count):
            j=(i+1)%count;triangles.extend([(i,j,count+j),(i,count+j,count+i),(2*count,j,i),(2*count+1,count+i,count+j)])
        matrix=transform.GetLocalTransformation()
        if (axis=='Y') != (np.linalg.det(np.asarray(matrix)[:3,:3])<0):triangles=[tuple(reversed(f)) for f in triangles]
        vertices=[matrix.Transform(Gf.Vec3d(*p)) for p in vertices]
        if prim.GetName() in ('Roller0', 'Roller1'):
            # The moving belt owns food support. Keep real roller collisions
            # underneath its cover instead of a second static surface at the top.
            parent=UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            delta=parent.GetInverse().TransformDir(Gf.Vec3d(0.,0.,-0.004))
            vertices=[v+delta for v in vertices]
        mesh=UsdGeom.Mesh.Define(stage,path+'_Contact');mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices]);mesh.CreateFaceVertexCountsAttr([3]*len(triangles));mesh.CreateFaceVertexIndicesAttr([v for face in triangles for v in face]);mesh.CreateSubdivisionSchemeAttr('none');UsdGeom.Imageable(mesh).CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        old=PhysxSchema.PhysxCollisionAPI(prim);new=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        for getter,creator in [(old.GetContactOffsetAttr,new.CreateContactOffsetAttr),(old.GetRestOffsetAttr,new.CreateRestOffsetAttr)]:
            value=getter().Get()
            if value is not None and math.isfinite(float(value)):creator(value)
        for relation in prim.GetRelationships():
            if str(relation.GetName()).startswith('material:binding:physics'):mesh.GetPrim().CreateRelationship(relation.GetName()).SetTargets(relation.GetTargets())
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        changes.append({'path':path,'repair':'nonuniform_conveyor_cylinder_to_static_mesh','collision_path':path+'_Contact','original_world_scale':world_scale.tolist(),'visual_unchanged':True,'segments':count})
    # These two finite closed rounded rails are convex fixtures. Use the
    # same authored vertices as a convex hull instead of long thin triangles.
    # All artwork, transforms, contact/material/CCD and X5 stay unchanged.
    for name in ('Rail0', 'Rail1'):
        prim=stage.GetPrimAtPath('/World/CP01/Conveyor/'+name)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError('CP01 final rail shape is missing: '+name)
        api=UsdPhysics.MeshCollisionAPI.Apply(prim)
        previous=api.GetApproximationAttr().Get()
        if previous != 'none':
            raise RuntimeError('Unexpected prior CP01 rail approximation: '+str(previous))
        api.CreateApproximationAttr('convexHull')
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(255)
        changes.append({'path':str(prim.GetPath()),'repair':'closed_convex_rail_triangle_to_same_vertex_convex_hull','previous_approximation':previous,'hull_vertex_limit':255,'visual_and_authored_vertices_unchanged':True,'transform_unchanged':True})
    from mind2act_eval.tasks.CP01.mechanism.contact_contract import audit_stage
    changes.append({'repair':'all_task_assets_contract_validated','receipt':audit_stage(stage)})
    return changes
