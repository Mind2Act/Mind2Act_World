"""Source task props only. Station, cameras, robot and app stay shared-owned."""
import numpy as np
from pxr import Gf,Sdf,UsdGeom,UsdShade,UsdPhysics
from .shapes import rounded,cylinder,material
from .layouts import from_case,CONFIG
from .retrieval_geometry import STORAGE_CENTER,BODY_Z

def write_ink_mesh(mesh,mask,z):
    """Exact live raster as opaque USD quads; no target-specific visual layer."""
    runs=[];colors=[]
    for color in range(1,len(CONFIG['colors'])+1):
        a=np.pad((mask==color).astype(np.int8),((0,0),(1,1)))
        diff=np.diff(a,axis=1);rows,left=np.where(diff==1);_,right=np.where(diff==-1)
        runs.append((rows,left,right));colors.extend([CONFIG['colors'][color-1]['material']]*len(rows))
    rows=np.concatenate([r[0] for r in runs]);left=np.concatenate([r[1] for r in runs]);right=np.concatenate([r[2] for r in runs])
    n=len(rows)
    if not n:
        mesh.GetPointsAttr().Set([]);mesh.GetFaceVertexCountsAttr().Set([]);mesh.GetFaceVertexIndicesAttr().Set([])
        return
    x0=left/2000.-.22;x1=right/2000.-.22
    y1=.1425-rows/2000.;y0=y1-.0005
    pts=np.empty((n,4,3),dtype=np.float32)
    pts[:,:,2]=z
    pts[:,0,0]=x0;pts[:,0,1]=y0
    pts[:,1,0]=x1;pts[:,1,1]=y0
    pts[:,2,0]=x1;pts[:,2,1]=y1
    pts[:,3,0]=x0;pts[:,3,1]=y1
    mesh.CreatePointsAttr().Set(pts.reshape(-1,3).tolist())
    mesh.CreateFaceVertexCountsAttr().Set([4]*n)
    mesh.CreateFaceVertexIndicesAttr().Set(np.arange(n*4).tolist())
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('displayColor',Sdf.ValueTypeNames.Color3fArray,UsdGeom.Tokens.uniform).Set(colors)


def build_props(stage,output,difficulty,seed,cfg,source_case):
    case,target,work=from_case(source_case)
    z=cfg['world_support_z'];board_z=z+.03315
    metal=material(stage,'CP09SatinAluminium',(.44,.49,.52),.68,.43)
    dark=material(stage,'CP09BlackPolymer',(.025,.035,.043),.05,.64)
    white=material(stage,'CP09BoardBacking',(.92,.915,.86),0,.85)
    teal=material(stage,'CP09Teal',(.025,.3,.30),.12,.5)
    navy=material(stage,'CP09AnodizedBlue',(.045,.15,.24),.52,.39)
    silver=material(stage,'CP09Steel',(.55,.59,.63),.74,.34)
    roots={};textures={};ink_meshes={}
    ink_material=material(stage,'CP09ActualInk',(1.,1.,1.),0,.9)
    reader=UsdShade.Shader.Define(stage,ink_material.GetPath().AppendChild('InkColor'))
    reader.CreateIdAttr('UsdPrimvarReader_float3');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    shader=ink_material.ComputeSurfaceSource()[0]
    shader.GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(),'result')
    for name,xy in [('Work',tuple(STORAGE_CENTER[:2]))]:
        path='/World/CP09/'+name
        root=UsdGeom.Xform.Define(stage,path)
        motion=root.AddTranslateOp();motion.Set(Gf.Vec3d(*xy,float(STORAGE_CENTER[2]-BODY_Z) if name=='Work' else 0.))
        for sx in [-1,1]:
            for sy in [-1,1]:rounded(stage,path+f'/Foot{sx+1}{sy+1}',(sx*.205,sy*.133,z+.004),(.035,.025,.008),.006,dark)
        rounded(stage,path+'/Backplate',(0,0,z+.016),(.48,.33,.020),.012,dark)
        rounded(stage,path+'/BrushedFrame',(0,0,z+.027),(.476,.326,.009),.010,metal)
        rounded(stage,path+'/BoardBacking',(0,0,z+.032),(.446,.291,.002),.003,white)
        paper=UsdGeom.Mesh.Define(stage,path+'/BoardFace')
        paper.CreatePointsAttr([(-.22,-.1425,board_z),(.22,-.1425,board_z),(.22,.1425,board_z),(-.22,.1425,board_z)])
        paper.CreateFaceVertexCountsAttr([4]);paper.CreateFaceVertexIndicesAttr([0,1,2,3]);paper.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(paper.GetPrim()).Bind(white)
        mesh=UsdGeom.Mesh.Define(stage,path+'/ActualInk')
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(ink_material)
        write_ink_mesh(mesh,target,board_z+.00008)
        ink_meshes[name]=mesh
        for sx in [-1,1]:
            for sy in [-1,1]:cylinder(stage,path+f'/FrameScrew{sx+1}{sy+1}',(sx*.228,sy*.151,z+.032),.0038,.0016,silver)
        roots[name]=motion
    # An open raised platform: no canopy or walls above the memorized canvas.
    # Keep the original camera/lights and expose the entire far board edge.
    # The board is removable;
    # none of its visuals or collision shapes is parented to this machine.
    machine=UsdGeom.Xform.Define(stage,'/World/CP09/StorageMachine')
    machine.AddTranslateOp().Set(Gf.Vec3d(0,0,0))
    UsdPhysics.RigidBodyAPI.Apply(machine.GetPrim()).CreateKinematicEnabledAttr(True)
    machine_parts=[('Base',(0,.445,z+.018),(.535,.330,.036),metal),
        ('RearDrive',(0,.586,z+.075),(.535,.028,.150),metal)]
    for sx in [-1,1]:
        machine_parts += [(f'Column{sx+1}',(sx*.200,.440,z+.076),(.028,.045,.098),metal),
                          (f'Shelf{sx+1}',(sx*.200,.380,.893),(.028,.200,.010),dark)]
    for name,position,size,mat in machine_parts:
        part=rounded(stage,'/World/CP09/StorageMachine/'+name,position,size,.005,mat)
        UsdPhysics.CollisionAPI.Apply(part.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr('convexHull')
    # The integrated shutter hides this same board while annotations are added.
    shutter=UsdGeom.Xform.Define(stage,'/World/CP09/StorageMachine/Shutter')
    roots['Shutter']=shutter.AddTranslateOp()
    roots['Shutter'].Set(Gf.Vec3d(0,.680,0))
    rounded(stage,'/World/CP09/StorageMachine/Shutter/Panel',
            (0,0,float(STORAGE_CENTER[2]+board_z-BODY_Z+.006)),(.446,.291,.003),.003,metal)
    # Same simple machined accessories as PR3, not a replacement tabletop.
    rounded(stage,'/World/CP09/ToolTray',(0,-.220,z+.008),(.52,.11,.016),.010,metal)
    rounded(stage,'/World/CP09/ToolTrayLiner',(0,-.220,z+.0165),(.505,.096,.002),.008,dark)
    # Matching left/right insets keep the tool groups centered at +/-150 mm.
    for side,name in [(-1,'Left'),(1,'Right')]:
        rounded(stage,f'/World/CP09/TrayInset{name}',(side*.150,-.220,z+.0176),(.180,.090,.0006),.008,navy)
        rounded(stage,f'/World/CP09/TrayPad{name}',(side*.150,-.220,z+.0180),(.174,.084,.0005),.007,dark)
    for k in range(3):
        rounded(stage,f'/World/CP09/CapBayMark{k}',(-.040+k*.040,-.259,z+.0178),(.024,.001,.0005),.0003,silver)
    # Tray and a plain pen socket support independent gravity-enabled tools.
    for name in ['ToolTray','ToolTrayLiner']:
        mesh=stage.GetPrimAtPath('/World/CP09/'+name)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexHull')
    for color in range(1,case['color_count']+1):
        cx=.150+(color-(case['color_count']+1)/2)*.065
        for i,(dx,dy,sx,sy) in enumerate([(-.019,0,.004,.042),(.019,0,.004,.042),(0,-.019,.034,.004),(0,.019,.034,.004)]):
            wall=rounded(stage,f'/World/CP09/PenSocket{color}_{i}',(cx+dx,-.220+dy,z+.040), (sx,sy,.047),.002,metal)
            UsdPhysics.CollisionAPI.Apply(wall.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(wall.GetPrim()).CreateApproximationAttr('convexHull')
    return dict(stage=stage,cfg=cfg,case=case,target=target,work=work,board_z=board_z,roots=roots,textures=textures,ink_meshes=ink_meshes,output=output)
