"""USD helpers for CP01 foods/trays.

Sources (A:code/legacy-cp01-run925de9/engine/simulation/scripts, md5 identical to
the Easy collection engine): mind2act_f2_assets.rounded/cylinder and
mind2act_station_scene.material. `material` takes the stage explicitly instead of
the original module-global.
"""
import math
from pxr import Gf, UsdGeom, UsdShade, Sdf


def rounded(stage,path,center,size,radius,material,bevel=.0015):
    """Closed rounded-rectangle extrusion with bevel rings, authored as a mesh."""
    w,d,h=size;r=min(radius,w/2,d/2);b=min(bevel,h/3,r/2)
    points=[]; rings=[]
    for z,inset in [(-h/2,b),(-h/2+b,0),(h/2-b,0),(h/2,b)]:
        ring=[]
        for cx,cy,start in [(w/2-r,d/2-r,0),(-w/2+r,d/2-r,90),(-w/2+r,-d/2+r,180),(w/2-r,-d/2+r,270)]:
            for i in range(9):
                a=math.radians(start+90*i/8);ring.append(len(points));points.append((cx+(r-inset)*math.cos(a),cy+(r-inset)*math.sin(a),z))
        rings.append(ring)
    faces=[list(reversed(rings[0])),rings[-1]]
    for lo,hi in zip(rings,rings[1:]):
        for i in range(len(lo)):j=(i+1)%len(lo);faces.append([lo[i],lo[j],hi[j],hi[i]])
    m=UsdGeom.Mesh.Define(stage,path);m.CreatePointsAttr(points);m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr([v for f in faces for v in f]);m.CreateSubdivisionSchemeAttr('none');m.AddTranslateOp().Set(Gf.Vec3d(*center));UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material);return m


def cylinder(stage,path,center,radius,height,material,axis='Z'):
    p=UsdGeom.Cylinder.Define(stage,path);p.CreateRadiusAttr(radius);p.CreateHeightAttr(height);p.CreateAxisAttr(axis);p.AddTranslateOp().Set(Gf.Vec3d(*center));UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(material);return p


def material(stage, name, color, metal=0, rough=.45):
    m = UsdShade.Material.Define(stage, '/World/Materials/'+name)
    s = UsdShade.Shader.Define(stage, m.GetPath().AppendChild('Shader'))
    s.CreateIdAttr('UsdPreviewSurface')
    for key, typ, value in [('diffuseColor',Sdf.ValueTypeNames.Color3f,Gf.Vec3f(*color)),
                            ('metallic',Sdf.ValueTypeNames.Float,metal),
                            ('roughness',Sdf.ValueTypeNames.Float,rough)]:
        s.CreateInput(key,typ).Set(value)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
    return m
