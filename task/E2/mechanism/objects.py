"""Original procedural 3D objects: horse, metal gears, ribbed shells, duck, key."""
import math
import numpy as np
from pxr import Gf,UsdGeom,UsdShade
from .geometry import cylinder


def mesh(stage,path,points,faces,material):
    m=UsdGeom.Mesh.Define(stage,path)
    m.CreatePointsAttr(np.asarray(points,dtype=float).tolist())
    m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr('none');UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def profile(stage,path,outline,depth,material):
    n=len(outline);points=[(x,y,z) for y in [-depth/2,depth/2] for x,z in outline]
    faces=[list(reversed(range(n))),list(range(n,2*n))]+[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
    return mesh(stage,path,points,faces,material)


def sphere(stage,path,pos,r,material,scale=(1,1,1)):
    p=UsdGeom.Sphere.Define(stage,path);p.CreateRadiusAttr(r)
    p.AddTranslateOp().Set(Gf.Vec3d(*pos));p.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(material)


def make_object(stage,path,kind,mats):
    root=UsdGeom.Xform.Define(stage,path)
    if kind=='horse':
        outline=[(-.027,.008),(-.025,.029),(-.014,.038),(.002,.037),(.011,.049),(.018,.051),(.025,.044),(.028,.034),(.018,.032),(.013,.023),(.015,.005),(.008,.005),(.004,.020),(-.010,.020),(-.014,.005),(-.021,.005),(-.019,.021),(-.024,.018)]
        profile(stage,path+'/WoodenHorse',outline,.015,mats['wood'])
        for y in [-.008,.008]:sphere(stage,path+('/EyeA' if y<0 else '/EyeB'),(.021,y,.043),.0014,mats['black'])
        profile(stage,path+'/Tail',[(-.024,.030),(-.036,.035),(-.031,.021),(-.026,.020)],.005,mats['wood'])
        cylinder(stage,path+'/Stand',(0,0,.003),.030,.005,mats['wood'])
    elif kind.startswith('gear'):
        teeth=int(kind[4:]);n=teeth*4;outer=[];inner=[]
        for i in range(n):
            a=2*math.pi*i/n;r=.026 if i%4 in (1,2) else .021
            outer.append((r*math.cos(a),r*math.sin(a)));inner.append((.009*math.cos(a),.009*math.sin(a)))
        points=[(x,y,z) for z in [.002,.013] for ring in [outer,inner] for x,y in ring]
        faces=[]
        for i in range(n):
            j=(i+1)%n
            faces.extend([[i,j,2*n+j,2*n+i],[n+i,3*n+i,3*n+j,n+j],
                          [2*n+i,2*n+j,3*n+j,3*n+i],[i,n+i,n+j,j]])
        mesh(stage,path+'/ToothedRing',points,faces,mats['steel'])
        cylinder(stage,path+'/HubDark',(0,0,.001),.008,.001,mats['black'])
    elif kind.startswith('shell'):
        ribs=int(kind[5:]);points=[];faces=[];na=65;nr=18
        for j in range(nr):
            r=j/(nr-1)
            for i in range(na):
                a=-1.22+2.44*i/(na-1)
                radial=.038*r*(1+.045*math.cos((a+1.22)/2.44*ribs*2*math.pi))
                x=radial*math.sin(a);y=radial*math.cos(a)-.013
                z=.004+.015*math.sin(r*math.pi/1.9)+.0025*r*math.cos((a+1.22)/2.44*ribs*2*math.pi)
                points.append((x,y,z))
        for j in range(nr-1):
            for i in range(na-1):a=j*na+i;faces.append([a,a+1,a+na+1,a+na])
        m=mesh(stage,path+'/Scallop',points,faces,mats['ivory']);m.CreateDoubleSidedAttr(True)
        cylinder(stage,path+'/Hinge',(0,-.013,.006),.007,.010,mats['ivory'],'X')
    elif kind=='duck':
        sphere(stage,path+'/Body',(0,0,.018),.022,mats['gold'],(1.25,.82,.8))
        sphere(stage,path+'/Head',(.014,0,.039),.013,mats['gold'])
        sphere(stage,path+'/Beak',(.029,0,.038),.008,mats['orange'],(1.3,.65,.35))
        for y in [-.011,.011]:sphere(stage,path+('/EyeA' if y<0 else '/EyeB'),(.018,y,.044),.0018,mats['black'])
    elif kind=='star':
        outline=[]
        for i in range(10):
            a=math.pi/2+i*math.pi/5; r=.027 if i%2==0 else .012
            outline.append((r*math.cos(a),r*math.sin(a)))
        points=[(x,y,z) for z in [.002,.014] for x,y in outline]
        faces=[list(reversed(range(10))),list(range(10,20))]
        faces.extend([[i,(i+1)%10,(i+1)%10+10,i+10] for i in range(10)])
        mesh(stage,path+'/Star',points,faces,mats['brass'])
    elif kind=='key':
        n=56;points=[];faces=[]
        for z in [.002,.010]:
            for r in [.009,.017]:
                for i in range(n):a=i*2*math.pi/n;points.append((-.015+r*math.cos(a),r*math.sin(a),z))
        for i in range(n):
            j=(i+1)%n;faces.extend([[i,j,n+j,n+i],[2*n+i,3*n+i,3*n+j,2*n+j],[i,2*n+i,2*n+j,j],[n+i,n+j,3*n+j,3*n+i]])
        mesh(stage,path+'/KeyRing',points,faces,mats['brass'])
        outline=[(-.003,.002),(.031,.002),(.031,.011),(.025,.011),(.025,.017),(.017,.017),(.017,.011),(-.003,.011)]
        # Key shaft lies flat; rotate the profile into the tabletop plane.
        p=profile(stage,path+'/KeyTeeth',outline,.008,mats['brass']);p.AddRotateXOp().Set(90)
    else:raise ValueError(kind)
    return root
