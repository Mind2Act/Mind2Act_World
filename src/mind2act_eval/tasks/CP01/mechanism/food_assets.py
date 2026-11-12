"""Original simple fast-food meshes with explicit primitive contact geometry.

All components of a serving share one rigid body. No deformable/loose-food claim.
"""
from pxr import Gf,UsdGeom,UsdShade,UsdPhysics
from mind2act_eval.tasks.CP01.mechanism.usd_helpers import rounded,cylinder,material

CATALOG=[('burger','Boxed burger'),('fries','Boxed fries'),('nuggets','Nugget carton'),('pie','Pie sleeve'),('tart','Egg tart box')]

def build_food(stage,path,kind,position,dynamic=False):
 root=UsdGeom.Xform.Define(stage,path);root.AddTranslateOp().Set(Gf.Vec3d(*position));parts=[]
 colors={'bun':(.70,.35,.10),'bread':(.93,.65,.27),'meat':(.19,.065,.024),'lettuce':(.20,.43,.06),'cheese':(.95,.63,.05),'red':(.68,.055,.035),'white':(.93,.89,.77),'gold':(.87,.49,.10),'custard':(.98,.75,.22),'brown':(.42,.16,.035)}
 mats={k:material(stage,'MealFood_'+k,v,.0,.7) for k,v in colors.items()}
 decorative=[]
 def box(name,center,size,mat,contact=True):
  obj=rounded(stage,path+'/'+name,center,size,min(.004,min(size)/3),mats[mat]);(parts if contact else decorative).append(obj.GetPrim());return obj
 def cyl(name,center,r,h,mat):
  obj=cylinder(stage,path+'/'+name,center,r,h,mats[mat]);parts.append(obj.GetPrim());return obj
 def ellipsoid(name,center,size,mat):
  obj=UsdGeom.Sphere.Define(stage,path+'/'+name);obj.CreateRadiusAttr(1.);obj.AddTranslateOp().Set(Gf.Vec3d(*center));obj.AddScaleOp().Set(Gf.Vec3f(*size));UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mats[mat]);parts.append(obj.GetPrim());return obj
 def carton(w,d,h,mat):
  box('CartonFloor',(0,0,.003),(w,d,.006),mat)
  box('Front',(0,-d/2+.002,h/2),(w,.004,h),mat);box('Back',(0,d/2-.002,h/2),(w,.004,h),mat)
  box('Left',(-w/2+.002,0,h/2),(.004,d-.008,h),mat);box('Right',(w/2-.002,0,h/2),(.004,d-.008,h),mat)
 if kind=='burger':
  # A packaged burger is visually distinct and gives the gripper one stable,
  # convex contact body instead of a rolling stack of round food layers.
  box('BurgerBox',(0,0,.025),(.068,.064,.050),'white')
  box('FrontBand',(0,-.0325,.025),(.046,.0015,.012),'red',False)
  box('LidMark',(0,0,.0508),(.034,.022,.0015),'bread',False)
 elif kind=='fries':
  carton(.050,.045,.043,'red')
  for i in range(3):
   for j in range(3):box('Fry%d%d'%(i,j),((i-1)*.011,(j-1)*.010,.041+(i+j)%3*.004),(.008,.008,.048),'gold')
  box('Stripe',(0,-.023,.027),(.030,.0015,.006),'white')
 elif kind=='nuggets':
  carton(.063,.059,.033,'white')
  for i,(x,y) in enumerate([(-.014,-.013),(.014,-.013),(-.014,.014),(.014,.014)]):ellipsoid('Nugget'+str(i),(x,y,.025),(.013,.012,.013),'gold')
  box('Band',(0,-.030,.016),(.045,.0015,.009),'red')
 elif kind=='pie':
  box('Sleeve',(0,0,.019),(.040,.084,.038),'brown');box('Window',(0,0,.0385),(.031,.067,.002),'bread')
  for i in range(5):box('CrustScore'+str(i),(0,-.026+i*.013,.040),(.024,.002,.002),'gold')
 elif kind=='tart':
  carton(.057,.057,.026,'red');cyl('Crust',(0,0,.024),.023,.026,'bread');cyl('Custard',(0,0,.038),.018,.004,'custard')
 else:raise ValueError(kind)
 for prim in parts+decorative:
  contact=prim in parts
  UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(dynamic and contact)
  if prim.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
 if dynamic:
  UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.08)
 return root
