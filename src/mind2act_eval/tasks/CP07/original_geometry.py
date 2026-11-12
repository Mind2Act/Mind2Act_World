"""Verify original-world geometry, retained control pool and narrow composition.

Pure USD shape functions are static reviewed copies of the original author
functions. Robot, World and simulation modules are never imported or executed.
No receipt boolean or caller-selected hash is authority.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from mind2act_eval.original_asset_graph import layer_bytes, verify_graph
from mind2act_eval.original_test_inputs import source_identity
from mind2act_eval.verified_files import verified_bytes
from .original_inputs import WORLD_SHA256

SHAPE_SHA256 = '28304130ee76aac9dcb30e31bb09d4068c8994a6262b3897512e3ededb0aaa95'
AUTHORS = {'world.py': WORLD_SHA256,
    'parameters.py': '37b3c3208ad28377632e725fcc8e74a80f7b253812890a689a722fd4379d9575',
    'train_v2.py': '6f7d68e6c9221163debc6d8944e24e69c36a9965d2263c125467e126635a0083'}
CAR_SHA256 = 'bc6395295a55a45de3c214e8fa3646668eed9667722925e042b47b61f94504b2'
PINE_SHA256 = '67bbb2ddc44c4a096ac71918cb3adbddf157dfdbeedbe8ba19ad29dd6198fca0'
ROOTS = ['/World/Terrain', '/World/Pieces', '/World/Car', '/World/StartButton',
         '/World/ButtonJoint', '/World/HardLooks']
BASE_SHA256 = {
    'easy': 'bf040b8a33f387fb9ff00332500cdba2546c5f7c57115c1097ba04e62f3fdd52',
    'medium': '2e452f0154e721cc4a56c3334a035753aaac75c5ddb8602f4bc2e337afc8e8bc',
    'hard': '1b7fa24ad34b38f84d949046008533dfdba7417c43d6c7b9759c320e0403e0c7'}
# These bind the retained control pools, including all file dependencies, and
# the static original-world opinions. They do not enumerate150 case scenes.
BASE_GRAPH_SHA256 = {
    'easy': '267a7217ab8a82b48a0a8249b026721459070ac203f6c05add7f0f1b37e817ae',
    'medium': '5fcfeeb896c0ad3da842c3450af835f23878d81bdd061f6662688113bc57a701',
    'hard': 'f4ac4863b17d1caef577f30302f05971d44f8ed7b0224a91f07a819ae87c908f'}
STATIC_WORLD_SHA256 = {
    'easy': '7c202439ea0a63f750a413aa408716af82411a1ca1a208defd5855b50247c468',
    'medium': 'd79b030811fb3b951017cdaf5eafc4f0c6e9ae692d9521a61d49499611fe8cca',
    'hard': '7d9059df123e82c69a664a0078465fbaf598a3dba6e24eff0f38cf24a799fc1e'}


@lru_cache(maxsize=16)
def _retained_paths(path, code_owned_sha):
    """Freeze only path names after the caller verifies the full pinned graph.

    Reopening the entire robot stage for every stock layout is unnecessary;
    file and shader closure freshness is still verified on every load.
    """
    from pxr import Usd
    stage = Usd.Stage.Open(path)
    return tuple((name, tuple(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(name))))
        for name in ROOTS if stage.GetPrimAtPath(name))


def _opinions(layer, prefix=None):
    result = {}
    def visit(path):
        if prefix is None or path.HasPrefix(prefix):
            item = layer.GetObjectAtPath(path)
            if item is not None:
                result[str(path)] = {k: (
                    [(offset.offset, offset.scale) for offset in layer.subLayerOffsets]
                    if k == 'subLayerOffsets' else item.GetInfo(k)
                ) for k in item.ListInfoKeys()}
    layer.Traverse('/', visit)
    return result


def static_world_digest(layer, path, root, raw):
    """All static author opinions, with only source-qualified rest positions."""
    from pxr import Sdf
    thin = next(p for p in raw['stock'] if p['name'] == 'SupportThin')
    def normalize(value):
        if isinstance(value, Sdf.AssetPath):
            p = (Path(path).parent/value.path).resolve(strict=True)
            if not p.is_relative_to(root):
                raise ValueError('CP07 original texture escapes asset root')
            return ['asset-sha256', hashlib.sha256(p.read_bytes()).hexdigest()]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if hasattr(value, 'isExplicit'):
            return {k: normalize(getattr(value, k)) for k in
                ('isExplicit', 'explicitItems', 'prependedItems', 'appendedItems',
                 'deletedItems', 'orderedItems')}
        try:
            return [normalize(v) for v in value]
        except TypeError:
            return str(value)
    data = _opinions(layer)
    for key in list(data):
        if Sdf.Path(key).HasPrefix('/World/Pieces'):
            del data[key]
    for i, dx in enumerate((-.032, .032)):
        key = '/World/Terrain/ThinStockRest'+str(i)+'.xformOp:translate'
        expected = [thin['start'][0]+dx, thin['start'][1], .773]
        if list(data[key]['default']) != expected:
            raise ValueError('CP07 original stock support rest differs')
        data[key]['default'] = 'authenticated-original-thin-rest'
    return hashlib.sha256(json.dumps(normalize(data), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _piece_layer(raw):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    import numpy as np
    from scipy.spatial import ConvexHull
    stage = Usd.Stage.CreateInMemory()
    # Exact attributes/types authored by the original source's four PhysX APIs.
    # The verifier writes only an expected in-memory layer; no global modules or
    # simulator API are installed or replaced.
    def api(name, prefix, fields):
        class API:
            @staticmethod
            def Apply(prim):
                prim.AddAppliedSchema(name)
                return SimpleNamespace(**{
                    'Create'+method+'Attr': (lambda value, p=prim, n=field, t=kind:
                        p.CreateAttribute(prefix+':'+n, t, False).Set(value))
                    for method, field, kind in fields})
        return API
    physx = SimpleNamespace(
        PhysxCollisionAPI=api('PhysxCollisionAPI', 'physxCollision', [
            ('ContactOffset', 'contactOffset', Sdf.ValueTypeNames.Float),
            ('RestOffset', 'restOffset', Sdf.ValueTypeNames.Float)]),
        PhysxContactReportAPI=api('PhysxContactReportAPI', 'physxContactReport', [
            ('Threshold', 'threshold', Sdf.ValueTypeNames.Float)]),
        PhysxRigidBodyAPI=api('PhysxRigidBodyAPI', 'physxRigidBody', [
            ('SolverPositionIterationCount', 'solverPositionIterationCount', Sdf.ValueTypeNames.Int),
            ('SolverVelocityIterationCount', 'solverVelocityIterationCount', Sdf.ValueTypeNames.Int),
            ('LinearDamping', 'linearDamping', Sdf.ValueTypeNames.Float),
            ('AngularDamping', 'angularDamping', Sdf.ValueTypeNames.Float)]))
    # Reviewed static copies of original pure USD functions below. Their source
    # identities are SHAPE_SHA256 and WORLD_SHA256; no source module is loaded.
    PhysxSchema = physx
    pieces = []
    def xf(stage,path,pos=(0,0,0),rot=(0,0,0)):
      q=UsdGeom.Xform.Define(stage,path);q.AddTranslateOp().Set(Gf.Vec3d(*pos))
      if any(rot):q.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
      return q

    def box(stage,path,size,pos,mat,bevel=0,rot=(0,0,0)):
      if bevel:
       h=np.array(size)/2;b=min(bevel,min(h)*.65);verts=[]
       for sx in [-1,1]:
        for sy in [-1,1]:
         for sz in [-1,1]:
          sign=np.array([sx,sy,sz])
          for k in range(3):
           v=h-b;v[k]=h[k];verts.append(v*sign)
       verts=np.array(verts);hull=ConvexHull(verts);faces=[];uv=[]
       for tri in hull.simplices:
        vv=verts[tri];normal=np.cross(vv[1]-vv[0],vv[2]-vv[0])
        if np.dot(normal,vv.mean(axis=0))<0:tri=tri[::-1];vv=verts[tri];normal=-normal
        faces.extend(tri.tolist());axis=int(np.argmax(np.abs(normal)));axes=[k for k in range(3) if k!=axis]
        for v in vv:uv.append(((v[axes[0]]/size[axes[0]])+.5,(v[axes[1]]/size[axes[1]])+.5))
       obj=UsdGeom.Mesh.Define(stage,path);obj.CreatePointsAttr([Gf.Vec3f(*v) for v in verts]);obj.CreateFaceVertexCountsAttr([3]*(len(faces)//3));obj.CreateFaceVertexIndicesAttr(faces);obj.CreateSubdivisionSchemeAttr('none')
       UsdGeom.PrimvarsAPI(obj).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying).Set(uv)
      else:
       obj=UsdGeom.Cube.Define(stage,path);obj.CreateSizeAttr(1)
      obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
      if not bevel:obj.AddScaleOp().Set(Gf.Vec3f(*size))
      if any(rot):obj.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
      bind(obj.GetPrim(),mat);return obj

    def collision(prim,hidden=False):
      UsdPhysics.CollisionAPI.Apply(prim);co=PhysxSchema.PhysxCollisionAPI.Apply(prim);co.CreateContactOffsetAttr(.0007);co.CreateRestOffsetAttr(0)
      if hidden:UsdGeom.Imageable(prim).MakeInvisible()

    def rigid(prim,mass):
      UsdPhysics.RigidBodyAPI.Apply(prim);UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    def pm(prim,m):UsdShade.MaterialBindingAPI.Apply(prim).Bind(m,materialPurpose='physics')

    def bind(prim,m):UsdShade.MaterialBindingAPI.Apply(prim).Bind(m)

    def add_piece(name,L,W,T,start,dest,yaw=0,pitch=0,kind='bar',height0=0,height1=0):
      path='/World/Pieces/'+name;node=xf(stage,path,start);rigid(node.GetPrim(),L*W*T*580 if kind in ['bar','tapered','step','L'] else L*W*(height0+height1)/2*580)
      if kind=='bar':
       v=box(stage,path+'/Visual',(L,W,T),(0,0,0),wood,.0005);c=box(stage,path+'/Collision',(L,W,T),(0,0,0),wood);collision(c.GetPrim(),True);pm(c.GetPrim(),fm)
      elif kind=='tapered':
       points=[(-L/2,-W*.45,-T/2),(-L/2,W*.45,-T/2),(L/2,W*.55,-T/2),(L/2,-W*.55,-T/2),(-L/2,-W*.45,T/2),(-L/2,W*.45,T/2),(L/2,W*.55,T/2),(L/2,-W*.55,T/2)]
       faces=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
       mesh=UsdGeom.Mesh.Define(stage,path+'/Tapered');mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr([4]*6);mesh.CreateFaceVertexIndicesAttr(sum(faces,[]));mesh.CreateSubdivisionSchemeAttr('none');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(wood);collision(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull');pm(mesh.GetPrim(),fm)
      elif kind in ['step','L']:
       sizes=[(L,W,T/2),(.45*L,W,T/2)] if kind=='step' else [(L,.45*W,T),(.35*L,.55*W,T)]
       poses=[(0,0,-T/4),(-.275*L,0,T/4)] if kind=='step' else [(0,-.275*W,0),(-.325*L,.225*W,0)]
       for c,(sz,ps) in enumerate(zip(sizes,poses)):
        o=box(stage,path+f'/Compound{c}',sz,ps,wood);collision(o.GetPrim());pm(o.GetPrim(),fm)
      else:
       # True convex trapezoidal wedge; flat bottom, slope +x, local origin at bottom.
       points=[(-L/2,-W/2,0),(-L/2,W/2,0),(L/2,-W/2,0),(L/2,W/2,0),(-L/2,-W/2,height0),(-L/2,W/2,height0),(L/2,-W/2,height1),(L/2,W/2,height1)]
       faces=[[0,2,3,1],[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[4,5,7,6]]
       m=UsdGeom.Mesh.Define(stage,path+'/Wedge');m.CreatePointsAttr(points);m.CreateFaceVertexCountsAttr([4]*6);m.CreateFaceVertexIndicesAttr(sum(faces,[]));m.CreateSubdivisionSchemeAttr('none');m.CreateDoubleSidedAttr(True);UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats['wedge']);collision(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull');pm(m.GetPrim(),fm)
      PhysxSchema.PhysxContactReportAPI.Apply(node.GetPrim()).CreateThresholdAttr(0)
      rb=PhysxSchema.PhysxRigidBodyAPI.Apply(node.GetPrim());rb.CreateSolverPositionIterationCountAttr(64);rb.CreateSolverVelocityIterationCountAttr(8);rb.CreateLinearDampingAttr(.08);rb.CreateAngularDampingAttr(.3)
      piece=dict(name=name,L=L,W=W,T=T,start=list(start),dest=list(dest),yaw=yaw,pitch=pitch,kind=kind,height0=height0,height1=height1,path=path)
      pieces.append(piece);return node
    wood = UsdShade.Material.Define(stage, '/World/HardLooks/GrainedPine')
    fm = UsdShade.Material.Define(stage, '/World/HardLooks/WoodFriction')
    mats = {'wedge': UsdShade.Material.Define(stage, '/World/HardLooks/wedge')}
    for piece in raw['stock']:
        add_piece(**{k: v for k, v in piece.items() if k != 'path'})
    if pieces != raw['stock']:
        raise ValueError('CP07 original piece author parameters differ')
    return stage.Flatten()


def validate_geometry(spec, assets, binding, raw):
    from pxr import Usd, Sdf
    root = Path(assets.root).resolve(strict=True)
    def contained(path):
        p = Path(path).resolve(strict=True)
        if not p.is_relative_to(root):
            raise ValueError('CP07 original geometry escapes asset root')
        return p
    def descriptor(value, pin=None):
        relative = Path(value['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('CP07 original geometry requires contained relative path')
        p = contained(root/relative)
        verified_bytes(p, pin or value['sha256'])
        if pin is not None and value['sha256'] != pin:
            raise ValueError('CP07 original geometry code-owned pin differs')
        return p
    proof = json.loads(verified_bytes(assets.resolve('source_geometry_proof'),
        spec.config['source_geometry_proof_sha256'], max_bytes=1024**2))
    if (proof.get('schema') != 'mind2act.cp07_original8_geometry.v1'
            or proof.get('task_id') != 'CP07' or proof.get('logical_slot') != spec.instance_id
            or proof.get('source_configuration_sha256') != source_identity(spec)['source_configuration_sha256']
            or proof.get('source_author_sha256') != AUTHORS
            or proof.get('shape_helper_sha256') != SHAPE_SHA256
            or proof.get('original_toy_car_sha256') != CAR_SHA256
            or proof.get('original_pine_texture_sha256') != PINE_SHA256
            or proof.get('copied_roots') != ROOTS
            or proof.get('original_native_scene_bytes_claimed') is not False):
        raise ValueError('CP07 original geometry proof authority differs')
    scene = descriptor(proof['compiled_scene'])
    task = descriptor(proof['actual_composed_task'])
    base = descriptor(proof['common_station'], BASE_SHA256[spec.difficulty])
    if (scene != contained(assets.resolve('station'))
            or task != contained(assets.resolve('task_source'))
            or proof['compiled_scene']['sha256'] != binding['scene_sha256']):
        raise ValueError('CP07 original actual geometry role differs')
    verify_graph(base, root, BASE_GRAPH_SHA256[spec.difficulty])
    _, task_layer = layer_bytes(task)
    if static_world_digest(task_layer, task, root, raw) != STATIC_WORLD_SHA256[spec.difficulty]:
        raise ValueError('CP07 original static world geometry differs')
    if _opinions(task_layer, Sdf.Path('/World/Pieces')) != _opinions(_piece_layer(raw), Sdf.Path('/World/Pieces')):
        raise ValueError('CP07 original actual stock shape or physics differs')
    _, layer = layer_bytes(scene)
    if len(layer.subLayerPaths) != 2 or [contained(scene.parent/p) for p in layer.subLayerPaths] != [task, base]:
        raise ValueError('CP07 original composition requires exact task and retained pool')
    expected = Sdf.Layer.CreateAnonymous('expected-cp07.usda')
    expected.defaultPrim = 'World'
    expected.startTimeCode = 0
    expected.endTimeCode = 1000000
    expected.timeCodesPerSecond = 60
    expected.pseudoRoot.SetInfo('metersPerUnit', 1.)
    expected.pseudoRoot.SetInfo('upAxis', 'Z')
    expected.subLayerPaths = list(layer.subLayerPaths)
    # Only retire legacy prims absent from the already authenticated task tree.
    for name, paths in _retained_paths(str(base), BASE_SHA256[spec.difficulty]):
        retired = []
        for value in paths:
            path = Sdf.Path(value)
            if any(path.HasPrefix(p) for p in retired):
                continue
            # The authenticated task is a complete flattened author layer.
            if not task_layer.GetPrimAtPath(path):
                Sdf.CreatePrimInLayer(expected, path).active = False
                retired.append(path)
    if _opinions(layer) != _opinions(expected):
        raise ValueError('CP07 original scene has extra geometry, physics, units or retiming opinions')
