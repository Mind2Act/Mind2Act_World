"""One source-qualified CP09 collision exception; never a general robot override API."""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
from mind2act_eval.contracts import to_json

ROOTS=('/World/RobotLeft/Asset','/World/RobotRight/Asset')
DECISION='benchmark maintainer 2026-11-01 裁决：保留采集指垫与几何，不得删除或改回通用'
SOURCE_DECISION_SHA256='d1bff678c2502eb4237b9d6974e9075ac46ac269c15e7e9d2b10390f1069e85c'
ENGLISH_DECISION=('Preserve the acquisition finger pads and geometry; '
    'do not remove them or replace them with generic geometry.')
SOURCE_ENGLISH_DECISION='Acquisition decision (2026-11-01): '+ENGLISH_DECISION


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def is_geometry_decision(value):
    """Only the two frozen labels and their exact English projections qualify."""
    return isinstance(value,str) and (value in (DECISION,ENGLISH_DECISION,SOURCE_ENGLISH_DECISION) or
        hashlib.sha256(value.encode()).hexdigest()==SOURCE_DECISION_SHA256)


def english_geometry_decision(value):
    """Project a qualified label without changing the collision authorization."""
    if value in (DECISION,ENGLISH_DECISION):return ENGLISH_DECISION
    if isinstance(value,str) and (value==SOURCE_ENGLISH_DECISION or
            hashlib.sha256(value.encode()).hexdigest()==SOURCE_DECISION_SHA256):
        return SOURCE_ENGLISH_DECISION
    raise ValueError('CP09 geometry decision has no exact English authority')


def canonical_geometry_decision(runtime_value, source_value):
    """Restore only a frozen source label paired with its exact English value.

    The caller retains the original source bytes/hash. This function authorizes
    one text field; it does not authorize any other spec or physical change.
    """
    if (not isinstance(source_value,str) or
            source_value!=DECISION and hashlib.sha256(source_value.encode()).hexdigest()!=SOURCE_DECISION_SHA256
            or runtime_value!=english_geometry_decision(source_value)):
        raise ValueError('CP09 English decision differs from its exact frozen source label')
    return source_value


def expected_descriptor(spec=None):
    """Select CP09 geometry only; scene validation authorizes the actual case."""
    if spec is not None and spec.task_id!='CP09':
        raise ValueError('CP09 acquisition pad geometry requires the CP09 task')
    from .cp09_pad_geometry import descriptor
    return descriptor()


def snapshot(stage):
    """All composed authored USD values, APIs and relationships in both X5 roots."""
    from pxr import Usd
    result={}
    for root in ROOTS:
        prim=stage.GetPrimAtPath(root)
        if not prim:raise ValueError('shared X5 root missing before collision overlay')
        for p in Usd.PrimRange(prim):
            attrs={}
            for a in p.GetAttributes():
                if a.HasAuthoredValueOpinion():
                    attrs[a.GetName()]=dict(type=str(a.GetTypeName()),value=repr(a.Get()),
                        samples=[(t,repr(a.Get(t))) for t in a.GetTimeSamples()])
            result[str(p.GetPath())]=dict(type=p.GetTypeName(),schemas=list(p.GetAppliedSchemas()),attrs=attrs,
                relationships={r.GetName():list(map(str,r.GetTargets())) for r in p.GetRelationships()})
    return result


def export_actual_usd(stage,path):
    from pxr import Sdf
    flat=stage.Flatten();layer=Sdf.Layer.CreateAnonymous('CP09-actual-robot.usda')
    Sdf.CreatePrimInLayer(layer,'/World')
    for root in ROOTS:
        Sdf.CreatePrimInLayer(layer,root.rsplit('/',1)[0])
        if not Sdf.CopySpec(flat,root,layer,root):raise ValueError('cannot audit actual composed X5 USD')
    for side in ('Left','Right'):
        for material in material_paths(side):
            if flat.GetPrimAtPath(material):
                Sdf.CreatePrimInLayer(layer,material.rsplit('/',1)[0]);Sdf.CopySpec(flat,material,layer,material)
    text=layer.ExportToString();Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def verify_official_parameters(stage):
    from pxr import Usd,UsdPhysics
    from . import PROFILE
    for root in ROOTS:
        prim=stage.GetPrimAtPath(root)
        joints={p.GetName():p for p in Usd.PrimRange(prim) if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)}
        for name in PROFILE['arm_joints']+PROFILE['finger_joints']:
            joint=joints.get(name)
            if joint is None:raise ValueError('CP09 overlay requires all official X5 joints')
            angular=name in PROFILE['arm_joints'];drive=UsdPhysics.DriveAPI(joint,'angular' if angular else 'linear')
            values=PROFILE['arm' if angular else 'finger']
            scale=math.pi/180 if angular else 1.
            for attr,expected in [(drive.GetStiffnessAttr(),values['stiffness']*scale),(drive.GetDampingAttr(),values['damping']*scale),
                    (drive.GetMaxForceAttr(),100.)]:
                if attr.Get() is None or not math.isclose(float(attr.Get()),expected,rel_tol=1e-7,abs_tol=1e-6):raise ValueError('CP09 overlay cannot authorize nonofficial X5 drives')
        for name,value in [('solverPositionIterationCount',PROFILE['articulation']['position_iterations']),
                           ('solverVelocityIterationCount',PROFILE['articulation']['velocity_iterations']),
                           ('enabledSelfCollisions',PROFILE['articulation']['self_collisions'])]:
            if prim.GetAttribute('physxArticulation:'+name).Get()!=value:
                raise ValueError('CP09 overlay cannot authorize nonofficial X5 solver')


def protected_values(stage):
    """Only fixed servo/solver/limit fields; actual rigid body poses may advance."""
    from pxr import Usd
    suffixes=('stiffness','damping','maxForce','lowerLimit','upperLimit','solverPositionIterationCount',
              'solverVelocityIterationCount','enabledSelfCollisions','maxJointVelocity','armature','disableGravity','maxDepenetrationVelocity')
    values={}
    for root in ROOTS:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for attr in prim.GetAttributes():
                if attr.GetName().endswith(suffixes):values[str(attr.GetPath())]=repr(attr.Get())
    return values


class SourceCollisionGuard:
    """Exact pre/post write scope, plus persistent pad and official parameter checks."""
    def __init__(self,stage,descriptor):
        verify_official_parameters(stage)
        self.descriptor=deepcopy(descriptor);self.before=snapshot(stage)
        self.protected=protected_values(stage)
        self._protected_handles={path:stage.GetAttributeAtPath(path) for path in self.protected}

    def authorize_after(self,stage):
        after=snapshot(stage);paths={p['path'] for p in self.descriptor['pads']}
        if set(after)-set(self.before) != paths-set(self.before) or set(self.before)-set(after):
            raise ValueError('CP09 overlay added or deleted an unauthorized robot prim')
        parents={p.rsplit('/',1)[0] for p in paths}
        for path,before in self.before.items():
            if path in paths:continue
            current=deepcopy(after[path]);original=deepcopy(before)
            if path in parents:
                # Source contact reports on these four moving finger bodies only.
                name='physxContactReport:threshold'
                if name not in original['attrs']:
                    attr=current['attrs'].pop(name,None)
                    if attr is not None and float(attr['value'])!=0:raise ValueError('CP09 report threshold differs')
                if 'PhysxContactReportAPI' not in original['schemas']:
                    current['schemas']=[s for s in current['schemas'] if s!='PhysxContactReportAPI']
            if current!=original:raise ValueError('CP09 overlay changed unauthorized robot USD: '+path)
        self.verify(stage)
        return after

    def verify(self,stage):
        validate_actual(stage,self.descriptor)
        if {path:repr(attr.Get()) for path,attr in self._protected_handles.items()}!=self.protected:
            raise ValueError('shared official X5 drive/solver/limits changed after CP09 overlay')


def material_paths(side):
    return '/World/CP09/GripMaterial'+side,'/World/Materials/CP09FingerRubber'+side


def validate_actual(stage,descriptor):
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade
    import numpy as np
    for row in descriptor['pads']:
        prim=stage.GetPrimAtPath(row['path'])
        if not prim or not prim.IsActive() or not prim.IsDefined() or not prim.IsA(UsdGeom.Cube) or list(prim.GetChildren()):raise ValueError('CP09 pad missing or extra descendant')
        if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible' or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
            raise ValueError('CP09 acquisition pad must remain visible and colliding')
        allowed={'size','xformOp:translate','xformOp:scale','xformOpOrder','physics:collisionEnabled',
                 'physxCollision:contactOffset','physxCollision:restOffset','material:binding','material:binding:physics'}
        if set(p.GetName() for p in prim.GetAuthoredProperties())-allowed:
            raise ValueError('CP09 pad has an unauthorized authored property')
        if set(prim.GetAppliedSchemas())-{'PhysicsCollisionAPI','PhysxCollisionAPI','MaterialBindingAPI'}:
            raise ValueError('CP09 pad has an unauthorized API schema')
        cube=UsdGeom.Cube(prim);ops=cube.GetOrderedXformOps()
        if (len(ops)!=2 or ops[0].GetOpType()!=UsdGeom.XformOp.TypeTranslate or ops[1].GetOpType()!=UsdGeom.XformOp.TypeScale
            or cube.GetSizeAttr().Get()!=1 or not np.allclose(ops[0].Get(),row['translate'],rtol=0,atol=1e-8)
            or not np.allclose(ops[1].Get(),row['scale'],rtol=0,atol=1e-8)):
            raise ValueError('CP09 pad geometry differs from acquisition source')
        collision=PhysxSchema.PhysxCollisionAPI(prim)
        if abs(collision.GetContactOffsetAttr().Get()-.0001)>1e-9 or collision.GetRestOffsetAttr().Get()!=0:
            raise ValueError('CP09 pad offsets differ')
        side='Left' if '/RobotLeft/' in row['path'] else 'Right';physics_path,visual_path=material_paths(side)
        for name,target in [('material:binding',visual_path),('material:binding:physics',physics_path)]:
            if list(map(str,prim.GetRelationship(name).GetTargets()))!=[target]:raise ValueError('CP09 pad source binding differs')
        material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')[0]
        if str(material.GetPath())!=physics_path:raise ValueError('CP09 pad physics binding differs')
        api=UsdPhysics.MaterialAPI(material.GetPrim())
        if (api.GetStaticFrictionAttr().Get()!=2 or abs(api.GetDynamicFrictionAttr().Get()-1.8)>1e-6 or api.GetRestitutionAttr().Get()!=0):
            raise ValueError('CP09 pad friction differs')
        visual=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if str(visual.GetPath())!=visual_path:raise ValueError('CP09 pad visual binding differs')
        shader=visual.ComputeSurfaceSource()[0]
        if not shader or str(shader.GetPath())!=visual_path+'/Shader' or shader.GetIdAttr().Get()!='UsdPreviewSurface':
            raise ValueError('CP09 pad visual shader differs')
        for key,value in [('diffuseColor',row['visual_color']),('metallic',0.),('roughness',.8)]:
            source_input=shader.GetInput(key)
            if not source_input or source_input.GetAttr().GetConnections() or not np.allclose(source_input.Get(),value,rtol=0,atol=1e-6):raise ValueError('CP09 pad visual material differs')


def apply(stage,spec,descriptor,output_dir,*,retained_source_pads=False):
    """Called exclusively by prepared, uninitialized X5Robot with its bound spec."""
    from pxr import Gf,UsdGeom,UsdShade,UsdPhysics,PhysxSchema,Sdf
    expected=expected_descriptor(spec)
    if digest(descriptor)!=digest(expected):raise ValueError('CP09 arbitrary collision descriptors are forbidden')
    if retained_source_pads:validate_actual(stage,expected)
    guard=SourceCollisionGuard(stage,expected)
    output=Path(output_dir)/'robot-collision-overlay';output.mkdir(parents=True,exist_ok=False)
    before_usd=export_actual_usd(stage,output/'before.usda')
    for side in ('Left','Right'):
        physics_path,visual_path=material_paths(side)
        # Adopt unchanged source visual material if the station retains it.
        # Never replace an existing material that differs from the source values.
        import numpy as np
        if stage.GetPrimAtPath(physics_path):
            physics=UsdShade.Material(stage.GetPrimAtPath(physics_path));api=UsdPhysics.MaterialAPI(physics.GetPrim())
            if (api.GetStaticFrictionAttr().Get()!=2 or abs(api.GetDynamicFrictionAttr().Get()-1.8)>1e-6
                    or api.GetRestitutionAttr().Get()!=0):raise ValueError('occupied CP09 physics material differs')
        else:
            from .cp09_source_pads import define_physics_material
            physics=define_physics_material(stage,physics_path)
        if stage.GetPrimAtPath(visual_path):
            visual=UsdShade.Material(stage.GetPrimAtPath(visual_path));shader=visual.ComputeSurfaceSource()[0]
            if not shader or shader.GetIdAttr().Get()!='UsdPreviewSurface':raise ValueError('occupied CP09 visual material differs')
            for key,value in [('diffuseColor',[.018,.023,.026]),('metallic',0.),('roughness',.8)]:
                if not np.allclose(shader.GetInput(key).Get(),value,rtol=0,atol=1e-6):raise ValueError('occupied CP09 visual material differs')
        else:
            visual=UsdShade.Material.Define(stage,visual_path);shader=UsdShade.Shader.Define(stage,visual_path+'/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.018,.023,.026))
            shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8)
            visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for row in expected['pads']:
            if '/Robot'+side+'/' not in row['path']:continue
            if retained_source_pads:continue  # Already strictly restored before appearance; never replace/rebind.
            if stage.GetPrimAtPath(row['path']):raise ValueError('CP09 pad unexpectedly survived official source restoration')
            pad=UsdGeom.Cube.Define(stage,row['path']);pad.CreateSizeAttr(1.)
            pad.AddTranslateOp().Set(Gf.Vec3d(*row['translate']));pad.AddScaleOp().Set(Gf.Vec3f(*row['scale']))
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim()).CreateCollisionEnabledAttr(True)
            collision=PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim());collision.CreateContactOffsetAttr(.0001);collision.CreateRestOffsetAttr(0.)
            bindings=UsdShade.MaterialBindingAPI.Apply(pad.GetPrim());bindings.Bind(visual);bindings.Bind(physics,materialPurpose='physics')
            parent=stage.GetPrimAtPath(row['path'].rsplit('/',1)[0])
            PhysxSchema.PhysxContactReportAPI.Apply(parent).CreateThresholdAttr(0.)
    after=guard.authorize_after(stage);after_usd=export_actual_usd(stage,output/'after.usda')
    receipt=dict(schema='mind2act.cp09-collision-overlay.v2',task_id=spec.task_id,instance_id=spec.instance_id,
        source_id=spec.source_id,source_sha256=expected['source_sha256'],descriptor_sha256=digest(expected),
        acquisition_version=expected['acquisition_version'],geometry_asset_sha256=expected['geometry_asset_sha256'],
        geometry_sha256=expected['geometry_sha256'],
        active_spec_sha256=digest(to_json(spec)),
        before_actual_usd_sha256=before_usd,after_actual_usd_sha256=after_usd,
        before_snapshot_sha256=digest(guard.before),after_snapshot_sha256=digest(after),
        added_robot_prims=sorted(set(after)-set(guard.before)),official_drive_solver_limits_unchanged=True)
    (output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return guard,receipt
