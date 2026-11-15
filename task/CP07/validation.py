"""Independent finite CP07 clean VAL with runtime case/stock/binding byte authority."""
import json
from mind2act_eval.contracts import InstanceSpec,to_json


def validate(spec):
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    c=spec.config
    if (spec.task_id!='CP07' or spec.source_id!='cp07-independent-clean-validation-v1'
            or spec.config_version!='cp07-source-binding-v2'
            or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30 or spec.clock.rgb_hz!=30
            or c.get('layout_split')!='validation' or c.get('layout_profile')!='clean'
            or c.get('validation_source')!='cp07-independent-clean-packer-v1'
            or c.get('cp07_mode')!='independent_validation'
            or 'train_source' in c or 'test_source' in c):
        raise ValueError('CP07 independent VAL full identity or mode differs')
    return c


def load(spec,assets):
    c=validate(spec)
    from mind2act_eval.sim.source_assets import verify_source_file,read_source_bytes
    verify_source_file(assets,'station',c['scene_sha256'])
    case=json.loads(read_source_bytes(assets,'validation_case',c['source_case_sha256']))
    binding=json.loads(read_source_bytes(assets,'cp07_binding',c['cp07_binding_sha256']))
    if (case.get('schema')!='mind2act.cp07.validation_runtime_case.v1'
            or case['task_id']!=spec.task_id or case['difficulty']!=spec.difficulty
            or case['instance_id']!=spec.instance_id
            or binding['source_spec_sha256']!=c['source_case_sha256']
            or binding['source_manifest_sha256']!=c['source_case_sha256']
            or binding['scene_sha256']!=c['scene_sha256'] or 'source_h5' in binding):
        raise ValueError('CP07 independent VAL runtime case or binding differs')
    from .binding import validate as validate_binding
    validate_binding(spec,binding)
    # Exact original NVIDIA glTF PBR module and its full real import closure.
    pbr=verify_source_file(assets,'validation_pbr',spec.config['validation_pbr_sha256'])
    from mind2act_eval.sim.mdl_dependencies import audit_mdl,_read
    import hashlib
    sha=lambda identifier:hashlib.sha256(_read(str(identifier))).hexdigest()
    audit=audit_mdl(str(pbr),[str(assets.resolve(k)) for k in spec.config['material_mdl_search_asset_ids']])
    if not audit['passed']:raise ValueError('validation PBR import closure unresolved')
    value=dict(root_sha256=sha(pbr),modules=sorted(sha(x) for x in audit['modules']),
        references=sorted((x['kind'],x['path'],sha(x['layer']),sha(x['resolved'])) for x in audit['references']))
    digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=spec.config['validation_pbr_closure_sha256']:
        raise ValueError('validation PBR real import/texture bytes differ')
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(str(assets.resolve('station')));cache=UsdGeom.XformCache();stock=[]
    for piece in binding['pieces']:
        prim=stage.GetPrimAtPath(piece['path'])
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('CP07 validation stock rigid body missing')
        matrix=cache.GetLocalToWorldTransform(prim);mass=UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        stock.append(dict(geometry=list(piece['geometry_signature'])+[float(mass) if mass is not None else None],
            position=[round(float(x),8) for x in matrix.ExtractTranslation()],
            orientation=[[round(float(matrix[i][j]),8) for j in range(3)] for i in range(3)]))
    stock.sort(key=lambda x:json.dumps(x,sort_keys=True))
    if stock!=case['stock']:raise ValueError('CP07 validation actual initial stock shape/mass/poses differ')
    return binding


def validate_evidence(evidence):
    if type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError('CP07 independent VAL requires a nonnegative physics tick')
    v=to_json(evidence.values);spec=InstanceSpec.from_dict(v['validation_source_spec']);validate(spec)
    if (spec.task_id!=evidence.task_id or spec.instance_id!=evidence.instance_id
            or spec.difficulty!=v['difficulty'] or v.get('binding_mode')!='independent_validation'
            or v.get('solutions_scope')!='known-source-solutions'
            or any(k in v for k in ['train_source','test_source','source_layout_sha256','training_source_spec'])):
        raise ValueError('CP07 independent VAL scoring identity or mode differs')
    return spec
