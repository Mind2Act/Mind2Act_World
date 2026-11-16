"""Finite independent VAL case and authored balance resources; no collector alias."""
import json,math
from mind2act_eval.contracts import InstanceSpec,to_json


def validate(spec):
    from mind2act_eval.public_layouts import validate as validate_public
    from .instances import public_parameters,instruction,INSTRUCTION_SHA,FRAMEWORK,signature
    validate_public(spec)
    c=spec.config;core=c['source_task']
    if (spec.task_id!='CP06' or spec.source_id!='cp06-independent-clean-validation-v1'
            or spec.config_version!='cp06-original-resource-projection-v1'
            or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30 or spec.clock.rgb_hz!=30
            or c.get('layout_split')!='validation' or c.get('layout_profile')!='clean'
            or c.get('validation_source')!='cp06-independent-clean-rack-v1'
            or 'train_source' in c or 'test_source' in c
            or core['difficulty']!=spec.difficulty or core['framework_commit']!=FRAMEWORK
            or core['physics_hz']!=120 or c['declared_task_sha256']!=signature(core)):
        raise ValueError('CP06 independent VAL complete identity or condition differs')
    if (to_json(c['public_parameters'])!=public_parameters(core)
            or c['instruction'] not in (instruction(spec.difficulty),instruction(spec.difficulty,'zh'))
            or c['instruction_source_sha256']!=INSTRUCTION_SHA
            or list(c['robot_rest'])!=[math.radians(v) for v in (0,65,100,-65,0,0)]+[.044,.044]):
        raise ValueError('CP06 independent VAL public parameters, rest or instruction differs')
    return dict(station_robots=to_json(c['station_robots']))


def validate_assets(spec,assets,*,verify_assets=False):
    row=validate(spec)
    from mind2act_eval.sim.source_assets import source_file,asset_bytes
    source_file(assets, 'station')
    raw=json.loads(asset_bytes(assets, 'validation_case'))
    expected=dict(schema='mind2act.cp06.validation_runtime_case.v1',task_id='CP06',difficulty=spec.difficulty,
        instance_id=spec.instance_id,source_task=to_json(spec.config['source_task']),station_robots=row['station_robots'])
    if raw!=expected:raise ValueError('CP06 independent VAL source document differs')
    # Resolve runtime shader dependencies. Export-byte comparison is an
    # optional offline diagnostic, not a prerequisite for loading the scene.
    pbr=source_file(assets, 'validation_pbr')
    from mind2act_eval.sim.mdl_dependencies import audit_mdl
    audit=audit_mdl(str(pbr),[str(assets.resolve(k)) for k in spec.config['material_mdl_search_asset_ids']])
    if not audit['passed']:raise ValueError('validation PBR import closure unresolved')
    if verify_assets:
        import hashlib
        from mind2act_eval.sim.mdl_dependencies import _read
        sha=lambda identifier:hashlib.sha256(_read(str(identifier))).hexdigest()
        value=dict(root_sha256=sha(pbr),modules=sorted(sha(x) for x in audit['modules']),
            references=sorted((x['kind'],x['path'],sha(x['layer']),sha(x['resolved'])) for x in audit['references']))
        digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=spec.config['validation_pbr_closure_sha256']:
            raise ValueError('validation PBR real import/texture bytes differ')
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(str(assets.resolve('station')));cache=UsdGeom.XformCache()
    for obj in raw['source_task']['all_objects']:
        prim=stage.GetPrimAtPath(obj['path'])
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('CP06 validation rigid body missing')
        position=cache.GetLocalToWorldTransform(prim).ExtractTranslation();mass=UsdPhysics.MassAPI(prim)
        if (max(abs(position[i]-obj['initial_position_m'][i]) for i in range(3))>1e-7
                or abs(mass.GetMassAttr().Get()-obj['grams']/1000)>1e-7
                or max(abs(v-w) for v,w in zip(mass.GetCenterOfMassAttr().Get(),obj['center_of_mass_m']))>1e-7
                or max(abs(v-w) for v,w in zip(mass.GetDiagonalInertiaAttr().Get(),obj['diagonal_inertia_kg_m2']))>1e-7):
            raise ValueError('CP06 validation actual pose/mass/COM/inertia differs')
    return row


def validate_evidence(evidence):
    v=to_json(evidence.values)
    spec=InstanceSpec.from_dict(v['validation_source_spec']);validate(spec)
    if (spec.task_id!=evidence.task_id or spec.instance_id!=evidence.instance_id
            or spec.difficulty!=v['difficulty'] or 'training_source_spec' in v or 'test_source' in v
            or v['source_spec_sha256']!=spec.config['source_spec_sha256']
            or v['source_task_sha256']!=spec.config['declared_task_sha256']
            or v['source_task']!=to_json(spec.config['source_task'])):
        raise ValueError('CP06 validation scoring identity/source/condition differs')
    return spec
