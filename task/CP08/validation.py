"""Finite independent CP08 VAL runtime cases, never train/test wrapper aliases."""
import json
from mind2act_eval.contracts import InstanceSpec,to_json


def validate(spec):
    from mind2act_eval.public_layouts import validate as validate_public
    validate_public(spec)
    c=spec.config
    if (spec.task_id!='CP08' or spec.source_id!='cp08-independent-clean-validation-v1'
            or spec.config_version!='cp08-source-projection-v1'
            or c.get('layout_split')!='validation' or c.get('layout_profile')!='clean'
            or c.get('validation_source')!='cp08-independent-authored-t0-v1'
            or 'train_source' in c or 'test_source' in c
            or spec.clock.physics_hz!=120 or spec.clock.control_hz!=30 or spec.clock.rgb_hz!=30):
        raise ValueError('CP08 requires the complete admitted independent VAL source')
    from .projection import validate_validation
    from .instances import instructions
    validate_validation(to_json(c['source_projection']),spec.difficulty,to_json(c['source_case']))
    if c['instruction'] not in (instructions(spec.difficulty)['en'],instructions(spec.difficulty)['zh']):
        raise ValueError('CP08 validation instruction differs')
    return dict(scene_sha256=c['scene_sha256'],cue_texture_sha256=to_json(c['cue_texture_sha256']))


def validate_assets(spec,assets):
    row=validate(spec)
    from mind2act_eval.sim.source_assets import verify_source_file,read_source_bytes
    verify_source_file(assets,'station',row['scene_sha256'])
    for role,name in [('cue_plan','plan.png'),('cue_idle','screen-idle.png')]:
        verify_source_file(assets,role,row['cue_texture_sha256'][name])
    source=json.loads(read_source_bytes(assets,'validation_case',spec.config['source_spec_sha256']))
    expected=dict(schema='mind2act.cp08.validation_runtime_case.v1',task_id='CP08',
        difficulty=spec.difficulty,instance_id=spec.instance_id,
        source_case=to_json(spec.config['source_case']),source_projection=to_json(spec.config['source_projection']))
    if source!=expected:
        raise ValueError('CP08 validation runtime case differs from complete spec')
    # Authenticate the actual body poses and shape consumed at build.
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(str(assets.resolve('station')))
    cache=UsdGeom.XformCache();bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    for body in source['source_case']['blocks']+source['source_case']['distractors']:
        prim=stage.GetPrimAtPath('/World/CP11/Blocks/'+body['id'])
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('CP08 validation body missing')
        p=cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        bounds=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if (max(abs(p[i]-body['initial'][i]) for i in range(3))>1e-12
                or max(abs(bounds.GetMax()[i]-bounds.GetMin()[i]-body['size'][i]) for i in range(3))>1e-6):
            raise ValueError('CP08 validation authored pose or shape differs')
    return row


def validate_evidence(evidence):
    values=to_json(evidence.values)
    if 'validation_source_spec' not in values:
        raise ValueError('CP08 validation evidence requires its complete producer spec')
    spec=InstanceSpec.from_dict(values['validation_source_spec'])
    validate(spec)
    if (spec.task_id!=evidence.task_id or spec.instance_id!=evidence.instance_id
            or values['source_case']!=to_json(spec.config['source_case'])
            or values['source_projection']!=to_json(spec.config['source_projection'])
            or 'training_source_spec' in values or 'test_source' in values):
        raise ValueError('CP08 validation scoring identity or condition differs')
    return spec
