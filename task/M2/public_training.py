"""Code-owned runtime bytes for the selected M2 training subset.

Original collector inputs and restoration tokens stay private. Full-spec,
binding and closed asset checks retain the already validated runtime state.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from mind2act_eval.verified_files import verified_bytes, verify_sha256

@lru_cache(maxsize=1)
def _bytes():
    path=Path(__file__).with_name('config')/'public_training_inputs.json'
    return verified_bytes(path,'ae7084443790ee5c5c7a35efd54063b0ee2f79e57d04c05d7af4a95c94e8b378')

def validate_runtime(spec,assets):
    from mind2act_eval.public_layouts import validate
    from mind2act_eval.spec_identity import spec_digest
    from mind2act_eval.original_asset_graph import _shader_members
    admitted=validate(spec)
    data=json.loads(_bytes())
    if data['schema']!='mind2act.m2_public_runtime.v1' or len(data['records'])!=60:
        raise ValueError('M2 public runtime authority differs')
    row=data['records'].get(spec.difficulty+'/'+spec.instance_id)
    if (spec.task_id!='M2' or admitted['split']!='train'
            or spec.config.get('m2_public_runtime_only') is not True
            or spec.config.get('m2_mode')!='diagnostic_training'
            or row is None or row['spec_sha256']!=spec_digest(spec)):
        raise ValueError('M2 public runtime identity differs')
    root=Path(assets.root).resolve(strict=True)
    def contained(name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('M2 public runtime path is not contained')
        path=root/relative
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('M2 public runtime resource escapes root')
        return path
    for role,prefix in [('station','station'),('m2_binding','binding')]:
        path=contained(row[prefix+'_path'])
        if assets.resolve(role)!=path:
            raise ValueError('M2 public runtime role differs')
        verify_sha256(path,row[prefix+'_sha256'])
    current=tuple('assets/'+name for name in _shader_members(root/'assets'))
    if current!=tuple(data['mdl_members']):
        raise ValueError('M2 public MDL inventory differs')
    for member in data['asset_files']:
        verify_sha256(contained(member['path']),member['sha256'])
