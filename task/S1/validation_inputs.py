"""Validation input boundary and optional offline release verification."""
import hashlib,json
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def catalogue():
    raw=Path(__file__).with_suffix('.json').read_bytes()
    data=json.loads(raw)
    if data['schema']!='mind2act.clean_validation_assets.v1' or data['task_id']!='S1' or len(data['records'])!=30:
        raise ValueError('clean VAL inventory differs')
    return data

@lru_cache(maxsize=2048)
def file_sha(path,size,mtime,ctime):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(spec,assets):
    from mind2act_eval.public_layouts import validate as public_validate
    from mind2act_eval.spec_identity import spec_digest
    public_validate(spec)
    if spec.config.get('clean_validation_authority')!='next-clean-v1' or spec.config.get('layout_split')!='validation':
        raise ValueError('independent VAL identity differs')
    data=catalogue()
    matches=[r for r in data['records'] if (r['task_id'],r['difficulty'],r['instance_id'])==(spec.task_id,spec.difficulty,spec.instance_id)]
    if len(matches)!=1 or matches[0]['spec_sha256']!=spec_digest(spec):
        raise ValueError('finite VAL spec differs')
    for row in data['files']:
        p=(assets.root/row['path']).resolve(strict=True)
        if not p.is_relative_to(assets.root):raise ValueError('VAL dependency escapes root')
        s=p.stat()
        if s.st_size!=row['size'] or file_sha(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=row['sha256']:
            raise ValueError('VAL dependency bytes differ: '+row['path'])
    return matches[0]


def validate(spec, assets):
    """Check input kind; the scene reader loads its required assets and conditions."""
    from mind2act_eval.public_layouts import validate as validate_layout
    validate_layout(spec)
    if spec.task_id != 'S1' or spec.config.get('layout_split') != 'validation':
        raise ValueError('expected a validation layout for ' + 'S1')
