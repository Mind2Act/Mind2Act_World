import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim import source_assets


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def catalogue(monkeypatch, row, closure):
    row = dict(row)
    restore = {key:row.pop(key) for key in ('json','usd') if key in row}
    if restore:
        restore.update(schema='mind2act_eval.source_projection_restoration.v1',
                       source_sha256=row['source_sha256'], runtime_sha256=row['runtime_sha256'],
                       identity=row.get('identity'))
        raw = (json.dumps(restore,sort_keys=True)+'\n').encode()
        temporary = tempfile.TemporaryDirectory(prefix='source-proofs-')
        monkeypatch._source_proof_directory = temporary
        root = Path(temporary.name)
        (root/(digest(raw)+'.json')).write_bytes(raw)
        monkeypatch.setenv('MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT', str(root))
        row['proof_sha256'] = digest(raw)
        row['restoration_kind'] = 'json' if 'json' in restore else 'usd'
    value = dict(approvals=[dict(row, closure_id='one')], closures={'one': closure})
    monkeypatch.setattr(source_assets, '_catalogue', lambda: value)


def test_original_bytes_remain_valid_without_an_approval(tmp_path, monkeypatch):
    path = tmp_path/'source.json'
    path.write_bytes(b'{"source":1}\n')
    monkeypatch.setattr(source_assets, '_catalogue', lambda: dict(approvals=[]))
    assets = LocalAssets(tmp_path, {'source': path.name})
    expected = digest(path.read_bytes())
    assert source_assets.read_source_bytes(assets, 'source', expected) == path.read_bytes()
    assert source_assets.verify_source_file(assets, 'source', expected) == path
    with pytest.raises(ValueError, match='code-approved'):
        source_assets.verify_source_file(assets, 'source', '0'*64)


def test_json_recovers_exact_original_bytes_and_rejects_mutations(tmp_path, monkeypatch):
    root = tmp_path/'fixture-release'
    root.mkdir()
    original = (json.dumps({'path': '/source/location', 'mass': 2}, indent=2)+'\n').encode()
    runtime = (json.dumps({'path': 'portable/location', 'mass': 2}, indent=2)+'\n').encode()
    path = root/'source.json'
    path.write_bytes(runtime)
    row = dict(asset_name=root.name, runtime_path=path.name, source_sha256=digest(original),
               runtime_sha256=digest(runtime), json=dict(
                   undo=[dict(pointer='/path', source='/source/location', runtime='portable/location')],
                   serialization=dict(indent=2, trailing_newline=True)))
    catalogue(monkeypatch, row, [dict(path=path.name, sha256=digest(runtime))])
    assets = LocalAssets(root, {'source': path.name})
    assert source_assets.read_source_bytes(assets, 'source', digest(original)) == original
    path.write_bytes(runtime.replace(b'"mass": 2', b'"mass": 3'))
    with pytest.raises(ValueError, match='code-approved'):
        source_assets.read_source_bytes(assets, 'source', digest(original))


def test_external_restoration_proof_cannot_authorize_changed_content(tmp_path, monkeypatch):
    root=tmp_path/'fixture-release';root.mkdir()
    path=root/'source.json';path.write_text('{"path":"portable"}')
    original=b'{"path":"original"}'
    row=dict(asset_name=root.name,runtime_path=path.name,source_sha256=digest(original),
             runtime_sha256=digest(path.read_bytes()),json=dict(
                 undo=[dict(pointer='/path',source='original',runtime='portable')],
                 serialization=dict(separators=[',',':'],trailing_newline=False)))
    catalogue(monkeypatch,row,[dict(path=path.name,sha256=row['runtime_sha256'])])
    assets=LocalAssets(root,{'source':path.name})
    assert source_assets.read_source_bytes(assets,'source',digest(original))==original
    proof_root=Path(__import__('os').environ['MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT'])
    next(proof_root.glob('*.json')).write_text('{"invented":"authority"}')
    with pytest.raises(ValueError,match='proof SHA differs'):
        source_assets.read_source_bytes(assets,'source',digest(original))


def test_approval_verifies_every_declared_dependency(tmp_path, monkeypatch):
    root = tmp_path/'fixture-release'
    root.mkdir()
    scene = root/'scene.usda'
    scene.write_bytes(b'portable scene')
    dependency = root/'image.png'
    dependency.write_bytes(b'accepted pixels')
    expected = digest(b'original scene')
    row = dict(asset_name=root.name, runtime_path=scene.name, source_sha256=expected,
               runtime_sha256=digest(scene.read_bytes()))
    closure = [dict(path=p.name, sha256=digest(p.read_bytes())) for p in (scene, dependency)]
    catalogue(monkeypatch, row, closure)
    assets = LocalAssets(root, {'station': scene.name})
    assert source_assets.verify_source_file(assets, 'station', expected) == scene
    dependency.write_bytes(b'wrong pixels')
    with pytest.raises(ValueError, match='closure changed'):
        source_assets.verify_source_file(assets, 'station', expected)
    with pytest.raises(ValueError, match='code-approved'):
        source_assets.verify_source_file(assets, 'station', '0'*64)


def private_closure_fixture(tmp_path, monkeypatch):
    from mind2act_eval.contracts import InstanceSpec
    root=tmp_path/'fixture-release';root.mkdir()
    scene=root/'scene.usda';scene.write_bytes(b'portable scene')
    texture=root/'texture.png';texture.write_bytes(b'accepted pixels')
    spec=InstanceSpec('demo','source-case','easy','source','v1','v1','v1')
    expected=digest(b'original scene')
    row=dict(asset_name=root.name,runtime_path=scene.name,source_sha256=expected,
             runtime_sha256=digest(scene.read_bytes()),identity=source_assets._identity(spec))
    members=[dict(path=p.name,sha256=digest(p.read_bytes())) for p in (scene,texture)]
    closure_id=digest(json.dumps(members,sort_keys=True).encode())
    proof=dict(schema='mind2act_eval.source_asset_closure.v1',closure_id=closure_id,members=members)
    proof_root=tmp_path/'private-proofs';proof_root.mkdir()
    catalogue(monkeypatch,row,members)
    value=source_assets._catalogue();value['approvals'][0]['closure_id']=closure_id
    value['closures']={}
    monkeypatch.setenv('MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT',str(proof_root))
    assets=LocalAssets(root,{'station':scene.name},spec=spec)
    return assets,expected,proof,proof_root,value


def bind_private_closure(proof,proof_root,value):
    raw=(json.dumps(proof,sort_keys=True)+'\n').encode();sha=digest(raw)
    path=proof_root/(sha+'.json');path.write_bytes(raw)
    value['closures'][value['approvals'][0]['closure_id']]=dict(
        schema='mind2act_eval.source_asset_closure_reference.v1',proof_sha256=sha)
    return path


@pytest.mark.parametrize('mutation',[None,'proof_bytes','dependency','source_identity','runtime_path','missing_proof','unset_root'])
def test_private_closure_keeps_full_projection_authorization(tmp_path,monkeypatch,mutation):
    from dataclasses import replace
    assets,expected,proof,proof_root,value=private_closure_fixture(tmp_path,monkeypatch)
    path=bind_private_closure(proof,proof_root,value)
    if mutation=='proof_bytes':path.write_text('{}')
    elif mutation=='dependency':(assets.root/'texture.png').write_bytes(b'changed pixels')
    elif mutation=='source_identity':assets.spec=replace(assets.spec,seed=1)
    elif mutation=='runtime_path':
        (assets.root/'other.usda').write_bytes((assets.root/'scene.usda').read_bytes())
        assets.entries['station']='other.usda'
    elif mutation=='missing_proof':path.unlink()
    elif mutation=='unset_root':monkeypatch.delenv('MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT')
    if mutation is None:
        assert source_assets.verify_source_file(assets,'station',expected)==assets.root/'scene.usda'
    else:
        with pytest.raises((ValueError,OSError)):
            source_assets.verify_source_file(assets,'station',expected)


@pytest.mark.parametrize('mutation',['schema','identity','drop_member','duplicate','unknown_member_field','reference_schema','reference_extra_field'])
def test_pinned_private_closure_still_rejects_malformed_proof(tmp_path,monkeypatch,mutation):
    assets,expected,proof,proof_root,value=private_closure_fixture(tmp_path,monkeypatch)
    if mutation=='schema':proof['schema']='unreviewed'
    elif mutation=='identity':proof['closure_id']='0'*64
    elif mutation=='drop_member':proof['members'].pop()
    elif mutation=='duplicate':proof['members'].append(proof['members'][0])
    elif mutation=='unknown_member_field':proof['members'][0]['optional']=True
    bind_private_closure(proof,proof_root,value)
    ref=value['closures'][value['approvals'][0]['closure_id']]
    if mutation=='reference_schema':ref['schema']='unreviewed'
    elif mutation=='reference_extra_field':ref['skip_sha']=True
    with pytest.raises(ValueError):source_assets.verify_source_file(assets,'station',expected)


@pytest.mark.parametrize('escape',['parent','symlink'])
def test_private_closure_cannot_escape_asset_root(tmp_path,monkeypatch,escape):
    assets,expected,proof,proof_root,value=private_closure_fixture(tmp_path,monkeypatch)
    outside=tmp_path/'outside.png';outside.write_bytes(b'accepted pixels')
    if escape=='parent':proof['members'][1]['path']='../outside.png'
    else:
        (assets.root/'escape.png').symlink_to(outside)
        proof['members'][1]['path']='escape.png'
    closure_id=digest(json.dumps(proof['members'],sort_keys=True).encode())
    proof['closure_id']=value['approvals'][0]['closure_id']=closure_id
    bind_private_closure(proof,proof_root,value)
    with pytest.raises(ValueError,match='contained|escapes'):
        source_assets.verify_source_file(assets,'station',expected)


def test_projection_is_bound_to_the_frozen_requested_source(tmp_path, monkeypatch):
    from mind2act_eval.contracts import InstanceSpec
    root = tmp_path/'fixture-release'
    root.mkdir()
    path = root/'scene.usda'
    path.write_bytes(b'portable scene')
    spec = InstanceSpec('demo','original-case','easy','source','v1','v1','v1')
    row = dict(asset_name=root.name, runtime_path=path.name,
               source_sha256=digest(b'original scene'), runtime_sha256=digest(path.read_bytes()),
               identity=source_assets._identity(spec))
    catalogue(monkeypatch, row, [dict(path=path.name, sha256=row['runtime_sha256'])])
    assert source_assets.verify_source_file(LocalAssets(root, {'station':path.name}, spec=spec),
                                           'station', row['source_sha256']) == path
    other = InstanceSpec('demo','different-case','easy','source','v1','v1','v1')
    with pytest.raises(ValueError, match='code-approved'):
        source_assets.verify_source_file(LocalAssets(root, {'station':path.name}, spec=other),
                                         'station', row['source_sha256'])


@pytest.mark.parametrize('mutation', [None, 'mass', 'recipe', 'instance', 'source', 'seed', 'closure'])
def test_exact_public_recipe_rebind_preserves_projection_and_original_json(tmp_path, monkeypatch, mutation):
    from dataclasses import replace
    from mind2act_eval.contracts import InstanceSpec
    root = tmp_path/'fixture-release'
    root.mkdir()
    original = b'{"path":"original","mass":2}'
    path = root/'source.json'
    path.write_bytes(b'{"path":"portable","mass":2}')
    dependency = root/'texture.png'
    dependency.write_bytes(b'accepted pixels')
    old = InstanceSpec('PR5', 'source-case', 'easy', 'source', 'v1', 'v1', 'v1',
                       config={'goal_recipe': 'old public recipe', 'mass': 2})
    new = replace(old, config={'goal_recipe': 'reviewed new public recipe', 'mass': 2})
    old_identity, new_identity = source_assets._identity(old), source_assets._identity(new)
    row = dict(asset_name=root.name, runtime_path=path.name, source_sha256=digest(original),
               runtime_sha256=digest(path.read_bytes()), identity=old_identity, json=dict(
                   undo=[dict(pointer='/path', source='original', runtime='portable')],
                   serialization=dict(separators=[',', ':'], trailing_newline=False)))
    closure = [dict(path=p.name, sha256=digest(p.read_bytes())) for p in (path, dependency)]
    catalogue(monkeypatch, row, closure)
    value = source_assets._catalogue()
    value['public_recipe_rebindings'] = [dict(identity=new_identity,
        approved_config_sha256=old_identity['config_sha256'], changed_fields=['goal_recipe'])]
    if mutation == 'mass':
        new = replace(new, config={'goal_recipe': 'reviewed new public recipe', 'mass': 3})
    elif mutation == 'recipe':
        new = replace(new, config={'goal_recipe': 'unreviewed recipe', 'mass': 2})
    elif mutation == 'instance':
        new = replace(new, instance_id='another-case')
    elif mutation == 'source':
        new = replace(new, source_id='another-source')
    elif mutation == 'seed':
        new = replace(new, seed=1)
    elif mutation == 'closure':
        dependency.write_bytes(b'changed pixels')
    assets = LocalAssets(root, {'source': path.name}, spec=new)
    if mutation is None:
        assert source_assets.verify_source_file(assets, 'source', digest(original)) == path
        assert source_assets.read_source_bytes(assets, 'source', digest(original)) == original
    else:
        with pytest.raises(ValueError):
            source_assets.read_source_bytes(assets, 'source', digest(original))


def test_typed_aliases_restore_distinct_original_tokens_by_location(tmp_path, monkeypatch):
    Sdf = pytest.importorskip('pxr.Sdf')
    root = tmp_path/'fixture-release'
    root.mkdir()
    original = Sdf.Layer.CreateAnonymous('original.usda')
    prim = Sdf.CreatePrimInLayer(original, '/World')
    a = Sdf.AttributeSpec(prim, 'first', Sdf.ValueTypeNames.Asset)
    b = Sdf.AttributeSpec(prim, 'second', Sdf.ValueTypeNames.Asset)
    a.default = Sdf.AssetPath('/original/first.png')
    b.default = Sdf.AssetPath('/alias/first.png')
    source_sha = digest(original.ExportToString().encode())
    a.default = b.default = Sdf.AssetPath('./same.png')
    runtime = root/'scene.usda'
    original.Export(str(runtime))
    actual = digest(runtime.read_bytes())
    row = dict(asset_name=root.name, runtime_path=runtime.name, source_sha256=source_sha,
               runtime_sha256=actual, usd=dict(tokens=[
                   dict(location='/World.first#default', source='/original/first.png', runtime='./same.png'),
                   dict(location='/World.second#default', source='/alias/first.png', runtime='./same.png')]))
    catalogue(monkeypatch, row, [dict(path=runtime.name, sha256=actual)])
    layer = source_assets.source_layer(LocalAssets(root, {'station': runtime.name}), 'station', source_sha)
    assert layer.GetAttributeAtPath('/World.first').default.path == '/original/first.png'
    assert layer.GetAttributeAtPath('/World.second').default.path == '/alias/first.png'
