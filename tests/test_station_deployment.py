import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Usd

from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim.station_deployment import prepare_open_station
from mind2act_eval.sim import source_assets
from mind2act_eval.contracts import InstanceSpec


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(config):
    return InstanceSpec('demo','source-case','easy','frozen-source','v1','v1','v1',config=config)


def approve(monkeypatch, spec, assets):
    selected=spec.config['station_material_relocation_manifest']
    row=dict(identity=source_assets._identity(spec),source_station_sha256=sha(assets.resolve('station')),
             manifest_sha256=sha(assets.resolve(selected)),manifest_asset_id=selected)
    roles=[dict(asset_id=k,path=str(assets.resolve(k).relative_to(assets.root)),sha256=sha(assets.resolve(k)))
           for k in assets.entries]
    proof=dict(row,schema='mind2act_eval.station_deployment_approval.v1',assets=roles,
               closure=[dict(path=r['path'],sha256=r['sha256']) for r in roles])
    raw=(json.dumps(proof,sort_keys=True)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
    directory=tempfile.TemporaryDirectory(prefix='deployment-proof-')
    monkeypatch._deployment_proof_directory=directory
    from pathlib import Path
    (Path(directory.name)/(digest+'.json')).write_bytes(raw)
    monkeypatch.setenv('MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT',directory.name)
    row['proof_sha256']=digest
    monkeypatch.setattr(source_assets,'_catalogue',lambda:dict(approvals=[],deployments=[row]))


def inputs(tmp_path):
    station = tmp_path / 'station.usda'
    station.write_text('''#usda 1.0
def Xform "World" {
    custom double mass = 2.75
    custom asset shader = @https://original.example/Metal.mdl@
}
''')
    material = tmp_path / 'Metal.mdl'
    material.write_text('mdl 1.7;\n')
    source_manifest = tmp_path / 'source.json'
    source_manifest.write_text(json.dumps(dict(schema='mind2act_eval.station_relocation.v1',
        source_sha256=sha(station), mappings=[])))
    deployment = tmp_path / 'deployment.json'
    deployment.write_text(json.dumps(dict(schema='mind2act_eval.station_relocation.v1',
        source_sha256=sha(station), mappings=[dict(source_layer_sha256=sha(station),
            authored_path='https://original.example/Metal.mdl', target_asset_id='metal',
            target_sha256=sha(material))])))
    assets = LocalAssets(tmp_path, dict(station=station.name, source=source_manifest.name,
        deployment=deployment.name, metal=material.name))
    return station, source_manifest, deployment, assets


def test_material_projection_preserves_original_source_pins(tmp_path, monkeypatch):
    station, source, deployment, assets = inputs(tmp_path)
    station_pin, source_pin = sha(station), sha(source)
    spec = request(dict(station_relocation_manifest='source',station_material_relocation_manifest='deployment'))
    approve(monkeypatch,spec,assets)
    output = tmp_path / 'attempt'
    derived = prepare_open_station(spec, assets, output)
    assert derived != station
    assert assets.resolve('station') == station and sha(station) == station_pin
    assert assets.resolve('source') == source and sha(source) == source_pin
    stage = Usd.Stage.Open(str(derived))
    assert stage.GetPrimAtPath('/World').GetAttribute('mass').Get() == 2.75
    token = stage.GetPrimAtPath('/World').GetAttribute('shader').Get()
    assert not Path(token.path).is_absolute()
    assert Path(token.resolvedPath) == assets.resolve('metal')
    proof = json.loads((output / 'station-material-deployment.json').read_text())
    assert proof['source_manifest_sha256'] == source_pin
    assert proof['material_manifest_sha256'] == sha(deployment)
    assert proof['source_asset_bindings_preserved'] is True


def test_legacy_unmapped_url_still_fails_the_portability_guard(tmp_path):
    _, _, _, assets = inputs(tmp_path)
    with pytest.raises(ValueError, match='nonportable generated layer'):
        prepare_open_station(SimpleNamespace(config=dict(station_relocation_manifest='source')),
            assets, tmp_path / 'attempt')


def test_deployment_does_not_bypass_target_hash_validation(tmp_path, monkeypatch):
    _, _, _, assets = inputs(tmp_path)
    spec=request(dict(station_relocation_manifest='source',station_material_relocation_manifest='deployment'))
    approve(monkeypatch,spec,assets)
    assets.resolve('metal').write_text('changed shader bytes')
    with pytest.raises(ValueError, match='target bytes changed|closure changed'):
        prepare_open_station(spec, assets, tmp_path / 'attempt')


def test_caller_manifest_cannot_replace_the_opened_geometry(tmp_path, monkeypatch):
    from mind2act_eval.sim.source_assets import verify_source_file
    source=tmp_path/'source.usda';source.write_text('#usda 1.0\n(defaultPrim = "World")\ndef Xform "World"\n{\n def Cube "Object" (prepend references = @original.usda@)\n {\n }\n}\n')
    original=tmp_path/'original.usda';original.write_text('#usda 1.0\n(defaultPrim = "Cube")\ndef Cube "Cube"\n{\n double size = 1\n}\n')
    changed=tmp_path/'changed.usda';changed.write_text('#usda 1.0\n(defaultPrim = "Cube")\ndef Cube "Cube"\n{\n double size = 9\n}\n')
    stage=Usd.Stage.Open(str(source))
    assert stage.GetPrimAtPath('/World/Object').GetAttribute('size').Get()==1
    manifest=tmp_path/'caller.json';manifest.write_text(json.dumps(dict(schema='mind2act_eval.station_relocation.v1',source_sha256=sha(source),mappings=[dict(source_layer_sha256=sha(source),authored_path='original.usda',target_asset_id='replacement',target_sha256=sha(changed))])))
    assets=LocalAssets(tmp_path,dict(station=source.name,deployment=manifest.name,replacement=changed.name))
    assert verify_source_file(assets,'station',sha(source))==source
    monkeypatch.setattr(source_assets,'_catalogue',lambda:dict(approvals=[],deployments=[]))
    with pytest.raises(ValueError,match='code-owned'):
        prepare_open_station(request(dict(station_material_relocation_manifest='deployment')),assets,tmp_path/'attempt')


def test_self_consistent_manifest_rewrite_cannot_refresh_approval(tmp_path, monkeypatch):
    _,_,_,assets=inputs(tmp_path)
    spec=request(dict(station_relocation_manifest='source',station_material_relocation_manifest='deployment'))
    approve(monkeypatch,spec,assets)
    assets.resolve('metal').write_text('new self-reported shader bytes')
    manifest=assets.resolve('deployment');value=json.loads(manifest.read_text())
    value['mappings'][0]['target_sha256']=sha(assets.resolve('metal'));manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='code-owned'):
        prepare_open_station(spec,assets,tmp_path/'attempt')


@pytest.mark.parametrize('value', [None, '', 1])
def test_invalid_explicit_deployment_is_rejected(tmp_path, value):
    _, _, _, assets = inputs(tmp_path)
    with pytest.raises(ValueError, match='declared asset ID'):
        prepare_open_station(SimpleNamespace(config=dict(station_material_relocation_manifest=value)),
            assets, tmp_path / 'attempt')
