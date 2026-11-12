"""CPU Sdf relocation tests; no Kit, GPU, or real source assets."""
import hashlib
import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd
from usdz_support import package_writer
from mind2act_eval.sim import LocalAssets
from mind2act_eval.sim.asset_relocation import prepare_station


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path


def fixture(tmp_path):
    root=tmp_path/'assets';root.mkdir()
    child=write(root/'child.usda','#usda 1.0\ndef Xform "Geom" {\n custom double size = 3\n}\n')
    source=write(root/'source.usda','''#usda 1.0
(defaultPrim = "World")
def Xform "World" (prepend references = @/old/geom.usda@</Geom>) {
 double3 xformOp:translate = (1, 2, 3)
 uniform token[] xformOpOrder = ["xformOp:translate"]
 custom string label = "/old/geom.usda"
 custom asset shader = @gltf/pbr.mdl@
 custom asset image = @/home/unknown/Image_0.png@
}
''')
    entries={'station':'source.usda','geom':'child.usda','relocation':'relocation.json'}
    manifest={'schema':'mind2act_eval.station_relocation.v1','source_sha256':sha(source),
        'mappings':[{'source_layer_sha256':sha(source),'authored_path':'/old/geom.usda',
                     'target_asset_id':'geom','target_sha256':sha(child)}]}
    (root/'relocation.json').write_text(json.dumps(manifest))
    return source,LocalAssets(root,entries),manifest


def test_default_is_exact_noop_without_resolver_or_output(tmp_path):
    source=tmp_path/'not-opened.usda'
    assert prepare_station(source,None,None,tmp_path/'output') is source
    assert not (tmp_path/'output').exists()


def test_station_episodes_are_thin_and_reuse_verified_immutable_layers(tmp_path, monkeypatch):
    from pxr import UsdGeom, UsdPhysics, UsdShade
    monkeypatch.setenv('M2A_CACHE_ROOT', str(tmp_path / 'cluster-cache'))
    source, assets, manifest = fixture(tmp_path)
    child = assets.resolve('geom')
    stage = Usd.Stage.Open(str(child))
    mesh = UsdGeom.Mesh.Define(stage, '/Geom/Mesh')
    mesh.CreatePointsAttr([(float(i), 0., 0.) for i in range(20000)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, '/Geom/Material')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    manifest['mappings'][0]['target_sha256'] = sha(child)
    (assets.root / 'relocation.json').write_text(json.dumps(manifest))
    first = prepare_station(source, assets, 'relocation', tmp_path / 'one')
    second = prepare_station(source, assets, 'relocation', tmp_path / 'two')
    assert first.stat().st_size < 1024 and second.stat().st_size < 1024
    assert sum(p.stat().st_size for p in first.parent.rglob('*') if p.is_file()) < 1024
    assert sum(p.stat().st_size for p in second.parent.rglob('*') if p.is_file()) < 1024
    one = json.loads((tmp_path / 'one/station-relocation.json').read_text())
    two = json.loads((tmp_path / 'two/station-relocation.json').read_text())
    assert one['immutable_runtime_path'] == two['immutable_runtime_path']
    for path in (first, second):
        composed = Usd.Stage.Open(str(path))
        assert composed.GetDefaultPrim().GetPath() == '/World'
        prim = composed.GetPrimAtPath('/World/Mesh')
        assert len(UsdGeom.Mesh(prim).GetPointsAttr().Get()) == 20000
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath() == '/World/Material'
        assert tuple(composed.GetPrimAtPath('/World').GetAttribute('xformOp:translate').Get()) == (1, 2, 3)
    child.write_text(child.read_text() + '\n# tampered immutable input\n')
    with pytest.raises(ValueError, match='SHA'):
        prepare_station(source, assets, 'relocation', tmp_path / 'three')


def test_source_geometry_pose_and_soft_missing_are_preserved(tmp_path):
    source,assets,manifest=fixture(tmp_path);before=source.read_bytes();output=tmp_path/'output'
    derived=prepare_station(source,assets,'relocation',output)
    assert source.read_bytes()==before and assets.resolve('station')==source
    stage=Usd.Stage.Open(str(derived));prim=stage.GetPrimAtPath('/World')
    assert prim.GetAttribute('size').Get()==3
    assert tuple(prim.GetAttribute('xformOp:translate').Get())==(1,2,3)
    assert prim.GetAttribute('label').Get()=='/old/geom.usda'
    assert prim.GetAttribute('shader').Get().path=='gltf/pbr.mdl'
    assert not Path(prim.GetAttribute('image').Get().path).is_absolute()
    image = prim.GetAttribute('image')
    assert Path(Sdf.ComputeAssetPathRelativeToLayer(image.GetPropertyStack()[0].layer,
        image.Get().path)).resolve()==Path('/home/unknown/Image_0.png').resolve()
    receipt=json.loads((output/'station-relocation.json').read_text())
    assert receipt['status']=='prepared' and receipt['source_sha256']==sha(source)
    assert receipt['runtime_sha256']==sha(derived)
    assert all(row['source_non_asset_sha256']==row['runtime_non_asset_sha256'] for row in receipt['layers'])
    assert {row['authored_path'] for row in receipt['unverified_assets']}=={'gltf/pbr.mdl','/home/unknown/Image_0.png'}


@pytest.mark.parametrize('bad', ['source_hash','target_hash','missing_mapping','missing_target','escape'])
def test_exact_geometry_and_hashes_fail_closed(tmp_path,bad):
    source,assets,manifest=fixture(tmp_path)
    if bad=='source_hash':manifest['source_sha256']='0'*64
    elif bad=='target_hash':manifest['mappings'][0]['target_sha256']='0'*64
    elif bad=='missing_mapping':manifest['mappings']=[]
    elif bad=='missing_target':assets.entries['geom']='not-the-neighbour.usda'
    else:
        write(tmp_path/'outside.usda','#usda 1.0\ndef Xform "Geom" {}\n')
        assets.entries['geom']='../outside.usda'
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError,FileNotFoundError)):
        prepare_station(source,assets,'relocation',tmp_path/'output')


def test_nested_relative_composition_and_relative_texture_anchor(tmp_path):
    source,assets,manifest=fixture(tmp_path)
    leaf=write(assets.root/'leaf.usda','#usda 1.0\ndef Xform "Leaf" {\n custom int nested = 7\n}\n')
    child=write(assets.root/'child.usda','''#usda 1.0
 def Xform "Geom" (prepend references = @leaf.usda@</Leaf>) {
 custom asset nested_image = @images/unknown.png@
 }
''')
    assets.entries['leaf']='leaf.usda'
    manifest['mappings'][0]['target_sha256']=sha(child)
    manifest['mappings'].append(dict(source_layer_sha256=sha(child),authored_path='leaf.usda',
        target_asset_id='leaf',target_sha256=sha(leaf)))
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    result=prepare_station(source,assets,'relocation',tmp_path/'output')
    stage=Usd.Stage.Open(str(result));prim=stage.GetPrimAtPath('/World')
    assert prim.GetAttribute('nested').Get()==7
    image = prim.GetAttribute('nested_image')
    # Relative tokens belong to their authored immutable layer, not the thin
    # episode layer that sublayers it.
    authored_layer = image.GetPropertyStack()[0].layer
    assert Path(Sdf.ComputeAssetPathRelativeToLayer(authored_layer, image.Get().path)).resolve()==assets.root/'images/unknown.png'
    assert child.read_text().find('@leaf.usda@')>=0


def test_missing_nested_geometry_mapping_is_not_ignored_as_material(tmp_path):
    source,assets,manifest=fixture(tmp_path)
    child=write(assets.root/'child.usda','#usda 1.0\ndef Xform "Geom" (prepend references = @missing.usda@) {}\n')
    manifest['mappings'][0]['target_sha256']=sha(child)
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='missing exact geometry/composition'):
        prepare_station(source,assets,'relocation',tmp_path/'output')


def test_usdz_package_is_never_rewritten(tmp_path):
    from pxr import UsdUtils
    source,assets,manifest=fixture(tmp_path)
    mesh=write(assets.root/'package-root.usda','#usda 1.0\n(defaultPrim="Geom")\ndef Xform "Geom" {\n custom asset shader = @gltf/pbr.mdl@\n}\n')
    package=assets.root/'geometry.usdz'
    with package_writer(package) as writer:
        assert writer.AddFile(str(mesh),'package-root.usda')
    before=package.read_bytes();assets.entries['geom']='geometry.usdz'
    source.write_text(source.read_text().replace('/old/geom.usda','/old/geom.usdz'))
    manifest['source_sha256']=sha(source)
    manifest['mappings']=[dict(source_layer_sha256=sha(source),authored_path='/old/geom.usdz',
        target_asset_id='geom',target_sha256=sha(package))]
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    runtime=prepare_station(source,assets,'relocation',tmp_path/'output')
    assert Usd.Stage.Open(str(runtime))
    assert package.read_bytes()==before
    receipt=json.loads((tmp_path/'output/station-relocation.json').read_text())
    assert any(row['authored_path']=='gltf/pbr.mdl' for row in receipt['unverified_assets'])
    assert any(row['kind']=='immutable_package' and row['runtime_path']==str(package) for row in receipt['layers'])


@pytest.mark.parametrize('enabled',[False,True])
def test_cli_opens_runtime_but_context_resolves_original_station(tmp_path,monkeypatch,enabled):
    from types import SimpleNamespace
    from mind2act_eval import cli,sim,robot
    from mind2act_eval.contracts import InstanceSpec,to_json
    from mind2act_eval.tasks import catalog
    source,assets,_=fixture(tmp_path);original=source.read_bytes()
    write(assets.root/'x5.usd','synthetic robot asset')
    entries=dict(assets.entries,x5='x5.usd')
    config={'station_relocation_manifest':'relocation'} if enabled else {}
    spec=InstanceSpec('demo','synthetic','easy','public','v1','v1','v1',assets=entries,config=config)
    spec_path=write(tmp_path/'spec.json',json.dumps(to_json(spec)))
    opened=[]
    class Simulator:
        def __init__(self,*a,**kw):pass
        def rgb(self):return {}
        def open_stage(self,path):opened.append(path)
        def close(self):pass
    monkeypatch.setattr(sim,'IsaacSimulator',Simulator)
    monkeypatch.setattr(robot,'X5Robot',lambda **kw:None)
    monkeypatch.setattr(catalog,'load_task',lambda task_id:None)
    def episode(spec,task,ctx,adapter,**kw):
        assert ctx.assets.resolve('station')==source
        assert source.read_bytes()==original
        return SimpleNamespace(infra_valid=True)
    monkeypatch.setattr(cli,'run_episode',episode)
    output=tmp_path/'run';output.mkdir()
    args=SimpleNamespace(spec=str(spec_path),asset_root=str(assets.root),output=str(output),backend='isaac',
        gpu=0,h5=None,adapter='mind2act_eval.adapters.demo:create',adapter_config=None,physics_ticks=4,wall_s=10)
    assert cli._worker(args)==0
    assert (opened[0]!=source) is enabled
    assert (output/'station-relocation.json').exists() is enabled


def test_sublayer_and_payload_paths_are_relocated_without_flattening(tmp_path):
    source,assets,manifest=fixture(tmp_path)
    sub=write(assets.root/'sub.usda','#usda 1.0\nover "World" {\n custom int from_sub = 8\n}\n')
    payload=write(assets.root/'payload.usda','#usda 1.0\ndef Xform "Payload" {\n custom int from_payload = 9\n}\n')
    source.write_text(source.read_text().replace('(defaultPrim = "World")',
        '(defaultPrim = "World"\n subLayers = [@/old/sub.usda@])').replace(
        'prepend references = @/old/geom.usda@</Geom>',
        'prepend references = @/old/geom.usda@</Geom>\n prepend payload = @/old/payload.usda@</Payload>'))
    pinned=source.read_bytes();root_sha=sha(source)
    manifest['source_sha256']=root_sha
    manifest['mappings'][0]['source_layer_sha256']=root_sha
    for name,path in (('sub',sub),('payload',payload)):
        assets.entries[name]=path.name
        manifest['mappings'].append(dict(source_layer_sha256=root_sha,authored_path='/old/'+name+'.usda',
            target_asset_id=name,target_sha256=sha(path)))
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    runtime=prepare_station(source,assets,'relocation',tmp_path/'output')
    stage=Usd.Stage.Open(str(runtime));prim=stage.GetPrimAtPath('/World')
    assert prim.GetAttribute('from_sub').Get()==8
    assert prim.GetAttribute('from_payload').Get()==9
    assert source.read_bytes()==pinned
    assert len(Sdf.Layer.FindOrOpen(str(runtime)).subLayerPaths)==1


@pytest.mark.parametrize('broken',[False,True])
def test_older_usd_prim_index_composition_check(tmp_path,monkeypatch,broken):
    source,assets,manifest=fixture(tmp_path)
    package=assets.root/'geometry.usdz'
    text='#usda 1.0\n(defaultPrim="Geom")\ndef Xform "Geom"'
    text+=' (prepend references = @missing.usda@)' if broken else ''
    mesh=write(assets.root/'package-root.usda',text+' {}\n')
    with package_writer(package) as writer:
        writer.AddFile(str(mesh),'package-root.usda')
    assets.entries['geom']='geometry.usdz'
    manifest['mappings'][0]['target_sha256']=sha(package)
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    open_stage=Usd.Stage.Open
    class OlderStage:
        def __init__(self,stage):self.stage=stage
        def __getattr__(self,name):
            if name=='GetCompositionErrors':raise AttributeError(name)
            return getattr(self.stage,name)
    monkeypatch.setattr(Usd.Stage,'Open',lambda *args,**kw:OlderStage(open_stage(*args,**kw)))
    if broken:
        with pytest.raises(ValueError,match='composition errors'):
            prepare_station(source,assets,'relocation',tmp_path/'output')
    else:
        assert prepare_station(source,assets,'relocation',tmp_path/'output').is_file()


@pytest.mark.parametrize('uri',[
    'https://assets.example.test/materials/Plastic_ABS.mdl',
    'omniverse://asset-server/materials/Plastic_ABS.mdl',
])
def test_unmapped_uri_fails_generated_layer_publication_without_changing_source(tmp_path,uri):
    source,assets,manifest=fixture(tmp_path)
    source.write_text(source.read_text().replace('gltf/pbr.mdl',uri))
    manifest['source_sha256']=sha(source)
    manifest['mappings'][0]['source_layer_sha256']=sha(source)
    (assets.root/'relocation.json').write_text(json.dumps(manifest))
    original=source.read_bytes()
    output=tmp_path/'output'
    with pytest.raises(ValueError,match='nonportable generated layer'):
        prepare_station(source,assets,'relocation',output)
    receipt=json.loads((output/'station-relocation.json').read_text())
    observed=[row for row in receipt['unverified_assets'] if row['authored_path']==uri]
    assert len(observed)==1 and observed[0]['runtime_asset_path']==uri
    assert receipt['status']=='failed'
    assert source.read_bytes()==original


def test_cli_mdl_paths_precede_simulator_construction(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    from mind2act_eval import cli,sim,robot
    from mind2act_eval.contracts import InstanceSpec,to_json
    from mind2act_eval.tasks import catalog
    from mind2act_eval.sim import material_closure
    root=tmp_path/'assets';root.mkdir();(root/'mdl').mkdir();write(root/'x5.usd','synthetic')
    spec=InstanceSpec('demo','mdl-order','easy','public','v1','v1','v1',
        assets=dict(x5='x5.usd',mdl='mdl'),config=dict(material_mdl_search_asset_ids=['mdl']))
    spec_path=write(tmp_path/'spec.json',json.dumps(to_json(spec)))
    monkeypatch.setenv('MDL_SYSTEM_PATH','prior-system');monkeypatch.setenv('MDL_USER_PATH','prior-user')
    class Simulator:
        def __init__(self,*args,**kwargs):
            system,user=os.environ.get('MDL_SYSTEM_PATH'),os.environ.get('MDL_USER_PATH')
            assert system==str((root/'mdl').resolve())
            assert user==str((root/'mdl').resolve())
        def rgb(self):return {}
        def close(self):pass
    monkeypatch.setattr(sim,'IsaacSimulator',Simulator)
    monkeypatch.setattr(robot,'X5Robot',lambda **kw:None)
    monkeypatch.setattr(catalog,'load_task',lambda task_id:None)
    monkeypatch.setattr(material_closure,'verify_native_mdl_search',lambda receipt:dict(receipt,native_compiler_paths_verified=True))
    monkeypatch.setattr(cli,'run_episode',lambda *a,**kw:SimpleNamespace(infra_valid=True))
    output=tmp_path/'run';output.mkdir()
    args=SimpleNamespace(spec=str(spec_path),asset_root=str(root),output=str(output),backend='isaac',
        gpu=0,h5=None,adapter='mind2act_eval.adapters.demo:create',adapter_config=None,physics_ticks=4,wall_s=10)
    assert cli._worker(args)==0
