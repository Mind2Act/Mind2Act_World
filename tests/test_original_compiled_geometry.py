"""Independent Linux reopen of compiler output; no Isaac/native qualification."""
from dataclasses import replace
from importlib import import_module
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.contracts import InstanceSpec,to_json
from mind2act_eval.original_test_inputs import digest,read


def compare_fruit_source(stage,raw):
    """Check resolved USD geometry, not just reference declarations or poses."""
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf
    for obj in raw['config']['task']['materialization']['objects']:
        source_path=Path(obj['asset_file'])
        assert digest(source_path.read_bytes())==obj['asset_sha256']
        original=Usd.Stage.Open(str(source_path));source_root=original.GetDefaultPrim().GetPath()
        source_bodies=[p for p in original.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert len(source_bodies)==1
        source_body=source_bodies[0]
        target_body_path=Sdf.Path('/World/Task'+obj['body_path'].removeprefix('/World'))
        target_root=target_body_path.GetParentPath()
        target_body=stage.GetPrimAtPath(target_body_path)
        def mapped(path):return path.ReplacePrefix(source_root,target_root) if path.HasPrefix(source_root) else path
        def value(v,attr,source=False):
            if isinstance(v,Sdf.AssetPath):
                path=v.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(attr.GetPropertyStack()[0].layer,v.path)
                return ('asset-bytes',digest(Path(path).read_bytes()))
            if isinstance(v,Sdf.Path):return str(mapped(v) if source else v)
            if v is None or isinstance(v,(str,bool,int,float)):return v
            return tuple(value(x,attr,source) for x in v)
        source_meshes={str(p.GetPath().MakeRelativePath(source_body.GetPath())):p
            for p in Usd.PrimRange(source_body) if p.IsA(UsdGeom.Mesh)}
        actual_meshes={str(p.GetPath().MakeRelativePath(target_body_path)):p
            for p in Usd.PrimRange(target_body) if p.IsA(UsdGeom.Mesh)}
        assert source_meshes and set(source_meshes)==set(actual_meshes)
        for name,mesh in source_meshes.items():
            actual=actual_meshes[name]
            for attr in mesh.GetAttributes():
                other=actual.GetAttribute(attr.GetName())
                assert other and value(attr.Get(),attr,True)==value(other.Get(),other), (name,attr.GetName())
            for purpose in ('allPurpose','physics'):
                material=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial(materialPurpose=purpose)[0]
                bound=UsdShade.MaterialBindingAPI(actual).ComputeBoundMaterial(materialPurpose=purpose)[0]
                assert bool(material)==bool(bound)
                if not material:continue
                assert bound.GetPath()==mapped(material.GetPath())
                for prim in Usd.PrimRange(material.GetPrim()):
                    target=stage.GetPrimAtPath(mapped(prim.GetPath()))
                    assert target and target.GetTypeName()==prim.GetTypeName()
                    for attr in prim.GetAttributes():
                        other=target.GetAttribute(attr.GetName())
                        assert other and value(attr.Get(),attr,True)==value(other.Get(),other)
                        assert [mapped(p) for p in attr.GetConnections()]==other.GetConnections()


@pytest.fixture(scope='module')
def compiled():
    value=os.environ.get('MIND2ACT_ORIGINAL_COMPILED_ROOT')
    if not value:pytest.skip('private compiled candidate root not configured')
    root=Path(value)
    paths=sorted(root.glob('test-cases/*/*/*/spec.json'))
    assert paths
    return root,paths


def assets_for(root,spec):
    return SimpleNamespace(root=root,spec=spec,resolve=lambda key:root/spec.assets[key])


def validate(spec,assets):
    if spec.task_id=='M2':
        from mind2act_eval.tasks.M2.binding import load
        return load(spec,assets)
    if spec.task_id=='CP09':
        from importlib.resources import files
        from mind2act_eval.tasks.CP09.binding import validate_station
        from mind2act_eval.tasks.CP09.instances import validate as validate_spec
        row=validate_spec(spec)
        sha=digest(files('mind2act_eval.tasks.CP09').joinpath('config/station.json').read_bytes())
        return validate_station(row,assets,sha,spec=spec)
    module=import_module('mind2act_eval.tasks.'+spec.task_id+'.instances')
    return module.validate_assets(spec,assets)


def test_reopen_real_compiled_source_cases_and_original_physical_parameters(compiled):
    from pxr import Usd,UsdGeom,UsdPhysics
    root,paths=compiled
    for path in paths:
        spec=InstanceSpec.from_dict(json.loads(path.read_text()));assets=assets_for(root,spec)
        assert spec.task_id in ('M2','CP06','CP08','CP09')
        binding=validate(spec,assets)
        stage=Usd.Stage.Open(str(assets.resolve('station')))
        assert stage and UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
        if spec.task_id=='M2':
            from mind2act_eval.tasks.M2.native_stage import check_stage
            check_stage(stage,len(binding['card_identities']))
        elif spec.task_id=='CP08':
            case=spec.config['source_case']
            for b in (*case['blocks'],*case['distractors']):
                prim=stage.GetPrimAtPath('/World/CP11/Blocks/'+b['id'])
                assert prim and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                pose=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                assert np.allclose(pose.ExtractTranslation(),b['initial'],rtol=0,atol=1e-6)
        elif spec.task_id=='CP06':
            for obj in spec.config['source_task']['all_objects']:
                prim=stage.GetPrimAtPath(obj['path']);assert prim and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                pose=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                assert np.allclose(pose.ExtractTranslation(),obj['initial_position_m'],rtol=0,atol=1e-6)
                q=pose.ExtractRotationQuat();actual=np.r_[q.GetReal(),q.GetImaginary()]
                assert abs(np.dot(actual,obj['initial_orientation_wxyz']))==pytest.approx(1,abs=1e-6)
                mass=UsdPhysics.MassAPI(prim)
                assert mass.GetMassAttr().Get()==pytest.approx(obj['grams']/1000,abs=1e-6)
                assert np.allclose(mass.GetCenterOfMassAttr().Get(),obj['center_of_mass_m'],rtol=0,atol=1e-6)
                assert np.allclose(mass.GetDiagonalInertiaAttr().Get(),obj['diagonal_inertia_kg_m2'],rtol=0,atol=1e-7)
            compare_fruit_source(stage,read(spec,assets))
        else:
            from mind2act_eval.sim.station_deployment import prepare_open_station
            assert prepare_open_station(spec,assets,root/'unused-output')==assets.resolve('station')
            old=stage.GetPrimAtPath('/World/CP09')
            assert not old or not old.IsActive()
            assert stage.GetPrimAtPath('/World/Worktable')


def test_rejects_swapped_scene_despite_valid_original_json_and_refreshed_proof_hash(compiled,tmp_path):
    root,paths=compiled
    paths=[p for p in paths if '/CP08/' in str(p) or '/CP06/' in str(p)]
    if len(paths)<2:pytest.skip('two proof-bound candidates required')
    spec=InstanceSpec.from_dict(json.loads(paths[0].read_text()));assets=assets_for(root,spec)
    other=InstanceSpec.from_dict(json.loads(paths[-1].read_text()))
    proof=json.loads(assets.resolve('source_geometry_proof').read_text())
    proof['source_configuration_sha256']=other.config['test_source']['source_configuration_sha256']
    raw=json.dumps(proof).encode();path=tmp_path/'wrong-proof.json';path.write_bytes(raw)
    c=to_json(spec.config);c['source_geometry_proof_sha256']=digest(raw)
    changed=replace(spec,config=c)
    mapped=SimpleNamespace(root=root,spec=changed,resolve=lambda key:
        path if key=='source_geometry_proof' else assets.resolve(key))
    with pytest.raises(ValueError,match='geometry provenance'):
        validate(changed,mapped)
    wrong_scene=SimpleNamespace(root=root,spec=spec,resolve=lambda key:
        root/other.assets['station'] if key=='station' else assets.resolve(key))
    with pytest.raises(ValueError):validate(spec,wrong_scene)


def test_m2_rejects_rebound_layout_and_another_case_scene(compiled,tmp_path):
    root,paths=compiled
    paths=[p for p in paths if '/M2/' in str(p)]
    if len(paths)<2:pytest.skip('two original M2 compiled candidates required')
    spec=InstanceSpec.from_dict(json.loads(paths[0].read_text()));assets=assets_for(root,spec)
    binding=json.loads(assets.resolve('m2_binding').read_text())
    binding['source_board_offset_xy'][0]+=.001
    raw=json.dumps(binding).encode();path=tmp_path/'wrong-binding.json';path.write_bytes(raw)
    c=to_json(spec.config);c['m2_binding_sha256']=digest(raw)
    changed=replace(spec,config=c)
    mapped=SimpleNamespace(root=root,spec=changed,resolve=lambda key:
        path if key=='m2_binding' else assets.resolve(key))
    with pytest.raises(ValueError,match='deck/layout'):
        validate(changed,mapped)
    other=InstanceSpec.from_dict(json.loads(paths[-1].read_text()))
    wrong_scene=SimpleNamespace(root=root,spec=spec,resolve=lambda key:
        root/other.assets['station'] if key=='station' else assets.resolve(key))
    with pytest.raises(ValueError):validate(spec,wrong_scene)


def test_cp09_portable_station_and_proof_are_code_owned_not_caller_selected(compiled):
    root,paths=compiled
    paths=[p for p in paths if '/CP09/' in str(p)]
    if len(paths)<2:pytest.skip('two original CP09 compiled candidates required')
    spec=InstanceSpec.from_dict(json.loads(paths[0].read_text()));assets=assets_for(root,spec)
    other=InstanceSpec.from_dict(json.loads(paths[-1].read_text()))
    for role in ('station','station_proof'):
        mapped=SimpleNamespace(root=root,spec=spec,resolve=lambda key:
            root/other.assets[role] if key==role else assets.resolve(key))
        with pytest.raises(ValueError):validate(spec,mapped)
    for legacy in ('station_relocation_manifest','station_material_relocation_manifest'):
        c=to_json(spec.config);c[legacy]='legacy-training-relocation'
        changed=replace(spec,config=c)
        with pytest.raises(ValueError,match='already-portable'):validate(changed,assets_for(root,changed))


def test_cp06_resolved_mesh_override_rejects_even_with_unchanged_body_pose_mass_and_receipt(compiled):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    root,paths=compiled;paths=[p for p in paths if '/CP06/' in str(p)]
    if not paths:pytest.skip('original CP06 compiled candidate required')
    spec=InstanceSpec.from_dict(json.loads(paths[0].read_text()));assets=assets_for(root,spec)
    validate(spec,assets);raw=read(spec,assets)
    stage=Usd.Stage.Open(str(assets.resolve('station')))
    body=stage.GetPrimAtPath('/World/Task'+raw['config']['task']['materialization']['objects'][0]['body_path'].removeprefix('/World'))
    pose=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0);mass=UsdPhysics.MassAPI(body).GetMassAttr().Get()
    mesh=next(p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh))
    points=list(UsdGeom.Mesh(mesh).GetPointsAttr().Get());p=points[0]
    points[0]=Gf.Vec3f(p[0]+.01,p[1],p[2])
    stage.SetEditTarget(stage.GetSessionLayer());UsdGeom.Mesh(mesh).GetPointsAttr().Set(points)
    assert UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)==pose
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get()==mass
    with pytest.raises(AssertionError):compare_fruit_source(stage,raw)
