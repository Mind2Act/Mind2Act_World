"""Actual derived physical mutations must reject after every caller rehash."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.sim import LocalAssets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(spec, assets):
    if spec.task_id == 'M2':
        from mind2act_eval.tasks.M2.binding import load
        return load(spec, assets)
    from importlib import import_module
    return import_module('mind2act_eval.tasks.'+spec.task_id+'.instances').validate_assets(spec, assets)


@pytest.fixture(scope='module', params=['M2', 'CP06', 'CP08'])
def delivery(request, tmp_path_factory):
    value = os.environ.get('MIND2ACT_ORIGINAL_DELIVERIES_ROOT')
    if not value:
        pytest.skip('private original deliveries not configured')
    root = tmp_path_factory.mktemp(request.param)/'delivery'
    shutil.copytree(Path(value)/request.param, root, copy_function=os.link)
    return request.param, root


@pytest.fixture(params=['easy', 'medium', 'hard'])
def actual(request, delivery):
    task, root = delivery
    path = root/'test-cases'/task/request.param/(task+'_'+request.param+'_clean_000')/'spec.json'
    spec = InstanceSpec.from_dict(json.loads(path.read_text()))
    return spec, LocalAssets(root, spec.assets, spec=spec)


@pytest.mark.parametrize('mutation', ['units', 'mass', 'task_or_targets', 'source_physics', 'source_pose'])
def test_rehashed_actual_geometry_is_not_source_authority(actual, tmp_path, mutation):
    from pxr import Usd, Sdf, Gf
    spec, assets = actual
    config = to_json(spec.config)
    roles = dict(spec.assets)
    scene = assets.resolve('station')
    layer = Sdf.Layer.OpenAsAnonymous(str(scene))
    changed_scene = scene.with_name(tmp_path.name+'.usda')
    proof = None
    if spec.task_id != 'M2':
        proof = json.loads(assets.resolve('source_geometry_proof').read_text())
    if mutation == 'units':
        layer.pseudoRoot.SetInfo('metersPerUnit', .01)
    elif mutation == 'mass':
        body = ('/World/Memory/Cards/Card0' if spec.task_id == 'M2' else
            spec.config['source_task']['all_objects'][0]['path'] if spec.task_id == 'CP06'
            else '/World/CP11/Blocks/A')
        prim = Sdf.CreatePrimInLayer(layer, body)
        Sdf.AttributeSpec(prim, 'physics:mass', Sdf.ValueTypeNames.Float).default = 100.
    elif spec.task_id != 'M2':
        task = assets.resolve('task_source')
        task_layer = Sdf.Layer.OpenAsAnonymous(str(task))
        if mutation == 'task_or_targets':
            points = []
            task_layer.Traverse('/', lambda p: points.append(task_layer.GetAttributeAtPath(p))
                if p.IsPropertyPath() and p.name == 'points' else None)
            values = list(points[0].default)
            values[0] = Gf.Vec3f(values[0][0], values[0][1], values[0][2]+1)
            points[0].default = values
        elif spec.task_id == 'CP06':
            body = spec.config['source_task']['all_objects'][0]['path']
            name = 'physics:centerOfMass' if mutation == 'source_physics' else 'xformOp:transform'
            attr = task_layer.GetAttributeAtPath(body+'.'+name)
            value = attr.default
            if mutation == 'source_physics':
                attr.default = Gf.Vec3f(value[0]+.1, value[1], value[2])
            else:
                value.SetTranslateOnly(value.ExtractTranslation()+Gf.Vec3d(.1,0,0))
                attr.default = value
        else:
            name = 'physics:mass' if mutation == 'source_physics' else 'xformOp:translate'
            attr = task_layer.GetAttributeAtPath('/World/CP11/Blocks/A.'+name)
            attr.default = 100. if mutation == 'source_physics' else attr.default+Gf.Vec3d(.1,0,0)
        changed_task = task.with_name(tmp_path.name+'-task.usda')
        task_layer.Export(str(changed_task))
        layer.subLayerPaths = [os.path.relpath(changed_task, scene.parent), layer.subLayerPaths[1]]
        roles['task_source'] = str(changed_task.relative_to(assets.root))
        proof['actual_composed_task'] = dict(path=roles['task_source'], sha256=sha(changed_task))
        if spec.task_id == 'CP06':
            config['derived_task_layer_sha256'] = sha(changed_task)
    layer.Export(str(changed_scene))
    roles['station'] = str(changed_scene.relative_to(assets.root))
    if spec.task_id == 'M2':
        binding = json.loads(assets.resolve('m2_binding').read_text())
        binding['scene_sha256'] = sha(changed_scene)
        if mutation in ('task_or_targets', 'source_pose'):
            binding['button_rest_m'][0] += .4
        elif mutation == 'source_physics':
            binding['faceup_quaternions_wxyz'][0] = [1.,0.,0.,0.]
        path = changed_scene.with_suffix('.binding.json')
        path.write_text(json.dumps(binding))
        config['m2_binding_sha256'] = sha(path)
        roles['m2_binding'] = str(path.relative_to(assets.root))
    else:
        proof['compiled_scene'] = dict(path=roles['station'], sha256=sha(changed_scene))
        path = changed_scene.with_suffix('.proof.json')
        path.write_text(json.dumps(proof))
        config['source_geometry_proof_sha256'] = sha(path)
        roles['source_geometry_proof'] = str(path.relative_to(assets.root))
        config['prepared_scene_sha256' if spec.task_id == 'CP06' else 'scene_sha256'] = sha(changed_scene)
    changed = replace(spec, config=config, assets=roles)
    with pytest.raises(ValueError):
        validate(changed, LocalAssets(assets.root, roles, spec=changed))


def test_all150_original_geometry_is_admitted_without_rewriting_source(delivery):
    task, root = delivery
    paths = sorted(root.glob('test-cases/'+task+'/*/*/spec.json'))
    assert len(paths) == 150
    for path in paths:
        spec = InstanceSpec.from_dict(json.loads(path.read_text()))
        validate(spec, LocalAssets(root, spec.assets, spec=spec))
