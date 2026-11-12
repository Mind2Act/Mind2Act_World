"""Nonfinite task matrices must reject even after all caller hashes are refreshed."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil

import pytest

from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.sim import LocalAssets
from test_original_geometry_authority import sha, validate


@pytest.fixture(scope='module', params=['CP06', 'CP08'])
def finite_actual(request, tmp_path_factory):
    value = os.environ.get('MIND2ACT_ORIGINAL_DELIVERIES_ROOT')
    if not value:
        pytest.skip('private original deliveries not configured')
    root = tmp_path_factory.mktemp(request.param+'-finite')/'delivery'
    shutil.copytree(Path(value)/request.param, root, copy_function=os.link)
    path = root/'test-cases'/request.param/'hard'/(request.param+'_hard_clean_000')/'spec.json'
    spec = InstanceSpec.from_dict(json.loads(path.read_text()))
    assets = LocalAssets(root, spec.assets, spec=spec)
    validate(spec, assets)
    return spec, assets


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')],
                         ids=['nan', 'positive_inf', 'negative_inf'])
def test_rehashed_nonfinite_transform_is_not_masked(finite_actual, tmp_path, bad):
    from pxr import Usd, Sdf
    spec, assets = finite_actual
    config, roles = to_json(spec.config), dict(spec.assets)
    task, scene = assets.resolve('task_source'), assets.resolve('station')
    layer = Sdf.Layer.OpenAsAnonymous(str(task))
    if spec.task_id == 'CP06':
        path = spec.config['source_task']['all_objects'][0]['path']+'.xformOp:transform'
    else:
        case = spec.config['source_case']
        prism = next(b for b in [*case['blocks'], *case['distractors']]
                     if b['shape'] == 'triangular_prism')
        path = '/World/CP11/Blocks/'+prism['id']+'/Normalized/Asset.xformOp:transform'
    attr = layer.GetAttributeAtPath(path)
    matrix = attr.default
    matrix[0, 0] = bad
    attr.default = matrix
    changed_task = task.with_name(tmp_path.name+'-finite-task.usda')
    layer.Export(str(changed_task))
    # Read back the serialized mutation before trusting the rejection test.
    reopened = Sdf.Layer.OpenAsAnonymous(str(changed_task))
    observed = reopened.GetAttributeAtPath(path).default[0][0]
    assert math.isnan(observed) if math.isnan(bad) else observed == bad
    changed_scene = scene.with_name(tmp_path.name+'-finite-scene.usda')
    wrapper = Sdf.Layer.OpenAsAnonymous(str(scene))
    wrapper.subLayerPaths = [os.path.relpath(changed_task, scene.parent), wrapper.subLayerPaths[1]]
    wrapper.Export(str(changed_scene))
    roles['task_source'], roles['station'] = (str(p.relative_to(assets.root)) for p in (changed_task, changed_scene))
    proof = json.loads(assets.resolve('source_geometry_proof').read_text())
    proof['actual_composed_task'] = dict(path=roles['task_source'], sha256=sha(changed_task))
    proof['compiled_scene'] = dict(path=roles['station'], sha256=sha(changed_scene))
    proof_path = changed_scene.with_suffix('.proof.json')
    proof_path.write_text(json.dumps(proof))
    roles['source_geometry_proof'] = str(proof_path.relative_to(assets.root))
    config['source_geometry_proof_sha256'] = sha(proof_path)
    config['prepared_scene_sha256' if spec.task_id == 'CP06' else 'scene_sha256'] = sha(changed_scene)
    if spec.task_id == 'CP06':
        config['derived_task_layer_sha256'] = sha(changed_task)
    changed = replace(spec, config=config, assets=roles)
    with pytest.raises(ValueError):
        validate(changed, LocalAssets(assets.root, roles, spec=changed))


@pytest.mark.parametrize('value,shape', [
    (float('nan'), ()), (float('inf'), ()), (float('-inf'), ()),
    (None, ()), ('nan', ()), (True, ()), ([1., 2.], (3,)),
    ([[1., 0.], [0., 'nan']], (2, 2)),
])
def test_finite_values_reject_nonfinite_and_malformed_numeric_shapes(value, shape):
    from mind2act_eval.source_geometry import finite_value
    with pytest.raises(ValueError, match='finite numeric'):
        finite_value(value, shape)


def test_finite_values_accept_usd_matrix_and_vector_array_and_reject_missing_default():
    from pxr import Usd, Gf, Vt
    from mind2act_eval.source_geometry import finite_value, finite_default
    matrix = Gf.Matrix4d(1.)
    assert finite_value(matrix, (4, 4)) == matrix
    points = Vt.Vec3fArray([Gf.Vec3f(1., 2., 3.)]*9)
    assert finite_value(points, (9, 3)) == points
    with pytest.raises(ValueError, match='missing or malformed'):
        finite_default({'/Body.mass': {}}, '/Body.mass')
