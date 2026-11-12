"""Permission, binding and real USD composition regressions (CPU, no Kit)."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import MappingProxyType

import pytest
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from mind2act_eval.contracts import BuildContext, InstanceSpec, VariationSpec, to_json
from mind2act_eval.variations import (AssetPin, VariationPlan, VariationSession, VariationTargets,
    apply, base_spec_digest, digest, resolve, station_targets)
from mind2act_eval.variations.layouts import Layout, SettledObject, read_layout
from mind2act_eval.variations.stage import verify_settled


def spec():
    return InstanceSpec('PR1', 'slot1', 'easy', 'source', 'v1', 'assets', 'v1',
                        config={'target': 4, 'camera': [1, 2, 3]})


def fixture_plan(tmp_path, *, objects=True):
    base = spec()
    overlay = VariationSpec('random', 'layout1', '1'*64, base_spec_digest(base), '2'*64)
    source = Usd.Stage.CreateNew(str(tmp_path / 'clutter.usda'))
    root = UsdGeom.Cube.Define(source, '/Object')
    root.CreateSizeAttr(.01)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.CollisionAPI.Apply(root.GetPrim())
    source.SetDefaultPrim(root.GetPrim())
    source.GetRootLayer().Save()
    from mind2act_eval.variations import file_digest
    asset = AssetPin(tmp_path / 'clutter.usda', file_digest(tmp_path/'clutter.usda'),
                     MappingProxyType({'delivered_physics': {'type': 'rigid'}, 'dependencies': []}))
    mdl = tmp_path / 'test.mdl'
    mdl.write_text('mdl 1.3; export material test() = material();')
    material = AssetPin(mdl, file_digest(mdl), MappingProxyType({'mdl_name': 'test', 'dependencies': []}))
    item = SettledObject('box_00000_0', 'box/00000', (.3, .2, .8), (1., 0., 0., 0.),
                         (1., 1., 1.), MappingProxyType({'type': 'rigid'}))
    layout = Layout('PR1', 'easy', 'slot1', (item,) if objects else (),
                    {'table_enabled': True, 'material': {'id': 'table'}, 'light_enabled': True,
                     'light': {'type': 'Disk', 'color_rgb': (1.6, .2, .3), 'intensity_scale': 1.1}},
                    'floor', {'settle': {'steps': 600}})
    return VariationPlan(overlay, base_spec_digest(base), layout,
                         {'box/00000': asset, 'table': material, 'floor': material},
                         {'forbidden_aabbs': [], 'initialization_pose_tolerance_m': .002,
                          'initialization_pose_tolerance_rad': math.radians(2)})


def runtime_targets(tmp_path, **kwargs):
    # Each direct CPU apply models its own attempt output directory.
    from uuid import uuid4
    return station_targets(runtime_layer=tmp_path / str(uuid4()) / "variation.usda", **kwargs)


def stage():
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, '/World')
    for name in ('Worktable', 'Floor', 'Task', 'RobotLeft'):
        UsdGeom.Cube.Define(s, '/World/' + name)
    UsdGeom.Camera.Define(s, '/World/HeadCamera')
    key = UsdLux.RectLight.Define(s, '/World/Softbox')
    key.CreateIntensityAttr(500)
    key.CreateWidthAttr(2)
    key.CreateHeightAttr(2)
    UsdLux.DomeLight.Define(s, '/World/Studio').CreateIntensityAttr(400)
    pm = UsdShade.Material.Define(s, '/World/Physics')
    UsdPhysics.MaterialAPI.Apply(pm.GetPrim()).CreateStaticFrictionAttr(.7)
    UsdShade.MaterialBindingAPI.Apply(s.GetPrimAtPath('/World/Worktable')).Bind(pm, materialPurpose='physics')
    return s


def test_clean_is_explicit_no_layout_and_does_not_resolve_assets():
    s = spec()
    assert s.variation == VariationSpec()
    class NoAssets:
        def resolve(self, _):
            pytest.fail('clean must not access variation assets')
    plan = resolve(s, NoAssets())
    assert plan.layout is None
    assert InstanceSpec.from_dict(to_json(s)) == s
    for kwargs in ({'layout_id': 'x'}, {'mode': 'random'}, {'mode': 'random', 'layout_id': 'x',
                   'layout_sha256': '../escape', 'base_spec_sha256': '1'*64, 'asset_manifest_sha256': '2'*64}):
        with pytest.raises(ValueError):
            VariationSpec(**kwargs)


def test_clean_apply_creates_no_runtime_file(tmp_path):
    clean = resolve(spec(), None)
    targets = station_targets(runtime_layer=tmp_path/'new-output'/'variation.usda')
    s = stage()
    before = s.Flatten().ExportToString()
    assert apply(clean, s, targets)['authored'] is False
    assert not (tmp_path/'new-output').exists()
    assert s.Flatten().ExportToString() == before


def test_random_requires_fresh_explicit_runtime_anchor(tmp_path):
    plan = fixture_plan(tmp_path)
    with pytest.raises(ValueError, match='builder-owned runtime layer'):
        apply(plan, stage(), station_targets())
    path = tmp_path/'existing.usda'
    path.write_text('retained previous attempt')
    with pytest.raises(FileExistsError):
        apply(plan, stage(), station_targets(runtime_layer=path))
    assert path.read_text() == 'retained previous attempt'


def test_builder_seam_anchors_layer_in_its_attempt_output(tmp_path):
    from types import SimpleNamespace
    from mind2act_eval.variations import prepare_station
    session = VariationSession(fixture_plan(tmp_path))
    ctx = BuildContext(SimpleNamespace(stage=stage()), None, None, tmp_path/'attempt', session)
    prepare_station(ctx)
    assert (ctx.output_dir/'variation.usda').is_file()
    assert session.receipt['runtime_layer'] == 'variation.usda'
    assert session.receipt['protected_diff'] == []


@pytest.mark.parametrize('task_id', ['P2','CP10'])
@pytest.mark.parametrize('mode', ['clean','random'])
def test_new15_builders_compose_variation_before_native_world(tmp_path, monkeypatch, task_id, mode):
    import importlib
    import sys
    from types import ModuleType, SimpleNamespace
    import pxr
    product = importlib.import_module('mind2act_eval.tasks.'+task_id+'.scene')
    plan = fixture_plan(tmp_path)
    base = replace(spec(),task_id=task_id)
    overlay = replace(plan.spec,base_spec_sha256=base_spec_digest(base))
    plan = replace(plan,spec=overlay,instance_sha256=base_spec_digest(base),
                   layout=replace(plan.layout,task=task_id))
    instance = replace(base,variation=overlay) if mode=='random' else base
    s = stage()
    task = s.GetPrimAtPath('/World/Task')
    UsdGeom.Cube(task).CreateSizeAttr(.01)
    UsdGeom.Xformable(task).AddTranslateOp().Set(Gf.Vec3d(2,2,2))
    for path in ('/World/Task/Ball0','/World/Task/Lid','/World/Task/Stove/PreplacedTier'):
        UsdGeom.Xform.Define(s,path)
    native = ModuleType('isaacsim.core.prims')
    for name in ('SingleRigidPrim','RigidPrim','SingleArticulation'):
        setattr(native,name,object)
    monkeypatch.setitem(sys.modules,'isaacsim.core.prims',native)
    native_types = ModuleType('isaacsim.core.utils.types')
    native_types.ArticulationAction = object
    monkeypatch.setitem(sys.modules,'isaacsim.core.utils.types',native_types)
    physx = ModuleType('omni.physx')
    physx.get_physx_interface = lambda:None
    physx.get_physx_simulation_interface = lambda:None
    monkeypatch.setitem(sys.modules,'omni.physx',physx)
    bindings = ModuleType('omni.physx.bindings._physx')
    bindings.ContactEventType = object
    monkeypatch.setitem(sys.modules,'omni.physx.bindings._physx',bindings)
    for name in ('PhysxSchema','PhysicsSchemaTools'):
        monkeypatch.setattr(pxr,name,ModuleType('pxr.'+name),raising=False)
    if task_id=='P2':
        monkeypatch.setattr(product,'validate',lambda spec,ctx:({},dict(target_initial_xyz={'Ball0':[0,0,0]},decoy_xyz={})))
        monkeypatch.setattr(product,'author_task',lambda *args:None)
        monkeypatch.setattr(product,'prepare_task_contacts',lambda *args:None)
    else:
        monkeypatch.setattr(product,'validate_binding',lambda *args:{})
    class NativeWorldBoundary(Exception): pass
    class Sim:
        stage = s
        @property
        def world(self):
            raise NativeWorldBoundary()
    output=tmp_path/'episode'
    session=VariationSession(plan) if mode=='random' else None
    ctx=BuildContext(Sim(),SimpleNamespace(prepare_stage=lambda stage:None),None,output,session)
    before=s.Flatten().ExportToString()
    with pytest.raises(NativeWorldBoundary):
        product.build(instance,ctx)
    if mode=='random':
        assert session.receipt is not None, 'builder reached native world without applying its variation seam'
        assert session.receipt['protected_diff'] == []
        assert s.GetPrimAtPath('/World/EvalVariation/box_00000_0')
        assert (output/'variation.usda').is_file()
    else:
        assert s.Flatten().ExportToString()==before
        assert not output.exists()


def test_relative_variation_and_session_resolve_after_bundle_move(tmp_path):
    import subprocess
    import sys
    from mind2act_eval.sim.asset_relocation import audit_generated_layers, export_relative_layer
    bundle = tmp_path/'original'
    bundle.mkdir()
    plan = fixture_plan(bundle)
    s = stage()
    root_path = bundle/'station.usda'
    s.GetRootLayer().Export(str(root_path))
    s = Usd.Stage.Open(str(root_path))
    targets = station_targets(runtime_layer=bundle/'runtime'/'variation.usda')
    receipt = apply(plan, s, targets)
    assert receipt['generated_layer_portability']['passed']
    session_path = export_relative_layer(s.GetSessionLayer(), bundle/'runtime'/'session.usda')
    assert audit_generated_layers([targets.runtime_layer, session_path])['passed']
    text = targets.runtime_layer.read_text()
    assert '@../clutter.usda@' in text and '@../test.mdl@' in text
    moved = tmp_path/'moved'
    bundle.rename(moved)
    # Fresh process: the source path and USD layer registry cannot satisfy it.
    script = '''from pathlib import Path
import sys
from pxr import Sdf, Usd, UsdShade
r=Path(sys.argv[1])
s=Usd.Stage.Open(Sdf.Layer.FindOrOpen(str(r/'station.usda')), Sdf.Layer.FindOrOpen(str(r/'runtime/session.usda')))
assert s.GetPrimAtPath('/World/EvalVariation/box_00000_0').GetAttribute('size').Get() == .01
shader=UsdShade.Shader(s.GetPrimAtPath('/World/EvalVariation/Materials/floor/Shader'))
assert Path(shader.GetSourceAsset('mdl').resolvedPath) == r/'test.mdl'
'''
    subprocess.run([sys.executable, '-c', script, str(moved)], check=True)


def test_overlay_identity_and_exact_base_binding():
    s = spec()
    v = VariationSpec('random', 'one', '1'*64, base_spec_digest(s), '2'*64)
    random = replace(s, variation=v)
    assert base_spec_digest(random) == base_spec_digest(s)
    assert digest(random) != digest(s)
    with pytest.raises(ValueError, match='frozen clean'):
        resolve(replace(random, seed=1), None)
    with pytest.raises(ValueError, match='binding mismatch'):
        random.assert_binding(replace(random, variation=replace(v, layout_id='two')))


def test_composition_keeps_physics_camera_robot_and_task(tmp_path):
    s = stage()
    plan = fixture_plan(tmp_path)
    originals = {p: s.GetPrimAtPath(p).GetAllMetadata() for p in ('/World/Task', '/World/RobotLeft', '/World/HeadCamera')}
    receipt = apply(plan, s, runtime_targets(tmp_path))
    assert receipt['protected_diff'] == []
    assert receipt['protected_before_sha256'] == receipt['protected_after_sha256']
    assert receipt['physics_steps_added'] == 0
    assert s.GetPrimAtPath('/World/Physics').GetAttribute('physics:staticFriction').Get() == pytest.approx(.7)
    physics = UsdShade.MaterialBindingAPI(s.GetPrimAtPath('/World/Worktable')).ComputeBoundMaterial(materialPurpose='physics')[0]
    assert str(physics.GetPath()) == '/World/Physics'
    for purpose in ('preview', 'full'):
        visual = UsdShade.MaterialBindingAPI(s.GetPrimAtPath('/World/Worktable')).ComputeBoundMaterial(materialPurpose=purpose)[0]
        assert str(visual.GetPath()) == '/World/EvalVariation/Materials/table'
    assert s.GetPrimAtPath('/World/Softbox').GetTypeName() == 'DiskLight'
    assert s.GetPrimAtPath('/World/Softbox').GetAttribute('inputs:color').Get()[0] == pytest.approx(1.6)
    for p, value in originals.items():
        assert s.GetPrimAtPath(p).GetAllMetadata() == value
    assert verify_settled(plan, s, runtime_targets(tmp_path))['readback'] == 'composed_usd'
    with pytest.raises(ValueError, match='already exists'):
        apply(plan, s, runtime_targets(tmp_path))


def test_default_render_binding_changes_without_changing_physics(tmp_path):
    s = stage()
    original = UsdShade.Material.Define(s, '/World/OriginalRenderMaterial')
    for name in ('Floor', 'Worktable'):
        UsdShade.MaterialBindingAPI.Apply(s.GetPrimAtPath('/World/'+name)).Bind(original)
    receipt = apply(fixture_plan(tmp_path, objects=False), s, runtime_targets(tmp_path))
    assert receipt['protected_diff'] == []
    for surface, role in [('Floor','floor'), ('Worktable','table')]:
        material = UsdShade.MaterialBindingAPI(s.GetPrimAtPath('/World/'+surface)).ComputeBoundMaterial()[0]
        assert str(material.GetPath()) == '/World/EvalVariation/Materials/'+role
    physics = UsdShade.MaterialBindingAPI(s.GetPrimAtPath('/World/Worktable')).ComputeBoundMaterial(materialPurpose='physics')[0]
    assert str(physics.GetPath()) == '/World/Physics'
    assert s.GetPrimAtPath('/World/Physics').GetAttribute('physics:staticFriction').Get() == pytest.approx(.7)


def test_all_purpose_physics_material_cannot_be_displaced(tmp_path):
    s = stage()
    table = s.GetPrimAtPath('/World/Worktable')
    table.RemoveProperty('material:binding:physics')
    UsdShade.MaterialBindingAPI(table).Bind(UsdShade.Material(s.GetPrimAtPath('/World/Physics')))
    before = s.Flatten().ExportToString()
    with pytest.raises(ValueError, match='protected stage'):
        apply(fixture_plan(tmp_path, objects=False), s, runtime_targets(tmp_path))
    assert s.Flatten().ExportToString() == before


@pytest.mark.parametrize('overwrite', ['layer', 'light'])
def test_post_initialization_detects_empty_layout_visual_loss(tmp_path, overwrite):
    s = stage()
    session = VariationSession(fixture_plan(tmp_path, objects=False))
    session.apply(s, runtime_targets(tmp_path))
    if overwrite == 'layer':
        s.GetSessionLayer().subLayerPaths = []
    else:
        with Usd.EditContext(s, s.GetSessionLayer()):
            s.GetPrimAtPath('/World/Softbox').GetAttribute('inputs:intensity').Set(500.)
    with pytest.raises(ValueError, match='variation (visual|layer)'):
        session.verify()


@pytest.mark.parametrize('clobber_on_ready,settled_status', [(False,'verified'), (True,'verified'), (False,'UNKNOWN')])
def test_runner_result_records_effective_variation_and_existing_first_frame(tmp_path, monkeypatch, clobber_on_ready, settled_status):
    import json
    import numpy as np
    from PIL import Image
    from mind2act_eval import variations
    from mind2act_eval.contracts import Health, NativeEvidence, Observation, Score, ConsumerKind
    from mind2act_eval.runner import run_episode, EpisodeLimits
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.sim.demo import DemoRobot
    from mind2act_eval.tasks import TaskDefinition
    plan = fixture_plan(tmp_path, objects=False)
    random = replace(spec(), variation=plan.spec)
    s = stage()
    class CpuSession(VariationSession):
        def verify(self, *, native=False, stage=None):
            receipt = super().verify(native=False, stage=stage)
            receipt['post_initialization']['status'] = settled_status
            return receipt
    monkeypatch.setattr(variations, 'resolve', lambda spec, assets: plan)
    monkeypatch.setattr(variations, 'VariationSession', CpuSession)
    pixels = np.full((8, 12, 3), [17, 43, 91], dtype=np.uint8)
    class Sim:
        stage = s
        world = None
        artifacts = {}
        def step(self, *, render):
            pass
        def ready(self):
            if clobber_on_ready:
                original = UsdShade.Material.Define(s, '/World/ResetMaterial')
                with Usd.EditContext(s, s.GetSessionLayer()):
                    UsdShade.MaterialBindingAPI(s.GetPrimAtPath('/World/Worktable')).Bind(original)
        def close(self):
            pass
    robot = DemoRobot()
    class Scene:
        def before_step(self, tick): pass
        def after_step(self, tick): pass
        def observe(self, tick):
            return Observation(tick, tick/120, **{**robot.observe(),'rgb':{'head':pixels}})
        def evidence(self, tick): return NativeEvidence(random.task_id, random.instance_id, tick, {})
        def health(self): return Health(True)
        def is_complete(self): return False
        def result_metadata(self): return {'clock_profile':'preserved'}
        def close(self): pass
    def build(spec, ctx):
        variations.prepare_station(ctx)
        return Scene()
    output = tmp_path/'episode'
    output.mkdir()
    ctx = BuildContext(Sim(), robot, None, output)
    result = run_episode(random, TaskDefinition('PR1',build,lambda e:Score(100.)),ctx,DemoAdapter(),
                         consumer_kind=ConsumerKind.DIAGNOSTIC,limits=EpisodeLimits(4,10))
    saved = json.loads((output/'result.json').read_text())
    if clobber_on_ready:
        assert not result.infra_valid and 'variation visual binding not effective' in result.error
        assert saved['task_metadata']['variation']['applied'] is False
        assert saved['task_metadata']['variation']['post_initialization_visual'] is None
        assert json.loads((output/'variation-receipt.json').read_text())['status'] == 'failed'
        assert 'first_frame_head' not in saved['artifacts']
        return
    if settled_status != 'verified':
        assert saved['task_metadata']['variation']['applied'] is False
        assert not result.infra_valid and result.score is None
        assert 'variation_not_applied' in result.error
        return
    assert result.infra_valid, result.error
    metadata = saved['task_metadata']
    assert metadata['clock_profile'] == 'preserved'
    assert metadata['variation']['layout_id'] == random.variation.layout_id
    assert metadata['variation']['layout_sha256'] == random.variation.layout_sha256
    assert metadata['variation']['receipt'] == 'variation-receipt.json'
    assert metadata['variation']['applied'] is True
    assert metadata['variation']['post_initialization_visual']['status'] == 'verified'
    assert metadata['variation']['post_initialization_visual']['pixels_verified'] is False
    assert saved['artifacts']['first_frame_head'] == 'first-frame/head.png'
    assert np.array_equal(np.asarray(Image.open(output/'first-frame/head.png')), pixels)


def test_builder_explicitly_declares_source_without_a_table(tmp_path):
    s = stage()
    s.RemovePrim('/World/Worktable')
    receipt = apply(fixture_plan(tmp_path, objects=False), s, runtime_targets(tmp_path, table_surfaces=()))
    assert receipt['protected_diff'] == []
    assert receipt['skipped_visuals'] == [{'role':'table', 'reason':'builder declares no table in the clean scene'}]
    assert not s.GetPrimAtPath('/World/Worktable')


def test_light_type_change_preserves_explicit_source_metadata(tmp_path):
    s = stage()
    light = s.GetPrimAtPath('/World/Softbox')
    light.SetDocumentation('frozen authored source note')
    assert apply(fixture_plan(tmp_path), s, runtime_targets(tmp_path))['protected_diff'] == []
    assert light.GetDocumentation() == 'frozen authored source note'


@pytest.mark.parametrize('target', ['/World/Task', '/World/RobotLeft', '/World/HeadCamera', '/World'])
def test_builder_cannot_grant_protected_targets(target):
    with pytest.raises(ValueError):
        VariationTargets('/World/EvalVariation', (target,), ('/World/Floor',), '/World/Softbox',
                         '/World/Studio', ('/World/Task', '/World/RobotLeft', '/World/HeadCamera'))


def test_source_exclusion_overlap_warns_without_changing_frozen_layout(tmp_path):
    s = stage()
    plan = fixture_plan(tmp_path)
    plan = replace(plan, binding={**plan.binding, 'forbidden_aabbs': [[[.2, .1, .7], [.4, .3, .9]]]})
    receipt = apply(plan, s, runtime_targets(tmp_path))
    assert receipt['clutter'][0]['warnings'] == [{
        'kind': 'source_exclusion_overlap', 'zone_index': 0,
        'severity': 'warning', 'geometry_intersection_established': False}]
    assert receipt['clutter'][0]['position'] == plan.layout.objects[0].position
    assert receipt['protected_diff'] == []


def test_failed_apply_rolls_back_own_layer(tmp_path):
    s = stage()
    plan = fixture_plan(tmp_path)
    s.RemovePrim('/World/Softbox')
    before = s.Flatten().ExportToString()
    with pytest.raises(ValueError):
        apply(plan, s, runtime_targets(tmp_path))
    assert s.Flatten().ExportToString() == before
    assert not s.GetPrimAtPath('/World/EvalVariation')
    failed = list(tmp_path.glob('*/variation.usda'))
    assert len(failed) == 1 and not s.GetSessionLayer().subLayerPaths
    assert 'box_00000_0' in failed[0].read_text()


def test_exclusions_use_rotated_footprint_rather_than_larger_world_aabb(tmp_path):
    from mind2act_eval.variations import file_digest
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    source = Usd.Stage.Open(str(pin.path))
    UsdGeom.Cube(source.GetDefaultPrim()).CreateSizeAttr(1)
    source.GetRootLayer().Save()
    pin = replace(pin, sha256=file_digest(pin.path))
    item = replace(plan.layout.objects[0], scale=(1,.02,.02),
                   orientation=(math.cos(math.pi/8),0,0,math.sin(math.pi/8)))
    plan = replace(plan, assets={**plan.assets,'box/00000':pin}, layout=replace(plan.layout,objects=(item,)),
                   binding={**plan.binding,'forbidden_aabbs':[[[-.04,.48,.7],[0,.52,.9]]]})
    assert apply(plan, stage(), runtime_targets(tmp_path))['protected_diff'] == []
    plan = replace(plan,binding={**plan.binding,'forbidden_aabbs':[[[.29,.19,.7],[.31,.21,.9]]]})
    receipt = apply(plan,stage(),runtime_targets(tmp_path))
    assert receipt['clutter'][0]['warnings'][0]['kind'] == 'source_exclusion_overlap'


def test_asset_cannot_import_camera_or_joint(tmp_path):
    plan = fixture_plan(tmp_path)
    p = plan.assets['box/00000'].path
    source = Usd.Stage.Open(str(p))
    UsdGeom.Camera.Define(source, '/Object/BadCamera')
    source.GetRootLayer().Save()
    from mind2act_eval.variations import file_digest
    pin = replace(plan.assets['box/00000'], sha256=file_digest(p))
    plan = replace(plan, assets={**plan.assets, 'box/00000': pin})
    with pytest.raises(ValueError, match='camera/light/joint'):
        apply(plan, stage(), runtime_targets(tmp_path))


def test_archived_mass_applies_only_to_new_distractor_without_friction_writes(tmp_path):
    plan = fixture_plan(tmp_path)
    item = replace(plan.layout.objects[0], physics={'type': 'rigid', 'mass': .012, 'friction': .45})
    plan = replace(plan, layout=replace(plan.layout, objects=(item,)))
    s = stage()
    receipt = apply(plan, s, runtime_targets(tmp_path))
    body = s.GetPrimAtPath('/World/EvalVariation/box_00000_0')
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(.012)
    assert tuple(UsdPhysics.RigidBodyAPI(body).GetVelocityAttr().Get()) == (0, 0, 0)
    assert receipt['protected_diff'] == []
    assert not any(p.HasAPI(UsdPhysics.MaterialAPI) for p in Usd.PrimRange(s.GetPrimAtPath('/World/EvalVariation')))
    assert s.GetPrimAtPath('/World/Physics').GetAttribute('physics:staticFriction').Get() == pytest.approx(.7)


def test_pinned_sdk_mdl_resolves_without_global_resolver_changes(tmp_path):
    from pxr import Sdf
    from mind2act_eval.variations import file_digest
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    modules = tmp_path/'modules'
    (modules/'gltf').mkdir(parents=True)
    module = modules/'gltf/pbr.mdl'
    module.write_text('mdl 1.8; export material pbr() = material();')
    source = Usd.Stage.Open(str(pin.path))
    source.GetDefaultPrim().CreateAttribute('module', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('gltf/pbr.mdl'))
    source.GetRootLayer().Save()
    pin = replace(pin, sha256=file_digest(pin.path), search_paths=(modules,),
                  dependencies=((module, file_digest(module)),))
    plan = replace(plan, assets={**plan.assets, 'box/00000': pin})
    assert apply(plan, stage(), runtime_targets(tmp_path))['protected_diff'] == []
    module.write_text('changed module')
    with pytest.raises(ValueError, match='dependency changed'):
        apply(plan, stage(), runtime_targets(tmp_path))


def test_exported_viewport_siblings_are_not_imported_or_treated_as_capabilities(tmp_path):
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    source = Usd.Stage.Open(str(pin.path))
    UsdGeom.Camera.Define(source, '/ExportedViewportCamera')
    source.DefinePrim('/Render/Products').CreateRelationship('camera').SetTargets(['/ExportedViewportCamera'])
    source.GetRootLayer().Save()
    from mind2act_eval.variations import file_digest
    pin = replace(pin, sha256=file_digest(pin.path))
    plan = replace(plan, assets={**plan.assets, 'box/00000': pin})
    s = stage()
    assert apply(plan, s, runtime_targets(tmp_path))['protected_diff'] == []
    assert not s.GetPrimAtPath('/Render')
    assert not s.GetPrimAtPath('/ExportedViewportCamera')


def test_delivered_float_transform_ops_do_not_collide_with_overlay_pose(tmp_path):
    from mind2act_eval.variations import file_digest
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    source = Usd.Stage.Open(str(pin.path))
    xform = UsdGeom.Xformable(source.GetDefaultPrim())
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(2))
    source.GetRootLayer().Save()
    pin = replace(pin, sha256=file_digest(pin.path))
    plan = replace(plan, assets={**plan.assets, 'box/00000': pin})
    s = stage()
    assert apply(plan, s, runtime_targets(tmp_path))['protected_diff'] == []
    assert verify_settled(plan, s, runtime_targets(tmp_path))['status'] == 'verified'
    assert source.GetDefaultPrim().GetAttribute('xformOp:orient').GetTypeName() == 'quatf'


def test_session_requires_exactly_one_builder_seam(tmp_path):
    session = VariationSession(fixture_plan(tmp_path))
    with pytest.raises(ValueError, match='no variation'):
        session.verify()
    session.apply(stage(), runtime_targets(tmp_path))
    with pytest.raises(ValueError, match='twice'):
        session.apply(stage(), runtime_targets(tmp_path))

@pytest.mark.parametrize('path,attribute,value', [
    ('/World/Physics', 'physics:staticFriction', .1),
    ('/World/HeadCamera', 'focalLength', 1.),
    ('/World/Task', 'size', 2.5),
    ('/World/RobotLeft', 'size', 3.),
])
def test_protected_mutation_is_rejected_and_rolled_back(tmp_path, monkeypatch, path, attribute, value):
    from mind2act_eval.variations import stage as author
    original = author._appearance
    def inject(s, plan, targets):
        result = original(s, plan, targets)
        s.GetPrimAtPath(path).GetAttribute(attribute).Set(value)
        return result
    monkeypatch.setattr(author, '_appearance', inject)
    s = stage()
    before = s.Flatten().ExportToString()
    with pytest.raises(ValueError, match='protected stage'):
        apply(fixture_plan(tmp_path), s, runtime_targets(tmp_path))
    assert s.Flatten().ExportToString() == before


def write_resolvable(tmp_path):
    from mind2act_eval.variations import file_digest
    base = spec()
    payload = {'schema': 'mind2act.random-layout.v1', 'status': 'ok', 'task': 'PR1',
               'difficulty': 'easy', 'logical_slot': 'native_slot', 'Rigid': {},
               'Appearance': {'schema': 'mind2act.visual-randomization.v2', 'mode': 'random',
                    'task': 'PR1', 'difficulty': 'easy', 'table_enabled': False, 'light_enabled': False,
                    'ground': 'floor', 'light': {'type': 'Dome', 'color_rgb': [1.2, 0., .2], 'intensity_scale': 1.}}}
    payload['layout_sha256'] = digest(payload['Rigid'])
    (tmp_path/'layout.json').write_text(json.dumps(payload))
    (tmp_path/'floor.mdl').write_text('mdl 1.3; export material test() = material();')
    binding = {'base_spec_sha256': base_spec_digest(base), 'task_id': 'PR1', 'instance_id': 'slot1',
               'difficulty': 'easy', 'seed': 0, 'path': 'layout.json', 'layout_task': 'PR1',
               'layout_slot': 'native_slot', 'coordinates': {'settled_frame': 'world', 'units': 'm', 'quaternion': 'wxyz'},
               'generator_version': 'source-v1', 'baseline_sha256': digest(base.config),
               'settled_receipt_sha256': digest(None), 'exclusion_receipt_sha256': digest([]), 'forbidden_aabbs': [],
               'initialization_pose_tolerance_m': .002, 'initialization_pose_tolerance_rad': .02}
    manifest = {'schema': 'mind2act_eval.variation_manifest.v1', 'layouts': {'layout1': binding},
                'assets': {'floor': {'path': 'floor.mdl', 'sha256': file_digest(tmp_path/'floor.mdl'),
                                     'dependencies': [], 'mdl_name': 'test'}}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    overlay = VariationSpec('random', 'layout1', file_digest(tmp_path/'layout.json'),
                            base_spec_digest(base), file_digest(tmp_path/'manifest.json'))
    class Assets:
        def resolve(self, asset_id):
            assert asset_id == 'variation_manifest'
            return tmp_path/'manifest.json'
    return replace(base, variation=overlay), Assets(), manifest


def test_resolve_binds_every_input_and_rejects_changed_bytes(tmp_path):
    s, assets, _ = write_resolvable(tmp_path)
    plan = resolve(s, assets)
    assert plan.layout.slot == 'native_slot'
    assert plan.layout.appearance['light']['color_rgb'][0] == 1.2
    (tmp_path/'layout.json').write_text('{}')
    with pytest.raises(ValueError, match='digest mismatch'):
        resolve(s, assets)


def test_archived_effective_mass_and_settled_pose_are_loaded_without_sampling(tmp_path):
    s, _, _ = write_resolvable(tmp_path)
    payload = json.loads((tmp_path/'layout.json').read_text())
    payload['Rigid'] = {'box': [dict(category_idx=0, clutter_idx=0, default_pos=[0,0,1],
        default_ori=[1,0,0,0], settled_pos=[.3,.2,.8], settled_ori=[1,0,0,0], scale=[1,1,1],
        physics={'type':'rigid', 'mass':1.5}, relative_plane='Table', type='cluttered', yaml_path='Clutter/clutter.yml')]}
    payload['layout_sha256'] = digest(payload['Rigid'])
    payload['settle'] = dict(steps=300, any_below_table=False, bodies=[dict(name='box_00000_0',
        verdict='stable', fell=False, below_table=False, final_pos=[.3,.2,.8], final_ori=[1,0,0,0], mass=.5)])
    layout = read_layout(payload, {'settled_frame':'world', 'units':'m', 'quaternion':'wxyz'})
    assert layout.objects[0].position == (.3,.2,.8)
    assert layout.objects[0].physics['mass'] == .5
    del payload['Rigid']['box'][0]['settled_pos']
    payload['layout_sha256'] = digest(payload['Rigid'])
    with pytest.raises(ValueError, match='missing settled pose'):
        read_layout(payload, {'settled_frame':'world', 'units':'m', 'quaternion':'wxyz'})


def test_resolve_pins_material_modules_and_preserves_baseline_roots(tmp_path, monkeypatch):
    from mind2act_eval.variations import file_digest, configure_material_search
    import os
    s, assets, manifest = write_resolvable(tmp_path)
    (tmp_path/'modules').mkdir()
    (tmp_path/'baseline').mkdir()
    module = tmp_path/'modules/shared.mdl'
    module.write_text('mdl 1.8;')
    manifest.update(mdl_search_paths=['modules'], mdl_files=[{'path':'modules/shared.mdl','sha256':file_digest(module)}])
    assets.resolve('variation_manifest').write_text(json.dumps(manifest))
    s = replace(s, variation=replace(s.variation,asset_manifest_sha256=file_digest(assets.resolve('variation_manifest'))))
    class Paths:
        def resolve(self, key):
            return tmp_path/'baseline' if key == 'baseline' else assets.resolve(key)
    monkeypatch.setenv('MDL_SYSTEM_PATH', '')
    monkeypatch.setenv('MDL_USER_PATH', '')
    receipt = configure_material_search(s, Paths(), ('baseline',))
    assert receipt['resolved_paths'] == [str(tmp_path/'baseline'), str(tmp_path/'modules')]
    assert os.environ['MDL_USER_PATH'].split(os.pathsep) == receipt['resolved_paths']
    module.write_text('changed')
    with pytest.raises(ValueError, match='module changed'):
        resolve(s, assets)


@pytest.mark.parametrize('mutation', ['path', 'slot', 'frame', 'baseline', 'seed'])
def test_manifest_cannot_redirect_inputs(tmp_path, mutation):
    s, assets, manifest = write_resolvable(tmp_path)
    b = manifest['layouts']['layout1']
    if mutation == 'path': b['path'] = '../missing-layout.json'
    if mutation == 'slot': b['layout_slot'] = 'other_answer'
    if mutation == 'frame': b['coordinates']['settled_frame'] = 'Table'
    if mutation == 'baseline': b['base_spec_sha256'] = '0'*64
    if mutation == 'seed': b['seed'] = 1
    assets.resolve('variation_manifest').write_text(json.dumps(manifest))
    from mind2act_eval.variations import file_digest
    s = replace(s, variation=replace(s.variation, asset_manifest_sha256=file_digest(assets.resolve('variation_manifest'))))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve(s, assets)


def test_random_expert_replay_requires_same_answer_slot_seed():
    from types import SimpleNamespace
    from mind2act_eval.replay import H5Replay
    clean = spec()
    replay = H5Replay.__new__(H5Replay)
    replay.metadata = SimpleNamespace(clock=clean.clock, source_id=clean.source_id, instance_spec=clean)
    random = replace(clean, variation=VariationSpec('random', 'layout1', '1'*64, base_spec_digest(clean), '2'*64))
    replay.assert_spec(random)
    for changed in (replace(random, seed=1), replace(random, instance_id='slot2'),
                    replace(random, config={'target': 7}), replace(random, variation=replace(random.variation, base_spec_sha256='0'*64))):
        with pytest.raises(ValueError):
            replay.assert_spec(changed)


def test_deployment_catalog_does_not_change_clean_source_identity(tmp_path):
    from types import SimpleNamespace
    from mind2act_eval.cli import _episode_assets
    s, _, _ = write_resolvable(tmp_path)
    clean = replace(s, variation=VariationSpec())
    before = to_json(clean)
    assert 'variation' not in before  # legacy clean source hash remains exact
    resolver = _episode_assets(s, SimpleNamespace(asset_root=str(tmp_path), variation_manifest='manifest.json'))
    assert resolver.resolve('variation_manifest') == tmp_path/'manifest.json'
    assert to_json(clean) == before
    assert resolve(s, resolver).instance_sha256 == base_spec_digest(clean)
    with pytest.raises(ValueError, match='clean'):
        _episode_assets(clean, SimpleNamespace(asset_root=str(tmp_path), variation_manifest='manifest.json'))
    with pytest.raises(ValueError, match='relative'):
        _episode_assets(s, SimpleNamespace(asset_root=str(tmp_path), variation_manifest=str(tmp_path/'manifest.json')))


def test_unpinned_usd_dependency_cannot_enter_clutter(tmp_path):
    from pxr import Sdf
    from mind2act_eval.variations import file_digest
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    (tmp_path/'texture.png').write_bytes(b'recorded texture')
    source = Usd.Stage.Open(str(pin.path))
    source.GetDefaultPrim().CreateAttribute('testTexture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('texture.png'))
    source.GetRootLayer().Save()
    pin = replace(pin, sha256=file_digest(pin.path))
    plan = replace(plan, assets={**plan.assets, 'box/00000': pin})
    with pytest.raises(ValueError, match='unpinned dependency'):
        apply(plan, stage(), runtime_targets(tmp_path))


@pytest.mark.parametrize('alias_type', ['hardlink', 'copy', 'mutated_hardlink'])
def test_sdk_dependency_hardlink_alias_keeps_exact_file_pin(tmp_path, alias_type):
    import shutil
    from pxr import Sdf
    from mind2act_eval.variations import file_digest
    from mind2act_eval.variations.stage import _asset_stage
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    declared = tmp_path/'sdk/modules/gltf/pbr.mdl'
    declared.parent.mkdir(parents=True)
    declared.write_text('mdl 1.3; export material pbr() = material();')
    expected = file_digest(declared)
    alias = tmp_path/'runtime-sdk/modules/gltf/pbr.mdl'
    alias.parent.mkdir(parents=True)
    if alias_type == 'copy':
        shutil.copyfile(declared, alias)
    else:
        alias.hardlink_to(declared)
    source = Usd.Stage.Open(str(pin.path))
    shader = UsdShade.Shader.Define(source, '/Object/Shader')
    shader.CreateImplementationSourceAttr('sourceAsset')
    shader.SetSourceAsset(Sdf.AssetPath(str(alias)), 'mdl')
    source.GetRootLayer().Save()
    pin = replace(pin, sha256=file_digest(pin.path), dependencies=((declared, expected),),
                  search_paths=(declared.parents[1],))
    if alias_type == 'mutated_hardlink':
        alias.write_text('mdl 1.3; export material changed() = material();')
        with pytest.raises(ValueError, match='dependency changed'):
            _asset_stage(pin)
    elif alias_type == 'copy':
        # Equal bytes alone must not grant another dependency path.
        with pytest.raises(ValueError, match='unpinned dependency'):
            _asset_stage(pin)
    else:
        assert alias.samefile(declared)
        assert _asset_stage(pin).GetDefaultPrim()


def test_frozen_denominator_has_no_replacement_for_missing_or_duplicate(tmp_path):
    from mind2act_eval.variations.audit import inspect
    _, _, _ = write_resolvable(tmp_path)
    path = tmp_path/'layout.json'
    expected = {('PR1', 'easy', 'native_slot'): {}, ('PR1', 'easy', 'missing'): {}}
    report = inspect([path, path], expected, coordinates={'settled_frame': 'world', 'units': 'm', 'quaternion': 'wxyz'})
    assert report['denominator'] == 2
    assert report['missing'] == [('PR1', 'easy', 'missing')]
    assert report['duplicates'] == [('PR1', 'easy', 'native_slot')]
    assert report['native_completed'] == 0


@pytest.mark.parametrize('task_id', ['DR3', 'CP06', 'CP07', 'CP08', 'PR5'])
@pytest.mark.parametrize('mode', ['clean', 'random'])
def test_additional_new15_builder_applies_overlay_before_world(tmp_path, monkeypatch, task_id, mode):
    import hashlib
    import importlib
    import sys
    from types import ModuleType, SimpleNamespace
    import pxr
    native = ModuleType('isaacsim.core.prims')
    for name in ('SingleRigidPrim', 'RigidPrim', 'SingleArticulation'):
        setattr(native, name, object)
    monkeypatch.setitem(sys.modules, 'isaacsim.core.prims', native)
    native_types = ModuleType('isaacsim.core.utils.types')
    native_types.ArticulationAction = object
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types', native_types)
    physx = ModuleType('omni.physx')
    physx.get_physx_interface = lambda: None
    physx.get_physx_simulation_interface = lambda: None
    monkeypatch.setitem(sys.modules, 'omni.physx', physx)
    bindings = ModuleType('omni.physx.bindings._physx')
    bindings.ContactEventType = object
    monkeypatch.setitem(sys.modules, 'omni.physx.bindings._physx', bindings)
    for name in ('PhysxSchema', 'PhysicsSchemaTools'):
        monkeypatch.setattr(pxr, name, ModuleType('pxr.' + name), raising=False)
    product = importlib.import_module('mind2act_eval.tasks.' + task_id + '.scene')
    source_assets = importlib.import_module('mind2act_eval.sim.source_assets')
    monkeypatch.setattr(source_assets, 'verify_source_file', lambda *args, **kwargs: None)
    texture = tmp_path / 'cue.png'
    texture.write_bytes(b'frozen cue texture')
    cue_hash = hashlib.sha256(texture.read_bytes()).hexdigest()
    if task_id == 'DR3':
        monkeypatch.setattr(product, 'validate', lambda *args: None)
    elif task_id == 'CP06':
        monkeypatch.setattr(product, 'validate_assets', lambda *args: dict(station_robots={}))
    elif task_id == 'CP07':
        monkeypatch.setattr(product, 'load', lambda *args: {})
    elif task_id == 'CP08':
        inputs = importlib.import_module('mind2act_eval.tasks.CP08.instances')
        monkeypatch.setattr(inputs, 'validate_assets', lambda *args: dict(scene_sha256='0' * 64,
            cue_texture_sha256={'plan.png': cue_hash, 'screen-idle.png': cue_hash}))
    else:
        monkeypatch.setattr(product, 'checked_inputs', lambda *args: {})
    base = replace(spec(), task_id=task_id, config=dict(scene_sha256='0' * 64, cp07_mode='diagnostic_training'))
    plan = fixture_plan(tmp_path)
    overlay = replace(plan.spec, base_spec_sha256=base_spec_digest(base))
    plan = replace(plan, spec=overlay, instance_sha256=base_spec_digest(base),
                   layout=replace(plan.layout, task=task_id))
    instance = replace(base, variation=overlay) if mode == 'random' else base
    s = stage()
    task = s.GetPrimAtPath('/World/Task')
    UsdGeom.Cube(task).CreateSizeAttr(.01)
    UsdGeom.Xformable(task).AddTranslateOp().Set(Gf.Vec3d(2, 2, 2))
    if task_id == 'PR5':
        for path in (product.STAMP, '/World/Task/CardA', '/World/Task/HolderA', '/World/Task/Board'):
            UsdGeom.Xform.Define(s, path)
    class NativeWorldBoundary(Exception):
        pass
    class Sim:
        stage = s
        @property
        def world(self):
            raise NativeWorldBoundary()
    robot = SimpleNamespace(prepare_stage=lambda *args: None, attach_cameras=lambda *args: None,
                            mount_bases=lambda *args, **kwargs: None)
    session = VariationSession(plan) if mode == 'random' else None
    output = tmp_path / 'episode'
    ctx = BuildContext(Sim(), robot, SimpleNamespace(resolve=lambda *args: texture), output, session)
    before = s.Flatten().ExportToString()
    with pytest.raises(NativeWorldBoundary):
        product.build(instance, ctx)
    if mode == 'random':
        assert session.receipt is not None, 'builder reached native world without applying its variation seam'
        assert session.receipt['protected_diff'] == []
        assert s.GetPrimAtPath('/World/EvalVariation/box_00000_0')
        assert (output / 'variation.usda').is_file()
    else:
        assert s.Flatten().ExportToString() == before
        assert not output.exists()


@pytest.mark.parametrize('failure', ['none', 'different_sdk_bytes', 'missing_release_module', 'missing_texture'])
def test_symbolic_sdk_module_is_rebound_to_exact_release_closure(tmp_path, monkeypatch, failure):
    from pxr import Sdf, UsdUtils
    from mind2act_eval.variations import file_digest
    plan = fixture_plan(tmp_path)
    pin = plan.assets['box/00000']
    root = tmp_path/'sdk/mdl'
    module = root/'gltf/pbr.mdl'
    module.parent.mkdir(parents=True)
    module.write_text('mdl 1.8; export material pbr() = material();')
    if failure == 'missing_texture':
        module.write_text('mdl 1.8; export texture_2d t = texture_2d("missing.png");')
    installed = tmp_path/'installation/mdl/gltf/pbr.mdl'
    installed.parent.mkdir(parents=True)
    installed.write_bytes(module.read_bytes())
    source = Usd.Stage.Open(str(pin.path))
    shader = UsdShade.Shader.Define(source, '/Object/Shader')
    shader.CreateImplementationSourceAttr('sourceAsset')
    shader.SetSourceAsset(Sdf.AssetPath('gltf/pbr.mdl'), 'mdl')
    source.GetRootLayer().Save()
    pin = replace(pin, sha256=file_digest(pin.path), search_paths=(root,),
                  dependencies=((module, file_digest(module)),))
    plan = replace(plan, assets={**plan.assets, 'box/00000': pin})
    # Reproduce Kit's SDK-first result without starting Kit/GPU.
    original = UsdUtils.ComputeAllDependencies
    def sdk_first(path):
        layers, resources, unresolved = original(path)
        return layers, [str(installed) if str(p).endswith('gltf/pbr.mdl') else p for p in resources], unresolved
    monkeypatch.setattr(UsdUtils, 'ComputeAllDependencies', sdk_first)
    if failure == 'different_sdk_bytes':
        installed.write_text('mdl 1.8; // different SDK')
    if failure == 'missing_release_module':
        module.unlink()
    if failure != 'none':
        with pytest.raises((ValueError, FileNotFoundError)):
            apply(plan, stage(), runtime_targets(tmp_path))
        return
    target = stage()
    assert apply(plan, target, runtime_targets(tmp_path))['protected_diff'] == []
    attr = target.GetPrimAtPath('/World/EvalVariation/box_00000_0/Shader').GetAttribute('info:mdl:sourceAsset')
    assert Path(attr.Get().resolvedPath) == module
    assert not Path(attr.Get().path).is_absolute()
    assert str(installed) not in target.GetEditTarget().GetLayer().ExportToString()
