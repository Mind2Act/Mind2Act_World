"""Real USD composition/products; native SDK imports are CPU-only seams.

These tests do not initialize Kit/Hydra or qualify native rendering. USD stage
edits, RenderProduct validity and camera relationships use the real pxr APIs.
"""
import json
from types import SimpleNamespace as NS

from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender

from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.sim import IsaacSimulator, LocalAssets
from native_fixture import native_modules


VIEWPORT = '/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0'
CAMERAS = {
    'head': '/World/HeadCamera',
    'left_wrist': '/World/RobotLeft/Asset/camera/PhysCoWristCamera',
    'right_wrist': '/World/RobotRight/Asset/camera/PhysCoWristCamera',
}


def render_stage(tmp_path):
    source = tmp_path / 'source.usda'
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, '/World')
    UsdGeom.Camera.Define(stage, '/World/SourceCamera')
    legacy = UsdRender.Product.Define(stage, '/Render/SourceLegacyProduct')
    legacy.CreateCameraRel().SetTargets(['/World/SourceCamera'])
    stage.GetRootLayer().Save()
    source_bytes = source.read_bytes()

    path = tmp_path / 'runtime.usda'
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    stage.GetRootLayer().Save()
    products = {VIEWPORT: '/OmniverseKit_Persp'}
    products.update({f'/Render/Public/{name}': camera for name, camera in CAMERAS.items()})
    for product_path, camera_path in products.items():
        UsdGeom.Camera.Define(stage, camera_path)
        product = UsdRender.Product.Define(stage, product_path)
        product.CreateCameraRel().SetTargets([camera_path])
        product.CreateResolutionAttr(Gf.Vec2i(64, 64))
    return source, source_bytes, path, stage, products


def render_snapshot(stage):
    layer = Sdf.Layer.CreateAnonymous()
    Sdf.CopySpec(stage.Flatten(), '/Render', layer, '/Render')
    return layer.ExportToString()


def assert_products(stage, products):
    for path, camera in products.items():
        product = UsdRender.Product.Get(stage, path)
        assert product, f'active RenderProduct removed: {path}'
        assert product.GetCameraRel().GetTargets() == [Sdf.Path(camera)]
        assert UsdGeom.Camera.Get(stage, camera)


def test_negative_control_whole_render_removal_breaks_live_sdk_products(tmp_path):
    source, original, _, stage, products = render_stage(tmp_path)
    assert_products(stage, products)
    assert stage.RemovePrim('/Render')
    assert not UsdRender.Product.Get(stage, VIEWPORT)
    # A remaining weaker source opinion does not save the active SDK products.
    assert UsdRender.Product.Get(stage, '/Render/SourceLegacyProduct')
    assert source.read_bytes() == original


def test_m1_authoring_preserves_active_products_public_cameras_and_source(tmp_path, monkeypatch):
    import pxr
    from mind2act_eval.tasks.M1.instances import layout
    from mind2act_eval.tasks.M1.mechanism import author_stage

    # usd-core does not ship Isaac PhysxSchema. The unrelated native metadata
    # setters are inert; actual USD/UsdPhysics/UsdRender authoring stays real.
    api = NS(Apply=lambda prim: NS(**{name: lambda value: None for name in (
        'CreateContactOffsetAttr', 'CreateRestOffsetAttr', 'CreateThresholdAttr',
        'CreateSolverPositionIterationCountAttr')}))
    monkeypatch.setattr(pxr, 'PhysxSchema', NS(PhysxCollisionAPI=api,
        PhysxContactReportAPI=api, PhysxRigidBodyAPI=api), raising=False)
    source, original, _, stage, products = render_stage(tmp_path)
    UsdGeom.Cube.Define(stage, '/World/TaskBody')
    UsdGeom.Cube.Define(stage, '/World/TaskStatic')
    manifest = dict(bodies={'body': dict(prim_path='/World/TaskBody', mass_kg=.1,
        collision_prims=['/World/TaskBody'])}, static_collision_roots=['/World/TaskStatic'])
    before = render_snapshot(stage)
    author_stage(stage, manifest, layout())
    assert_products(stage, products)
    assert render_snapshot(stage) == before
    assert source.read_bytes() == original


def test_p1_real_build_preserves_active_products_before_robot_setup(tmp_path, monkeypatch):
    from mind2act_eval.tasks.P1 import scene
    import pytest

    source, original, path, stage, products = render_stage(tmp_path)
    before = render_snapshot(stage)
    module = native_modules(monkeypatch)
    # The build stops before geometry is loaded; no cup-geometry algorithm runs.
    module('trimesh')
    active, opened = {}, []

    def open_stage(value):
        opened.append(value)
        active['stage'] = Usd.Stage.Open(value)
        return True

    module('isaacsim.core.utils.stage', open_stage=open_stage,
           get_current_stage=lambda: active['stage'])
    (tmp_path / 'manifest.json').write_text(json.dumps({'case': {'orders': []}}))
    cards = tmp_path / 'cards'
    cards.mkdir()
    for name in scene.REQUIRED_CARDS:
        (cards / name).touch()
    assets = LocalAssets(tmp_path, dict(scene=path.name, manifest='manifest.json', cards='cards'))
    sim = IsaacSimulator.__new__(IsaacSimulator)
    sim._world = None

    class ReachedRobotSetup(Exception):
        pass

    def prepare(current):
        assert_products(current, products)
        assert render_snapshot(current) == before
        raise ReachedRobotSetup

    ctx = NS(sim=sim, assets=assets, robot=NS(prepare_stage=prepare))
    spec = InstanceSpec('P1', 'render-contract', 'easy', 'source', 'v1', 'v1', 'v1')
    with pytest.raises(ReachedRobotSetup):
        scene.build(spec, ctx)
    assert opened == [str(path)]
    assert source.read_bytes() == original
