"""Support requires actual collision faces; no contact-force or native-run claim."""
from copy import deepcopy
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, Sdf
from mind2act_eval.tasks.PR4.mechanism.support import capture_support, support_witness, validate_support, _composition_errors
from .support_fixture import PATH, author_support, support_geometry


@pytest.mark.parametrize('rigid_rest,height,expected', [
    (0., .8005, True), (0., .8007, False), (0., .7993, False),
    (.0006, .8011, True), (.0006, .8013, False), (.0006, .7999, False)])
def test_only_actual_sum_of_rest_surfaces_can_support(rigid_rest, height, expected):
    facts = support_witness([[0., 0., height]], support_geometry(rigid_rest=rigid_rest))
    assert facts['supported'] is expected
    assert facts['fem_contact_force_claimed'] is False


def test_contact_offset_does_not_extend_effective_rest_surface():
    stage=Usd.Stage.CreateInMemory();support=author_support(stage)
    for prim in (stage.GetPrimAtPath('/World/DustCloth'),support.GetPrim()):
        prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.1)
    geometry=capture_support(stage,dict(body_path='/World/DustCloth',panel_prim=PATH))
    assert not support_witness([[0.,0.,.801]],geometry)['supported']


def test_vertical_collision_side_is_not_table_support():
    stage=Usd.Stage.CreateInMemory();support=author_support(stage)
    support.GetPointsAttr().Set([(0.,-.5,.3),(0.,.5,.3),(0.,.5,1.3),(0.,-.5,1.3)])
    geometry=capture_support(stage,dict(body_path='/World/DustCloth',panel_prim=PATH))
    assert not support_witness([[0.,0.,.8]],geometry)['supported']


@pytest.mark.parametrize('bad', ['disabled_cloth','disabled_support','missing_cloth_rest','missing_support_rest',
                                'dynamic_support','excluded_pair','missing_support','source_identity'])
def test_missing_or_nonphysical_source_proof_is_infra(bad):
    stage=Usd.Stage.CreateInMemory();support=author_support(stage)
    cloth=stage.GetPrimAtPath('/World/DustCloth')
    origin=dict(body_path='/World/DustCloth',panel_prim=PATH)
    if bad=='disabled_cloth':cloth.GetAttribute('physics:collisionEnabled').Set(False)
    if bad=='disabled_support':support.GetPrim().GetAttribute('physics:collisionEnabled').Set(False)
    if bad=='missing_cloth_rest':cloth.GetAttribute('physxCollision:restOffset').Clear()
    if bad=='missing_support_rest':support.GetPrim().GetAttribute('physxCollision:restOffset').Clear()
    if bad=='dynamic_support':UsdPhysics.RigidBodyAPI.Apply(support.GetPrim()).CreateRigidBodyEnabledAttr(True)
    if bad=='excluded_pair':UsdPhysics.FilteredPairsAPI.Apply(cloth).CreateFilteredPairsRel().SetTargets([PATH])
    if bad=='missing_support':stage.RemovePrim(PATH)
    if bad=='source_identity':origin['panel_prim']='/World/NoSuchTarget'
    with pytest.raises(ValueError,match='PR4'):
        capture_support(stage,origin)


def test_old_or_forged_support_boolean_is_not_a_native_geometry_receipt():
    with pytest.raises(ValueError,match='support geometry'):
        validate_support(dict(supported=True))
    proof=deepcopy(support_geometry());proof['surfaces'][0]['path']='/World/UnrelatedTable'
    with pytest.raises(ValueError,match='identity'):
        validate_support(proof,dict(panel_prim=PATH))


def test_point_beside_finite_source_triangle_cannot_use_infinite_plane():
    proof=support_geometry()
    assert support_witness([[.499,0.,.8]],proof)['supported']
    assert not support_witness([[.501,0.,.8]],proof)['supported']


class _OlderStage:
    def __init__(self, stage):self.stage=stage
    def __getattr__(self, name):
        if name=='GetCompositionErrors':raise AttributeError(name)
        return getattr(self.stage,name)


def test_older_usd_real_support_geometry_unchanged():
    stage=Usd.Stage.CreateInMemory();author_support(stage)
    origin=dict(body_path='/World/DustCloth',panel_prim=PATH)
    assert capture_support(_OlderStage(stage),origin)==capture_support(stage,origin)


def test_older_usd_rejects_real_unresolved_support_reference(tmp_path):
    stage=Usd.Stage.CreateInMemory();panel=author_support(stage)
    panel.GetPrim().GetReferences().AddReference(str(tmp_path/'missing-support.usda'))
    assert list(panel.GetPrim().GetPrimIndex().localErrors)
    with pytest.raises(ValueError,match='unresolved native support geometry composition'):
        capture_support(_OlderStage(stage),dict(body_path='/World/DustCloth',panel_prim=PATH))


def test_older_usd_keeps_real_layer_stack_errors(tmp_path):
    stage=Usd.Stage.CreateInMemory();author_support(stage)
    stage.GetRootLayer().subLayerPaths=[str(tmp_path/'missing-sublayer.usda')]
    errors={str(e) for e in _composition_errors(_OlderStage(stage))}
    assert any('missing-sublayer.usda' in e for e in errors)
    if hasattr(stage,'GetCompositionErrors'):
        assert errors=={str(e) for e in stage.GetCompositionErrors()}
