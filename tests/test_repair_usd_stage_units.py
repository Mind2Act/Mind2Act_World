from pathlib import Path
from runpy import run_path

import pytest
from pxr import Sdf, Usd, UsdGeom


TOOL = run_path(str(Path(__file__).resolve().parents[1] / 'tools' / 'repair_usd_stage_units.py'))


def overlay(tmp_path, units=1.0, axis='Z'):
    source = tmp_path / 'source.usda'
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, axis)
    stage.GetRootLayer().Save()
    target = tmp_path / 'overlay.usda'
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = ['source.usda']
    layer.Save()
    return source, target


def test_repair_only_writes_the_overlay(tmp_path):
    source, target = overlay(tmp_path)
    source_before, before = source.read_bytes(), target.read_bytes()

    assert TOOL['main']([str(target), '--check-only']) == 1
    assert TOOL['main']([str(target), '--dry-run']) == 0
    assert target.read_bytes() == before

    assert TOOL['main']([str(target), '--apply']) == 0
    assert TOOL['main']([str(target), '--check-only']) == 0
    assert source.read_bytes() == source_before
    stage = Usd.Stage.Open(str(target))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == 'Z'
    after = target.read_bytes()
    assert TOOL['repair_usd_root_units'](target)['status'] == 'already_authored'
    assert target.read_bytes() == after


@pytest.mark.parametrize(('units', 'axis'), [(.01, 'Z'), (1., 'Y')])
def test_repair_rejects_wrong_source_metrics_before_writing(tmp_path, units, axis):
    _, target = overlay(tmp_path, units, axis)
    before = target.read_bytes()
    with pytest.raises(ValueError, match='stage units/axis mismatch'):
        TOOL['repair_usd_root_units'](target)
    assert target.read_bytes() == before
