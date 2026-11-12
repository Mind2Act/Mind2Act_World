"""Pad ownership is geometry-only; case/source admission remains task-owned."""
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mind2act_eval.robot.task_collision_overlay import expected_descriptor
from mind2act_eval.tasks.CP09.instances import validate
from test_cp09_robot_overlay import selected
from test_cp09_source_pad_restore import physx_metadata, source_stage


def test_pad_descriptor_is_identical_across_cases_and_difficulties():
    reference=expected_descriptor(selected('easy'))
    for diff in ('easy','medium','hard'):
        spec=selected(diff)
        assert expected_descriptor(spec)==reference
        changed=replace(spec,instance_id='different-case',source_id='different-source',seed=77,
                        config={**spec.config,'h5_sha256':'0'*64,'source_case_sha256':'1'*64})
        assert expected_descriptor(changed)==reference
        with pytest.raises(ValueError):validate(changed)


def test_same_real_prepare_does_not_authorize_a_different_case(tmp_path,monkeypatch):
    from mind2act_eval.robot import X5Robot
    stage,robot,spec=source_stage(tmp_path,monkeypatch)
    changed=replace(spec,instance_id='different-case',seed=77)
    independent=X5Robot(usd_path=robot._usd_path)
    independent.bind_task_instance(changed,tmp_path/'different')
    independent.prepare_stage(stage)
    assert len(independent.native_receipt()['source_pad_binding']['restored_pad_paths'])==4
    with pytest.raises(ValueError):validate(changed)
    from mind2act_eval.tasks.CP09.scene import build
    with pytest.raises(ValueError):build(changed,SimpleNamespace())


def test_packaged_asset_matches_independent_acquisition_fixture():
    import json
    from importlib import resources
    from mind2act_eval.robot.cp09_pad_geometry import ASSET_NAME,ASSET_SHA256,ACQUISITION_VERSION,descriptor
    raw=resources.files('mind2act_eval.robot').joinpath(ASSET_NAME).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ASSET_SHA256
    asset=json.loads(raw);desc=descriptor()
    fixture=Path(__file__).parent/'fixtures/cp09_source_pads.usda'
    assert desc['source_geometry_asset_sha256']==hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert desc['acquisition_version']==ACQUISITION_VERSION
    assert desc['geometry_sha256']==hashlib.sha256(json.dumps(asset['pads'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert all(k not in desc for k in ('instance_id','seed','h5_sha256','source_id','difficulty'))


def test_altered_geometry_asset_fails_closed(tmp_path,monkeypatch):
    from mind2act_eval.robot import cp09_pad_geometry as geometry
    (tmp_path/geometry.ASSET_NAME).write_text('{}')
    monkeypatch.setattr(geometry.resources,'files',lambda _:tmp_path)
    with pytest.raises(ValueError,match='asset SHA'):geometry.descriptor()


def test_task_and_shared_geometry_declarations_are_identical():
    from mind2act_eval.tasks.CP09.geometry import descriptor
    for difficulty in ('easy','medium','hard'):
        assert descriptor(difficulty)==expected_descriptor()
