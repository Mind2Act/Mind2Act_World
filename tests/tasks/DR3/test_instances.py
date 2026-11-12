"""The source release, condition, scene and delivered input remain one binding."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.tasks.DR3.instances import (
    SOURCE_REVISION, instructions, signature, spec_from_row, validate)


def source_row(level='easy', split='train'):
    source = dict(difficulty=level, seed=0, cams={'head': '/World/HeadCamera'},
                  Z=.776, GATE_X=-.16, GATE_Y=-.105, W=.30, H=.26, TRAVEL=.34,
                  SOURCE=[-.34, -.235, .788], DEST=[.015, -.13], SZ=[.063, .033, .024],
                  TRANS=.8, CLOSED=3., DUR=6.6, PERIOD=11.2)
    if level == 'medium':
        source.update(DUR=6., PERIOD=10.6)
    if level == 'hard':
        source['gates'] = [dict(name='Slow', x=-.23, period=18.4, first_open=4.8, open_seconds=12.8),
                           dict(name='Fast', x=-.105, period=9.2, first_open=2., open_seconds=6.4)]
    config = dict(source_case=source, scene_sha256='a' * 64, phase_ticks=6,
                  physics_budget=21600, phase_condition_version='fr3-initial-mechanism-phase-v1',
                  native_inputs={'args': {'scene': '/private/source/scene.usda'}})
    row = dict(task_id='FR3', case_id='source-condition-' + level, difficulty=level,
               split=split, variant='clean', instruction=instructions(level)['zh'], config=config,
               binding={'source_revision': SOURCE_REVISION, 'config_sha256': 'b' * 64})
    condition = dict(instance_id=row['case_id'], difficulty=level, source_revision=SOURCE_REVISION,
                     source_case_sha256=signature(source), native_config_sha256=signature(config),
                     semantic_signature='b' * 64, scene_sha256='a' * 64, phase_ticks=6)
    return row, condition


def convert(row, condition, **kwargs):
    return spec_from_row(row, condition=condition, scene_asset='DR3/selected/scene.usda',
                         x5_asset='robot/x5/ARX.usd', asset_version='synthetic-fixture-v1', **kwargs)


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize('split', ['train', 'eval'])
def test_selected_train_and_eval_condition_use_the_same_validation(level, split):
    row, condition = source_row(level, split)
    spec = convert(row, condition)
    assert spec.task_id == 'DR3' and spec.instance_id == row['case_id']
    assert spec.config['instruction'] == instructions(level)['en']
    assert spec.config['phase_ticks'] == 6
    assert 'native_inputs' not in spec.config and len(spec.config['cameras']) == 3
    assert '/private/' not in str(spec.config)
    with pytest.raises(TypeError):
        spec.config['source_case']['seed'] = 123


@pytest.mark.parametrize('field,value', [('phase_ticks', 72), ('scene_sha256', 'c' * 64),
                                         ('difficulty', 'hard'), ('source_revision', 'wrong')])
def test_condition_cannot_substitute_phase_scene_tier_or_source(field, value):
    row, condition = source_row()
    bad = deepcopy(condition)
    bad[field] = value
    with pytest.raises(ValueError):
        convert(row, bad)


def test_native_config_cannot_change_after_condition_selection():
    row, condition = source_row()
    row['config']['source_case']['W'] += .01
    with pytest.raises(ValueError):
        convert(row, condition)


def test_delivered_chinese_is_valid_but_other_tier_or_legacy_text_is_not():
    row, condition = source_row('hard')
    spec = convert(row, condition, language='zh')
    assert spec.config['instruction'] == instructions('hard')['zh']
    for text in (instructions('easy')['en'], 'Place the object using an obsolete instruction.'):
        with pytest.raises(ValueError):
            validate(replace(spec, config={**spec.config, 'instruction': text}))


def test_absolute_asset_paths_and_implicit_condition_are_rejected():
    row, condition = source_row()
    with pytest.raises(ValueError):
        spec_from_row(row, condition=condition, scene_asset='/private/source/scene.usda',
                      x5_asset='robot/x5/ARX.usd', asset_version='test')
    with pytest.raises(TypeError):
        spec_from_row(row, scene_asset='scene.usda', x5_asset='x5.usd', asset_version='test')


def test_wrong_runtime_clock_and_changed_instance_fail_before_kit_import():
    row, condition = source_row()
    spec = convert(row, condition)
    with pytest.raises(ValueError):
        validate(replace(spec, clock={'physics_hz': 120, 'control_hz': 60, 'rgb_hz': 30}))
    with pytest.raises(ValueError):
        validate(replace(spec, instance_id='another-instance'))


def test_serialized_round_trip_keeps_full_binding():
    from mind2act_eval.contracts import to_json
    row, condition = source_row()
    spec = convert(row, condition)
    resolved = InstanceSpec.from_dict(to_json(spec))
    spec.assert_binding(resolved)
    validate(resolved)
