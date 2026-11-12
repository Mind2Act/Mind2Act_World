import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from mind2act_eval.english_spec_projection import project_instruction
from mind2act_eval.model_text import cjk_count
from mind2act_eval.tasks._instructions import public_instruction


@pytest.mark.parametrize('task,key,target,hold', [('PR1','target_ohm',20.,'0.5 seconds'),
                                               ('PR2','target_volt',140.,'1 second')])
def test_parameterized_precision_text_matches_existing_scene_template(task, key, target, hold):
    text = public_instruction(SimpleNamespace(task_id=task, difficulty='hard', config={key: target}))
    assert str(int(target)) in text and hold in text
    assert cjk_count(text) == 0 and '{TARGET}' not in text


def test_projection_changes_only_instruction_and_does_not_mutate_raw_source():
    raw = dict(task_id='F1', difficulty='easy', instance_id='case', seed=123,
               config=dict(instruction='中文原始说明', source_record_sha256='f'*64,
                           case=dict(buttons=['A','B'], force=100.)), assets={'station': 'source'})
    before = deepcopy(raw)
    projected = project_instruction(raw)
    assert raw == before and projected['config']['instruction'].startswith('Probe the two unmarked buttons')
    projected['config']['instruction'] = raw['config']['instruction']
    assert projected == before
    projected['config']['case']['force'] = 999.
    assert raw['config']['case']['force'] == 100.


def test_new15_instruction_remains_unchanged_for_explicit_separate_review():
    raw = dict(task_id='CP09', config=dict(instruction='Approved English.', geometry_decision='原始裁决'))
    assert project_instruction(raw) == raw


def test_missing_public_target_is_rejected_instead_of_invented():
    with pytest.raises(KeyError):
        project_instruction(dict(task_id='PR2', difficulty='easy', config={'instruction':'source'}))




