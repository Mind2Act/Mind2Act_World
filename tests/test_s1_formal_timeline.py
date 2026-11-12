from copy import deepcopy
from types import SimpleNamespace

import pytest

from mind2act_eval.tasks.S1.mechanism.schedule import schedule, timeline_config


def source_config(formal):
    config = dict(cue_seconds=2., cue_gap_s=.3, rotation_seconds=3., stop_seconds=.4)
    if formal:
        config['test_source'] = dict(instance_set='test50-r5', split='test')
    return config


@pytest.mark.parametrize('mode', ['clean', 'random'])
def test_formal_cue_disappears_at_one_second_and_rotation_has_no_gap(mode):
    original = source_config(True)
    before = deepcopy(original)
    spec = SimpleNamespace(config=original, variation=SimpleNamespace(mode=mode))
    effective = timeline_config(spec)
    tables = {'table': {'rotation_degrees': [90.]}}
    assert schedule(119, effective, tables, effective['cue_gap_s'])[:2] == ('cue', True)
    assert schedule(120, effective, tables, effective['cue_gap_s'])[:2] == ('rotation', False)
    assert schedule(121, effective, tables, effective['cue_gap_s'])[2]['table'] > 0.
    assert original == before  # original source identity and acquisition timing retained
    assert effective['rotation_seconds'] == 3. and effective['stop_seconds'] == .4


def test_training_replay_keeps_the_explicit_two_second_cue():
    spec = SimpleNamespace(config=source_config(False))
    effective = timeline_config(spec)
    tables = {'table': {'rotation_degrees': [90.]}}
    assert schedule(120, effective, tables, effective['cue_gap_s'])[:2] == ('cue', True)
    assert schedule(240, effective, tables, effective['cue_gap_s'])[:2] == ('rotation', False)
    assert effective == spec.config


def test_missing_source_timeline_is_not_filled_by_the_formal_ruling():
    config = source_config(True)
    del config['rotation_seconds']
    with pytest.raises(ValueError, match='rotation_seconds'):
        timeline_config(SimpleNamespace(config=config))
