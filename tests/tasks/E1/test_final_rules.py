from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.E1.evidence import Evidence, WINDOW_TICKS
from mind2act_eval.tasks.E1.scorer import score
from .test_evidence import item, row, settle


def test_actual_supported_submission_seals_once_after_release_and_dwell():
    recorder = Evidence({1: 'red'}, 'easy')
    food = item('/a', 'red', True, touch=True)
    assert not settle(recorder, 1, lambda t: [food], 1, WINDOW_TICKS)
    food['touch_robot'] = False
    assert not settle(recorder, 1, lambda t: [food], WINDOW_TICKS + 1, WINDOW_TICKS - 1)
    assert recorder.sample(row(2 * WINDOW_TICKS, 1, [food]))
    assert recorder.results[0]['pass'] is True
    assert not recorder.sample(row(2 * WINDOW_TICKS + 1, 1, [item('/b', 'red', True)]))
    assert len(recorder.results) == 1


def test_rim_brush_without_real_support_does_not_consume_opportunity():
    recorder = Evidence({1: 'red'}, 'easy')
    food = item('/a', 'red', True)
    food['support'] = False
    assert not recorder.sample(row(1, 1, [food]))
    assert settle(recorder, 1, lambda t: [item('/a', 'red', True)], 2, WINDOW_TICKS)
    assert recorder.results[0]['pass'] is True


def test_wrong_first_submission_cannot_be_replaced_and_later_batch_is_independent():
    recorder = Evidence({1: 'red', 2: 'blue'}, 'easy')
    wrong = lambda t: [item('/a', 'blue', True), item('/b', 'red', False)]
    assert settle(recorder, 1, wrong, 1, WINDOW_TICKS)
    assert not recorder.sample(row(WINDOW_TICKS + 1, 1, [item('/b', 'red', True)]))
    assert settle(recorder, 2, lambda t: [item('/c', 'blue', True)], WINDOW_TICKS + 2, WINDOW_TICKS)
    assert [r['pass'] for r in recorder.results] == [False, True]
    assert score(recorder.native_evidence('E1', 'x', 2 * WINDOW_TICKS + 1)).value == 15


def test_proportional_credit_for_one_of_six_batches():
    result = score(NativeEvidence('E1', 'x', 1, {'difficulty': 'hard', 'rounds': [{'round': 1, 'pass': True}]}))
    assert result.value == 15 / 6
