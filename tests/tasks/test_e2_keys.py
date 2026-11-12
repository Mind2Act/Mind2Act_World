"""Measured debounce boundaries, independent of any replay phase labels."""
import pytest

from mind2act_eval.tasks.E2.mechanism.keys import Keys


def advance(keys, count, *, slot=0, depth=0.0, contact=False, enabled=True):
    for _ in range(count):
        travel, contacts = [0.0] * 9, [False] * 9
        travel[slot], contacts[slot] = depth, contact
        keys.sample(keys.last_tick + 1, travel, contacts, enabled=enabled)


def test_press_is_recorded_on_depression_and_last_key_must_return():
    keys = Keys()
    advance(keys, 1, depth=-.003)
    assert keys.digits == [1] and not keys.complete([1])
    advance(keys, 1)
    assert keys.complete([1])
    assert keys.events == [dict(start_tick=1, press_tick=1, slot=0, digit=1, release_tick=2)]


@pytest.mark.parametrize('depth,enabled', [(-.002999, True), (-.004, False)])
def test_shallow_and_pre_input_press_are_ignored(depth, enabled):
    keys = Keys()
    advance(keys, 8, depth=depth, enabled=enabled)
    advance(keys, 1)
    assert keys.digits == []


def test_long_hold_cannot_repeat_until_mechanical_return():
    keys = Keys()
    advance(keys, 100, depth=-.004)
    assert keys.digits == [1]
    advance(keys, 1)
    advance(keys, 1, depth=-.004)
    advance(keys, 1)
    assert keys.digits == [1, 1]


def test_measured_return_boundary_has_no_extra_debounce():
    keys = Keys()
    advance(keys, 1, depth=-.003)
    advance(keys, 4, depth=-.001)
    assert not keys.complete([1])
    advance(keys, 1, depth=-.000999)
    assert keys.complete([1])


def test_nonpressed_key_does_not_add_an_all_nine_release_gate():
    keys = Keys()
    advance(keys, 1, depth=-.004)
    advance(keys, 1, slot=8, depth=-.002)
    assert keys.complete([1])


@pytest.mark.parametrize("tick,travel,contacts", [(2, [0.] * 9, [False] * 9), (1, [float('nan')] * 9, [False] * 9), (1, [0.] * 8, [False] * 9), (1, [0.] * 9, [1] * 9)])
def test_invalid_measurement_cannot_create_a_native_key(tick, travel, contacts):
    with pytest.raises(ValueError):
        Keys().sample(tick, travel, contacts, enabled=True)


@pytest.mark.parametrize('target', [[1, 2], [2, 1]])
def test_out_of_turn_press_does_not_enter_the_input_string(target):
    keys = Keys()
    advance(keys, 3, slot=0, depth=-.003, contact=True)
    for _ in range(3):
        keys.sample(keys.last_tick + 1, [-.003, -.003] + [0.] * 7,
                    [True, True] + [False] * 7, enabled=True)
    advance(keys, 4, slot=0, depth=-.003, contact=True)  # release 2 before 1
    advance(keys, 4)
    assert keys.digits == [1]
    assert not keys.complete(target)


def test_next_digit_needs_no_extra_release_debounce():
    keys = Keys()
    advance(keys, 3, slot=0, depth=-.003, contact=True)
    advance(keys, 3)
    advance(keys, 3, slot=1, depth=-.003, contact=True)
    advance(keys, 4)
    assert keys.complete([1, 2]) is True


def test_single_tick_mechanical_press_and_return_needs_no_finger_or_dwell():
    keys = Keys()
    advance(keys, 1, depth=-.003, contact=False)
    assert keys.digits == [1] and not keys.complete([1])
    advance(keys, 1, depth=0., contact=True)
    assert keys.complete([1])


def test_held_previous_key_rejects_the_next_edge():
    keys = Keys()
    advance(keys, 1, depth=-.003)
    keys.sample(2, [-.003, -.003] + [0.] * 7, [False] * 9, enabled=True)
    keys.sample(3, [-.003, 0.] + [0.] * 7, [True] * 9, enabled=True)
    assert keys.digits == [1] and not keys.complete([1, 2])


def test_next_digit_is_not_accepted_until_previous_key_returns():
    keys=Keys()
    advance(keys,1,slot=0,depth=-.004)
    keys.sample(2,[-.004,-.004]+[0.]*7,[False]*9,enabled=True)
    assert keys.digits == [1]
    advance(keys,1)
    advance(keys,1,slot=1,depth=-.004)
    advance(keys,1)
    assert keys.complete([1,2])
