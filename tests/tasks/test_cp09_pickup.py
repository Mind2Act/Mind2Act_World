import pytest
from mind2act_eval.tasks.CP09.pickup import PickupWitness


def test_real_bilateral_lift_requires_all_24_ticks():
    witness = PickupWitness("pen", 1)
    for tick in range(23):
        assert witness.sample(tick, [True, True], .040001) is None
    event = witness.sample(23, [True, True], .040001)
    assert event["consecutive_lift_ticks"] == 24
    assert event["attachment_joint_created_before_lift"] is False


@pytest.mark.parametrize("contacts,lift,attached", [
    ([False, True], .05, False), ([True, True], .04, False),
    ([True, True], .05, True),
])
def test_interrupted_or_assisted_lift_restarts_proof(contacts, lift, attached):
    witness = PickupWitness("eraser")
    for tick in range(23):
        witness.sample(tick, [True, True], .05)
    assert witness.sample(23, contacts, lift, attachment_before_lift=attached) is None
    assert witness.consecutive == 0
    for tick in range(24, 47):
        assert witness.sample(tick, [True, True], .05) is None
    assert witness.sample(47, [True, True], .05) is not None


def test_missing_physics_ticks_are_invalid():
    witness = PickupWitness("pen", 1)
    witness.sample(0, [True, True], .05)
    with pytest.raises(ValueError, match="contiguous"):
        witness.sample(4, [True, True], .05)
