"""Quiet parcels never end control before T0."""
from mind2act_eval.tasks.DR1.scorer import round_over, score
from test_dr1_scorer import parcel, evidence


def test_goal_does_not_trigger_early_round_end():
    ev = evidence([[parcel()]] * 120)
    assert not round_over(ev.values['ticks'], 1)
    assert score(ev).value == 100


def test_off_tray_parcel_retains_chance_to_be_recovered():
    ev = evidence([[parcel(x=2., z=.1, tray=False)]] * 120)
    assert not round_over(ev.values['ticks'], 1)
    assert score(ev).value == 0


def test_side_contact_does_not_masquerade_as_target_support():
    p = dict(parcel(), tray_supported=False)
    assert score(evidence([[p]])).value == 0


def test_native_contact_without_owned_normal_is_infra():
    import pytest
    from mind2act_eval.tasks.DR1.evidence import ContactTable
    root = '/World/FullStream/Goods_00'
    table = ContactTable([root])
    table.report([root+'/Visual/collision', '/World/FR1/InspectionTray/Floor'], True, .1, 0.)
    with pytest.raises(ValueError, match='support normal'):
        table.supports(0)
