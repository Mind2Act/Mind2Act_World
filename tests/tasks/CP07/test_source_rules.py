"""Approved per-kind predicates and source-bound lower-body assignment."""
from copy import deepcopy
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP07.binding import validate
from mind2act_eval.tasks.CP07.rules import IncompleteEvidence, orientation_matches, placed
from mind2act_eval.tasks.CP07.scorer import score
from test_physical import Trace, fixture


def row(**changes):
    return dict(piece_id="measured", kind="bar", orientation_rule="not_applicable",
                orientation_error_rad=None, lift_m=.04, bilateral_fraction=1.,
                position_error_m=.009, released=True, supported=True, **changes)


@pytest.mark.parametrize("error,expected", [(.099999, True), (.10, True), (.100001, False)])
def test_l_road_only_inclusive_source_angular_boundary(error, expected):
    r = row()
    r.update(kind="L_road", orientation_rule="source_l_road_max_0.10rad", orientation_error_rad=error)
    assert placed(r) is expected


def test_non_l_road_has_no_invented_angle_or_blanket_unknown_success():
    assert placed(row())  # 9mm passes the normative 12mm, not old runtime 8mm.
    r = row()
    r["orientation_error_rad"] = .1
    with pytest.raises(ValueError, match="no angular criterion"):
        placed(r)
    for key in ("kind", "orientation_rule"):
        r = row()
        del r[key]
        with pytest.raises(IncompleteEvidence):
            placed(r)
    r = row()
    r.update(kind="L_road", orientation_rule="source_l_road_max_0.10rad")
    with pytest.raises(IncompleteEvidence):
        placed(r)
    r["lift_m"] = 0.
    assert placed(r) is False  # Independently proven failure dominates unknown.


def test_non_l_road_rotation_does_not_create_a_new_support_angle_gate():
    t = Trace()
    t.place(0)
    t.state["pieces"]["p0"]["q_wxyz"] = [0., 1., 0., 0.]
    t.step()
    assert t.result().value == 0


def test_binding_requires_typed_kind_and_rule_but_no_formal_hard_inventory():
    spec, binding, _ = fixture("hard")
    assert validate(spec, binding) is binding
    binding["solutions"][0]["slots"][0]["angle_tolerance_rad"] = .10
    with pytest.raises(ValueError, match="per-kind"):
        validate(spec, binding)
    del binding["solutions"][0]["slots"][0]["angle_tolerance_rad"]
    del binding["pieces"][0]["kind"]
    with pytest.raises(KeyError):
        validate(spec, binding)


@pytest.mark.parametrize("lower_id", ["p0", "p13"])
def test_lower_support_variants_do_not_create_partial_bridge_credit(lower_id):
    t = Trace("hard")
    second = t.binding["solutions"][0]["slots"][1]
    second.update(support_paths=[], support_slots=[0])
    t.place(0, name=lower_id)
    t.place(1)
    t.state["pieces"]["p1"]["support_contacts"] = ["/World/Pieces/" + lower_id]
    t.step()
    assert t.result().value == 0
    unrelated = "p13" if lower_id == "p0" else "p0"
    t.state["pieces"]["p1"]["support_contacts"] = ["/World/Pieces/" + unrelated]
    t.step()
    assert t.result().value == 0


def test_unknown_alternative_cannot_invent_zero_or_hide_proven_success():
    t=Trace();t.press();t.state['car']['p_m'][0]=.4;t.step()
    v=t.scene.facts.snapshot();v['solutions']=[{'unknown':None}]
    assert score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v)).value==100
    v['terminal_car']['far_bank_supported']=False
    assert score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v)).value==0


def test_incomplete_bridge_metadata_does_not_change_binary_score():
    t=Trace();t.place(0);v=t.scene.facts.snapshot()
    v['solutions']=[{'first_extra_slot':1,'slots':[{'kind':None}]}]
    assert score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v)).value==0


def test_candidate_local_frame_target_is_not_replaced_by_slot_default():
    t = Trace()
    slots = t.binding["solutions"][0]["slots"]
    for piece in t.binding["pieces"]:
        piece["kind"] = "L_road"
    for slot in slots:
        slot.update(kind="L_road", orientation_rule="source_l_road_max_0.10rad")
    slots[0]["candidate_goal_poses"] = {p["id"]: dict(p_m=slots[0]["p_m"][:],
        q_wxyz=[0., 0., 0., 1.] if p["id"] == "p0" else [1., 0., 0., 0.])
        for p in t.binding["pieces"]}
    validate(t.spec, t.binding)
    t.place(0)
    assert t.result().value == 0
    t.state["pieces"]["p0"]["q_wxyz"] = [0., 0., 0., 1.]
    t.step()
    assert t.result().value == 0
