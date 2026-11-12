"""Unambiguous section6 arithmetic; does not choose the pending mat predicate."""
import pytest
from mind2act_eval.tasks.P2.stages import stage_value


@pytest.mark.parametrize('n,prefix,expected',[(3,0,0),(3,1,10),(3,2,20),(3,3,30),
                                           (4,0,0),(4,1,10),(4,2,20),(4,3,20),(4,4,30)])
def test_highest_stage_and_ceil_half_are_not_added(n,prefix,expected):
    assert stage_value(valid_disassembly=True,restored_prefix=prefix,target_count=n,
                       final_conditions_met=False)==expected


@pytest.mark.parametrize('n',[3,4])
def test_full_rebuild_without_final_finish_is30_and_with_finish_is100(n):
    args=dict(valid_disassembly=True,restored_prefix=n,target_count=n)
    assert stage_value(**args,final_conditions_met=False)==30
    assert stage_value(**args,final_conditions_met=True)==100


@pytest.mark.parametrize('n',[3,4])
def test_terminal_stack_needs_no_dismantling_eligibility(n):
    args=dict(restored_prefix=n,target_count=n,final_conditions_met=True)
    assert stage_value(**args,valid_disassembly=False)==100


@pytest.mark.parametrize('field,value',[('target_count',0),('target_count',True),('restored_prefix',-1),
                                       ('restored_prefix',5),('restored_prefix',2.0),('valid_disassembly',None)])
def test_missing_or_invalid_stage_facts_are_not_filled_with_a_default(field,value):
    args=dict(valid_disassembly=True,restored_prefix=2,target_count=4,
              final_conditions_met=False)
    args[field]=value
    with pytest.raises(ValueError):stage_value(**args)
