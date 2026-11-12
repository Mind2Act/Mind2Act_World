"""Real runner seams: final success after an action, sequence completion per event."""
from dataclasses import replace

import pytest

from mind2act_eval.contracts import Termination
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.scoring import EVENT_SEQUENCE_TASKS, RESULT_TASKS
from test_observation_closeout import window_case


def test_thirty_tasks_have_one_termination_kind():
    assert len(EVENT_SEQUENCE_TASKS)==8 and len(RESULT_TASKS)==22
    assert not EVENT_SEQUENCE_TASKS & RESULT_TASKS


@pytest.mark.parametrize('task_id', sorted(RESULT_TASKS))
def test_result_success_waits_for_completed_action_not_first_subtick(tmp_path, task_id):
    spec,task,ctx,actor,applied,_=window_case(tmp_path,task_id=task_id,complete_at=1)
    result=run_episode(spec,task,ctx,actor,limits=EpisodeLimits(12,10))
    assert result.infra_valid,result.error
    assert result.termination==Termination.COMPLETED and result.ticks==4
    assert actor.calls==1 and len(applied)==4


@pytest.mark.parametrize('task_id', sorted(EVENT_SEQUENCE_TASKS))
def test_event_sequence_completion_ends_at_its_actual_tick(tmp_path, task_id):
    spec,task,ctx,actor,applied,_=window_case(tmp_path,task_id=task_id,complete_at=1)
    result=run_episode(spec,task,ctx,actor,limits=EpisodeLimits(12,10))
    assert result.infra_valid,result.error
    assert result.termination==Termination.COMPLETED and result.ticks==1
    assert len(applied)==1


def test_transient_success_inside_action_does_not_latch_full_credit(tmp_path):
    spec,task,ctx,actor,_,_=window_case(tmp_path,complete_at=1,fall_at=3)
    actor.act=lambda obs: __import__('mind2act_eval.adapters.demo',fromlist=['DemoAdapter']).DemoAdapter().act(obs)
    result=run_episode(spec,task,ctx,actor,limits=EpisodeLimits(8,10))
    assert result.infra_valid and result.ticks==8 and result.score==0
    assert result.termination==Termination.PHYSICS_BUDGET


def test_result_checks_only_at_model_chunk_end(tmp_path):
    spec,task,ctx,actor,_,_=window_case(tmp_path,complete_at=1)
    calls=[]
    def act(obs):
        from mind2act_eval.adapters.demo import DemoAdapter
        calls.append(obs.tick)
        return DemoAdapter().act(obs)
    actor.act=act
    actor.peek_next_action=lambda: act_copy if len(calls)==1 else None
    from mind2act_eval.adapters.demo import DemoAdapter
    act_copy=DemoAdapter().act(task.build(spec,ctx).observe(0))
    result=run_episode(spec,task,ctx,actor,limits=EpisodeLimits(12,10))
    assert result.infra_valid,result.error
    assert result.ticks==8 and calls==[0,4]


@pytest.mark.parametrize('task_id', ['PR1','DR2'])
def test_legacy_violation_done_never_terminates_model(tmp_path,task_id):
    spec,task,ctx,actor,_,_=window_case(tmp_path,task_id=task_id,complete_at=999)
    scene=task.build(spec,ctx)
    scene.is_complete=lambda:True
    scene.is_success=lambda:False
    scene.events_complete=lambda:False
    task=replace(task,build=lambda *args:scene)
    actor.act=lambda obs: __import__('mind2act_eval.adapters.demo',fromlist=['DemoAdapter']).DemoAdapter().act(obs)
    result=run_episode(spec,task,ctx,actor,limits=EpisodeLimits(8,10))
    assert result.infra_valid and result.ticks==8
    assert result.termination==Termination.PHYSICS_BUDGET
