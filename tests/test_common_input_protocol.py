from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from common_input_fixture import sdk, Client, response, make_policy, observation, delivered_instruction
from mind2act_eval.adapters.agent_input.templates import VERSION, GOALS, TEMPLATE, render_goal, public_instruction
from mind2act_eval.adapters.agent_input.recipes import CATALOG, RECIPES, render_public_text
from mind2act_eval.adapters.coding_payload import require_english
from mind2act_eval.adapters.coding_agent_common_input import CommonInputAdapter, CommonInputIntegration, create
from mind2act_eval.budgets import DecisionBudget, CallBudgetExhausted, AGENT_BUDGET_VERSION
from mind2act_eval.contracts import PolicyStop, Action, ConsumerKind
from mind2act_eval.runner import EpisodeLimits, run_episode
from test_runner import fixture


def test_thirty_task_ninety_recipe_catalog_is_english_and_parameters_are_public():
    assert len(CATALOG['tasks'])==30
    require_english((CATALOG,TEMPLATE))
    for task,row in CATALOG['tasks'].items():
        for tier in ('easy','medium','hard'):
            parameters = {'target':100.}
            if task == 'CP07':
                from mind2act_eval.adapters.agent_input.cp07 import public_profile
                parameters = {'cp07_recipe_profile':public_profile('diagnostic-training',tier)}
            goal=render_goal(dict(task=task,difficulty=tier,environment_parameters=parameters,
                instruction=delivered_instruction(task,tier,parameters)))
            assert goal.startswith('Goal: ') and 'TASK RECIPE:' in goal
            assert '{TARGET}' not in goal and 'TASK RECIPE:' not in public_instruction(goal)
            assert public_instruction(goal)


@pytest.mark.parametrize('steps',[1,5,1200])
def test_wait_uses_exact_physics_ticks_and_no_planner(sdk,steps):
    client=Client([response('wait_steps',{'steps':steps})])
    policy,budget=make_policy(sdk,client)
    chunk=policy.act(observation(sdk))
    assert sum(row.meta['physics_ticks'] for row in chunk.actions)==steps
    assert all(1<=row.meta['physics_ticks']<=4 for row in chunk.actions)
    assert budget.used==1 and [t['function']['name'] for t in client.sent[0][1]]==['move_eef','give_up','wait_steps','memory_resolve']
    assert 'TASK RECIPE:' in client.sent[0][0][1]['content']


def test_readonly_list_and_open_advance_no_physics_and_attach_exact_evidence(sdk):
    client=Client([response('memory_resolve',dict(query='',event_ref=None,camera='head',cursor=0)),
        response('memory_resolve',dict(query='',event_ref='obs-000000000',camera='head',cursor=0),2),
        response('wait_steps',{'steps':5},3)])
    policy,budget=make_policy(sdk,client)
    chunk=policy.act(observation(sdk))
    assert budget.used==3 and sum(row.meta['physics_ticks'] for row in chunk.actions)==5
    audit=policy.last_input['image_budget']
    assert audit['ordinary_images']==3 and audit['retrieved_images']==1
    assert client.sent[-1][0][-1]['content'][0]['text'].startswith('Instruction:')
    require_english(policy.last_input)


def test_move_eef_uses_native_gripper_execution(sdk):
    policy,budget=make_policy(sdk,Client([response('move_eef',{'targets':{'left_gripper':0.},'note':'Close the left gripper.'})]))
    chunk=policy.act(observation(sdk))
    assert chunk.actions and budget.used==1
    assert all(set(row.data)=={'left_arm_joint_state','right_arm_joint_state','left_ee_joint_state','right_ee_joint_state'} for row in chunk.actions)


def test_measured_arrival_survives_final_current_observation_replay(sdk):
    import numpy as np
    client=Client([response('wait_steps',{'steps':1})])
    policy,_=make_policy(sdk,client)
    policy._last_targets={'left':np.array([-.2,0.,1.,1.,0.,0.,0.])}
    policy.act(observation(sdk))
    assert 'Arrival check for the previous move_eef target:' in client.sent[0][0][-1]['content'][0]['text']


def test_give_up_stops_without_a_synthetic_action(sdk):
    policy,budget=make_policy(sdk,Client([response('give_up',dict(reason='Task cannot be recovered.',hindsight='A different grasp was needed.'))]))
    with pytest.raises(PolicyStop,match='Task cannot be recovered'):policy.act(observation(sdk))
    assert budget.used==1


def test_150_decisions_then_harness_stop_without_151st_request(sdk):
    client=Client(lambda n:response('memory_resolve',dict(query='',event_ref=None,camera='head',cursor=0),n))
    policy,budget=make_policy(sdk,client)
    with pytest.raises(CallBudgetExhausted):policy.act(observation(sdk))
    assert len(client.sent)==budget.used==150


def test_failed_request_does_not_spend_the_150th_completed_decision(sdk):
    budget=DecisionBudget();budget.used=149
    client=Client([RuntimeError('synthetic failed decision')])
    policy,_=make_policy(sdk,client,budget=budget)
    with pytest.raises(RuntimeError, match='synthetic failed decision'):policy.act(observation(sdk))
    assert budget.used==149 and len(client.sent)==1


def test_cjk_model_feedback_cannot_reenter_a_second_request(sdk):
    client=Client([response('move_eef',{'targets':{'left_gripper':0.},'note':'\u4e2d\u6587'})])
    policy,budget=make_policy(sdk,client)
    with pytest.raises(ValueError,match='CJK'):policy.act(observation(sdk))
    assert len(client.sent)==1 and budget.used==1


def test_budget_binding_requires_exact_fresh_150_and_frozen_limits():
    integration=CommonInputIntegration.__new__(CommonInputIntegration)
    for budget in (DecisionBudget(151),DecisionBudget(149)):
        with pytest.raises(ValueError):integration.bind_decision_budget(budget)
    integration.physics_ticks=100
    with pytest.raises(ValueError):integration.validate_limits(EpisodeLimits(104,10,'coding_agent',AGENT_BUDGET_VERSION))
    with pytest.raises(ValueError):integration.validate_limits(EpisodeLimits(100,10))
    integration.validate_limits(EpisodeLimits(100,10,'coding_agent',AGENT_BUDGET_VERSION))


def test_runner_exact_wait_tail_and_score_at_stop_do_not_advance_extra_ticks(tmp_path,monkeypatch):
    monkeypatch.setattr('mind2act_eval.runner.observation.observation_ticks',lambda spec:121)
    spec,task,ctx,events=fixture(tmp_path)
    class Adapter:
        score_at_stop=True
        def reset(self,seed):self.index=0
        def act(self,obs):
            if self.index==2:raise PolicyStop('give_up','CPU stop')
            self.index+=1
            return Action((0.,)*6,(0.,)*6,0.,0.)
        @property
        def action_physics_ticks(self):return 4 if self.index==1 else 1
        def close(self):pass
    result=run_episode(spec,task,ctx,Adapter(),consumer_kind=ConsumerKind.MODEL,limits=EpisodeLimits(100,10))
    assert result.infra_valid,result.error
    assert result.ticks==5 and result.score==25.
    assert [tick for kind,tick in events if kind=='step']==list(range(1,6))


def test_native_adapter_accepts_a_planned_one_tick_wait_tail(sdk):
    from mind2act_eval.contracts import ClockSpec, Observation
    client=Client([response('wait_steps',{'steps':5}),response('give_up',dict(reason='Finished CPU check.',hindsight='None.'),2)])
    policy,budget=make_policy(sdk,client)
    integration=NS(policy=policy,reset=lambda **kw:policy.reset(),observe=lambda public:observation(sdk,public.tick),
        act=policy.act,confirm_executed=policy.confirm_executed,close=lambda:None,result_metadata=lambda:{})
    adapter=CommonInputAdapter(integration)
    adapter.reset(1);adapter.validate_clock(ClockSpec())
    public=lambda tick:Observation(tick,tick/120,(0.,)*6,(0.,)*6,1.,1.)
    adapter.act(public(0));assert adapter.action_physics_ticks==4
    adapter.confirm_action_executed(4)
    adapter.act(public(4));assert adapter.action_physics_ticks==1
    adapter.confirm_action_executed(1)
    assert adapter._ready
    with pytest.raises(PolicyStop):adapter.act(public(5))
    assert budget.used==2


def test_p2_hard_actual_recipe_uses_final_stack_rules(sdk):
    client = Client([response('wait_steps', {'steps': 1})])
    policy, budget = make_policy(sdk, client, task='P2', tier='hard')
    policy.act(observation(sdk,task='P2',tier='hard'))
    goal = client.sent[0][0][1]['content']
    assert public_instruction(goal) == delivered_instruction('P2','hard')
    assert 'A distractor in the stack is wrong.' in goal
    assert goal.endswith(RECIPES['P2']['hard'].rstrip())
    assert 'Let p = correct layers counted from the bottom.' in goal
    for obsolete in ('moving or using', 'do not move or use', 'unused and unmoved',
                     'displaced by at most 1 mm'):
        assert obsolete not in goal
    assert budget.used == 1


@pytest.mark.parametrize('instruction', ['', '   ', '中文任务', 'Привет мир'])
def test_invalid_runtime_e2e_instruction_fails_before_any_provider_call(sdk, instruction):
    client = Client([response('wait_steps', {'steps': 1})])
    policy, budget = make_policy(sdk, client)
    with pytest.raises(ValueError):
        policy.act(observation(sdk, instruction=instruction))
    assert client.sent == [] and budget.used == 0


def test_runtime_goal_keeps_e2e_instruction_verbatim(sdk):
    instruction = '  Read the displayed scale.\nSet the dial to the target.  '
    client = Client([response('wait_steps', {'steps': 1})])
    policy, _ = make_policy(sdk, client)
    policy.act(observation(sdk, instruction=instruction))
    messages, _ = client.sent[0]
    assert messages[1]['content'].startswith('Goal: ' + instruction + '\n\nTASK RECIPE:\n')
    assert messages[-1]['content'][0]['text'].startswith('Instruction: ' + instruction + '\n\n')


def test_missing_runtime_e2e_instruction_fails_before_any_provider_call(sdk):
    client = Client([response('wait_steps', {'steps': 1})])
    policy, budget = make_policy(sdk, client)
    current = replace(observation(sdk), instruction=None)
    with pytest.raises(ValueError, match='runtime E2E instruction'):
        policy.act(current)
    assert client.sent == [] and budget.used == 0
