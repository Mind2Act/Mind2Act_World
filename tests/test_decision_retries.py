"""Pinned CodingAgent retry loop with fake replies: no network or simulator."""
import json
from types import SimpleNamespace

import pytest

from common_input_fixture import sdk, response, Client, make_policy, observation
from coding_payload_fixture import valid_input
from mind2act_eval.adapters.coding_agent_client import LoggedNativeClient
from mind2act_eval.budgets import AGENT_BUDGET_VERSION, CallBudgetExhausted, DecisionBudget
from mind2act_eval.contracts import Termination
from mind2act_eval.runner import EpisodeLimits, run_episode
from test_runner import fixture


def audited_client(sdk, tmp_path, monkeypatch):
    joint = sdk[0]
    delays = []
    monkeypatch.setattr(joint, 'time', SimpleNamespace(sleep=delays.append))
    native = joint.AzureAgentClient(model='cpu-fixture', azure_endpoint='https://offline.invalid',
        api_version='fixture', api_key='synthetic-key', max_retries=0)
    path = tmp_path/'ledger.json'
    path.write_text(json.dumps(dict(schema_version=1, limit=10000, attempts=[])))
    client = LoggedNativeClient(native, ledger=path, total_limit=10000, episode='cpu',
        episode_limit=450, attempt_dir=tmp_path/'attempts')
    return client, native, path, delays


def test_native_12_retries_then_reply_counts_once_and_has_no_450_episode_cap(sdk, tmp_path, monkeypatch):
    client, native, ledger, delays = audited_client(sdk, tmp_path, monkeypatch)
    budget = DecisionBudget()
    client.bind_decision_budget(budget)
    seen = []
    def request(*args):
        seen.append(len(seen))
        if len(seen) % 13:
            raise sdk[0].InfrastructureFailure('synthetic retryable error', retryable=True)
        return response('wait_steps', {'steps':1})
    client._request_once = request
    for _ in range(40):
        native.complete(*valid_input())
    assert budget.used == 40 and native.max_retries == budget.retry_limit == 12
    assert len(seen) == len(json.loads(ledger.read_text())['attempts']) == 520
    assert len(delays) == 40*12
    assert client.result_metadata()['episode_attempt_limit'] is None
    budget.used = 150
    with pytest.raises(CallBudgetExhausted):
        native.complete(*valid_input())
    assert len(seen) == 520


def test_retry_exhaustion_is_infra_with_no_decision_spent(sdk, tmp_path, monkeypatch):
    client, native, ledger, delays = audited_client(sdk, tmp_path, monkeypatch)
    (tmp_path/'episode').mkdir()
    spec, task, ctx, _ = fixture(tmp_path/'episode')
    from mind2act_eval.adapters.coding_agent_native import NativeCodingAgentAdapter
    from test_coding_agent_native import Integration
    class AgentIntegration(Integration):
        def bind_decision_budget(self, budget):
            client.bind_decision_budget(budget)
        def act(self, observation):
            native.complete(*valid_input())
            raise AssertionError('All requests must fail')
    def request(*args):
        raise sdk[0].InfrastructureFailure('synthetic retryable error', retryable=True)
    client._request_once = request
    result = run_episode(spec, task, ctx, NativeCodingAgentAdapter(AgentIntegration([])), limits=EpisodeLimits(
        800, 30, 'coding_agent', AGENT_BUDGET_VERSION))
    assert result.termination == Termination.ERROR and not result.infra_valid
    assert result.score is None and result.calls_used == 0 and result.decision_retry_limit == 12
    assert len(json.loads(ledger.read_text())['attempts']) == 13 and len(delays) == 12


def test_nonretryable_failure_does_not_retry_or_spend_decision(sdk, tmp_path, monkeypatch):
    client, native, ledger, delays = audited_client(sdk, tmp_path, monkeypatch)
    budget = DecisionBudget()
    client.bind_decision_budget(budget)
    def request(*args):
        raise sdk[0].InfrastructureFailure('synthetic permanent failure', retryable=False)
    client._request_once = request
    with pytest.raises(sdk[0].InfrastructureFailure):
        native.complete(*valid_input())
    assert budget.used == 0 and not delays
    assert len(json.loads(ledger.read_text())['attempts']) == 1


def test_full_reply_requiring_action_repair_spends_a_decision(sdk):
    client = Client([response('move_eef', {'targets':{}, 'note':'Repair this invalid action.'}),
                     response('wait_steps', {'steps':1})])
    policy, budget = make_policy(sdk, client)
    chunk = policy.act(observation(sdk))
    assert chunk.actions and budget.used == 2 and len(client.sent) == 2
