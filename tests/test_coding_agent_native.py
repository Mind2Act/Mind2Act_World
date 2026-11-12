"""Native-shaped local fixtures; no model, GPU, planner or private inputs."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mind2act_eval.adapters.coding_agent_native import NativeCodingAgentAdapter, create
from mind2act_eval.contracts import ClockSpec, ConsumerKind, PolicyStop, Score, Termination
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.tasks import TaskDefinition
from test_runner import fixture


def row(value=0., **meta):
    return NS(data={"left_arm_joint_state": np.full(6, value),
                    "right_arm_joint_state": np.full(6, -value),
                    "left_ee_joint_state": np.ones(1),
                    "right_ee_joint_state": np.zeros(1)}, meta=meta)


def chunk(*rows, hz=30.):
    return NS(actions=list(rows), control_hz=hz)


class Integration:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls, self.confirmed, self.observations = [], [], []
        self.closed = 0

    def reset(self, *, seed):
        self.seed = seed
        self.index = 0

    def observe(self, public):
        self.observations.append(public)
        assert type(public).__name__ == "Observation"
        assert not hasattr(public, "evidence")
        return public

    def act(self, observation):
        self.calls.append(observation.tick)
        value = self.chunks[self.index]
        self.index += 1
        if isinstance(value, BaseException):
            raise value
        return value

    def confirm_executed(self, count):
        self.confirmed.append(count)

    def close(self):
        self.closed += 1

    def result_metadata(self):
        return {"model": "synthetic-native-shaped-fixture", "network_calls": 0, "seed_confirmed": False}


def run(tmp_path, chunks, *, ticks=100, consumer=ConsumerKind.MODEL):
    spec, task, ctx, _ = fixture(tmp_path)
    task = TaskDefinition(task.task_id, task.build, lambda evidence: Score(0.))
    integration = Integration(chunks)
    adapter = NativeCodingAgentAdapter(integration)
    applied = []
    ctx.robot.apply = applied.append
    result = run_episode(spec, task, ctx, adapter, consumer_kind=consumer, limits=EpisodeLimits(ticks, 10))
    return result, integration, adapter, applied


@pytest.mark.parametrize("reason", ["done", "give_up"])
def test_complete_native_chunks_then_terminal_hold_and_independent_zero_score(tmp_path, reason):
    result, integration, adapter, applied = run(tmp_path, [
        chunk(row(0.), row(1.), row(2., chunk_final=True)),
        chunk(row(2., request_stop=True, stop_reason=reason, stop_detail="local test")),
    ])
    assert result.infra_valid, result.error
    assert result.model_tested and result.score == 0.
    assert result.termination == Termination.POLICY_STOP
    assert result.ticks == 16
    assert integration.calls == [0, 12]
    assert integration.confirmed == [3, 1]
    assert integration.closed == 1
    assert [a.left_arm[0] for a in applied] == [0., .25, .5, .75, 1., 1.25, 1.5, 1.75] + [2.] * 8
    trace = result.policy_metadata["action_trace"]
    assert [r["executed_physics_ticks"] for r in trace] == [4, 4, 4, 4]
    assert result.policy_metadata["policy_stop"]["reason"] == reason


def test_validate_entire_chunk_before_first_physics_action(tmp_path):
    result, integration, _, applied = run(tmp_path, [chunk(row(0.), row(float("nan")))])
    assert not result.infra_valid and result.score is None
    assert not applied and not result.model_tested
    assert len(integration.calls) == 1 and integration.closed == 1


@pytest.mark.parametrize("bad", [chunk(hz=30), chunk(row(), hz=25),
    chunk(row(request_stop=True, stop_reason="give_up"), row()),
    chunk(row(request_stop=True, stop_reason="award_success")),
    chunk(row(request_stop="false"))])
def test_reject_native_protocol_violations(tmp_path, bad):
    result, _, _, applied = run(tmp_path, [bad])
    assert not result.infra_valid and result.score is None and not applied


def test_native_baseexception_becomes_durable_invalid_result(tmp_path):
    class NativeInfrastructureFailure(BaseException):
        pass
    result, integration, _, _ = run(tmp_path, [NativeInfrastructureFailure("provider failed")])
    assert not result.infra_valid and result.score is None
    assert "NativeInfrastructureFailure: provider failed" in result.error
    assert integration.closed == 1


def test_truncated_control_hold_is_recorded_without_another_model_decision(tmp_path):
    result, integration, _, applied = run(tmp_path, [chunk(row(), row(1.), row(2.))], ticks=6)
    assert result.infra_valid and result.ticks == len(applied) == 6
    assert integration.calls == [0]
    assert [r["executed_physics_ticks"] for r in result.policy_metadata["action_trace"]] == [4, 2]
    assert result.policy_metadata["unconsumed_rows"] == 1


def test_clock_mismatch_before_native_call(tmp_path):
    from dataclasses import replace
    spec, task, ctx, _ = fixture(tmp_path)
    adapter = NativeCodingAgentAdapter(Integration([]))
    spec = replace(spec, clock=ClockSpec(control_hz=120))
    result = run_episode(spec, task, ctx, adapter)
    assert not result.infra_valid and not result.model_tested
    assert adapter.integration.calls == []


def test_stopiteration_remains_invalid_for_models(tmp_path):
    result, _, _, _ = run(tmp_path, [StopIteration()])
    assert not result.infra_valid and "only expert replay" in result.error


def test_policy_stop_cannot_be_used_by_expert_replay(tmp_path):
    result, _, _, _ = run(tmp_path, [PolicyStop("done")], consumer=ConsumerKind.EXPERT_REPLAY)
    assert not result.infra_valid and not result.model_tested


def test_factory_has_no_implicit_native_configuration():
    for config in ({}, {"integration_factory": "bad"}, {"api_key": "must-not-be-accepted"}):
        with pytest.raises(ValueError):
            create(config)


def test_reset_discards_native_fifo_and_requires_fresh_execution_confirmation(tmp_path):
    from mind2act_eval.contracts import Observation
    integration = Integration([chunk(row(1.), row(2.))])
    adapter = NativeCodingAgentAdapter(integration)
    observation = Observation(0, 0., (0.,) * 6, (0.,) * 6, 1., 1.)
    adapter.reset(0)
    adapter.validate_clock(ClockSpec())
    assert adapter.act(observation).left_arm[0] == 1.
    with pytest.raises(RuntimeError, match="not been confirmed"):
        adapter.act(observation)
    adapter.reset(1)
    adapter.validate_clock(ClockSpec())
    assert adapter.act(observation).left_arm[0] == 1.
    assert adapter.result_metadata()["returned_rows"] == 1
    assert integration.seed == 1
