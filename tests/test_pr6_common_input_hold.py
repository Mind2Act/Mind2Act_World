"""Real CommonInputAdapter scheduling with CPU-only public native rows."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mind2act_eval.adapters.coding_agent_common_input import CommonInputAdapter
from mind2act_eval.adapters.agent_input.templates import VERSION
from mind2act_eval.contracts import PolicyStop, Termination
from mind2act_eval.runner import EpisodeLimits, run_episode
from test_runner import initial_hold_fixture


def native_wait_adapter(events, tail):
    def row(ticks):
        return NS(data={name: np.zeros(6 if 'arm' in name else 1)
                       for name in ('left_arm_joint_state', 'right_arm_joint_state',
                                    'left_ee_joint_state', 'right_ee_joint_state')},
                  meta={'physics_ticks': ticks})

    def observe(public):
        events.append(('native_observe', public.tick))
        return public

    def act(public):
        events.append(('native_act', public.tick))
        if sum(name == 'native_act' for name, _ in events) > 1:
            raise PolicyStop('give_up', 'CPU wait complete')
        return NS(control_hz=30, actions=[row(4), row(tail)])

    integration = NS(
        config={'input_protocol': VERSION},
        reset=lambda **kwargs: None, observe=observe, act=act,
        confirm_executed=lambda count: events.append(('native_confirm', count)),
        close=lambda: None, result_metadata=lambda: {}, policy=NS(),
        bind_public_task=lambda task, difficulty, parameters:
            events.append(('public_binding', (task, difficulty, parameters))))
    return CommonInputAdapter(integration)


@pytest.mark.parametrize('hold_ticks', [599, 600])
@pytest.mark.parametrize('tail', [1, 2, 3, 4])
def test_real_common_input_adapter_hold_boundary_and_exact_wait_tail(
        tmp_path, monkeypatch, hold_ticks, tail):
    # An unwanted closeout would add 121 ticks after the explicit policy stop.
    monkeypatch.setattr('mind2act_eval.runner.observation.observation_ticks', lambda spec: 121)
    spec, task, ctx, events = initial_hold_fixture(tmp_path, hold_ticks)
    adapter = native_wait_adapter(events, tail)
    assert adapter._pending is None
    result = run_episode(spec, task, ctx, adapter, limits=EpisodeLimits(1000, 10))
    assert result.infra_valid, result.error
    assert result.termination == Termination.POLICY_STOP
    assert result.ticks == ctx.sim.ticks == 4 + tail
    assert result.score == 25.
    assert [value for name, value in events if name == 'public_binding'] == [('demo', 'easy', {})]
    assert [tick for name, tick in events if name == 'native_act'] == [0, 4 + tail]
    assert [tick for name, tick in events if name == 'observe'] == [0, 4, 4 + tail]
    assert [count for name, count in events if name == 'native_confirm'] == [2]
    assert [row['observation_tick'] for row in adapter._trace] == [0, 4]
    assert [row['executed_physics_ticks'] for row in adapter._trace] == [4, tail]
    assert [tick for name, tick in events if name == 'evidence'] == list(range(1, result.ticks + 1))
    assert not (tmp_path / 'initial-environment-hold.json').exists()
    assert not (tmp_path / 'observation-closeout.json').exists()


@pytest.mark.parametrize('budget', [13, 600])
def test_real_common_input_adapter_hold_before_first_pending_action(tmp_path, budget):
    spec, task, ctx, events = initial_hold_fixture(tmp_path, 600)
    adapter = native_wait_adapter(events, 1)
    result = run_episode(spec, task, ctx, adapter, limits=EpisodeLimits(budget, 10))
    assert result.infra_valid, result.error
    assert result.ticks == 5 and result.termination == Termination.POLICY_STOP
    assert adapter._pending is None and len(adapter._trace) == 2
    assert [tick for name, tick in events if name == 'native_act'] == [0, 5]
    assert result.model_tested is True
