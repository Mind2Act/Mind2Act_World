"""Final scoring uses T0 for every task without advancing post-actor physics."""
import json

import pytest

from mind2act_eval.adapters.demo import DemoAdapter
from mind2act_eval.contracts import (
    BuildContext, ConsumerKind, Health, InstanceSpec, NativeEvidence, Observation, PolicyStop, Score,
    Termination,
)
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.sim.demo import DemoRobot, DemoSimulator
from mind2act_eval.tasks import TaskDefinition


def window_case(tmp_path, *, task_id='PR4', fall_at=None, complete_at=125):
    spec = InstanceSpec(task_id, 'one', 'hard', 'public', 'v1', 'none', 'v1')
    robot, sim, applied, hooks = DemoRobot(), DemoSimulator(), [], []
    robot.apply = applied.append

    class Scene:
        def before_step(self, tick):
            hooks.append(('before', tick))

        def after_step(self, tick):
            hooks.append(('after', tick))

        def observe(self, tick):
            return Observation(tick, tick / 120, **robot.observe())

        def evidence(self, tick):
            hooks.append(('evidence', tick))
            return NativeEvidence(task_id, 'one', tick, dict(complete=self.is_complete()))

        def health(self):
            return Health(True)

        def is_complete(self):
            return sim.ticks >= complete_at and (fall_at is None or sim.ticks < fall_at)

        def is_success(self):return self.is_complete()
        def events_complete(self):return self.is_complete()

        def close(self):
            pass

    class Actor(DemoAdapter):
        calls = 0

        def act(self, observation):
            self.calls += 1
            assert self.calls == 1, 'environment observation must never request another action'
            return super().act(observation)

    actor = Actor()
    task = TaskDefinition(task_id, lambda spec, ctx: Scene(),
                          lambda evidence: Score(100. if evidence.values['complete'] else 0.))
    return spec, task, BuildContext(sim, robot, None, tmp_path), actor, applied, hooks


@pytest.mark.parametrize('task_id', [f'PR{i}' for i in range(1, 7)] +
                         [f'DR{i}' for i in range(1, 5)] +
                         [f'{group}{i}' for group in 'ESPFM' for i in (1, 2)] +
                         [f'CP{i:02}' for i in range(1, 11)])
@pytest.mark.parametrize('consumer', [ConsumerKind.MODEL, ConsumerKind.EXPERT_REPLAY])
def test_all_tasks_score_t0_without_post_actor_physics(tmp_path, task_id, consumer):
    spec, task, ctx, actor, applied, hooks = window_case(tmp_path, task_id=task_id)
    result = run_episode(spec, task, ctx, actor, consumer_kind=consumer,
                         limits=EpisodeLimits(4, 10))
    assert result.infra_valid, result.error
    assert result.score == 0 and result.ticks == len(applied) == 4
    assert actor.calls == 1
    assert hooks == [(name, tick) for tick in range(1, 5)
                     for name in ('before', 'after', 'evidence')]
    assert not (tmp_path / 'observation-closeout.json').exists()


@pytest.mark.parametrize('consumer', [ConsumerKind.MODEL, ConsumerKind.EXPERT_REPLAY])
def test_stop_and_eof_score_last_actual_row(tmp_path, consumer):
    spec, task, ctx, actor, applied, _ = window_case(tmp_path, complete_at=4)
    original = actor.act
    calls = []
    def act(obs):
        calls.append(obs.tick)
        if obs.tick:
            raise PolicyStop('done') if consumer == ConsumerKind.MODEL else StopIteration()
        return original(obs)
    actor.act = act
    result = run_episode(spec, task, ctx, actor, consumer_kind=consumer,
                         limits=EpisodeLimits(1000, 10))
    assert result.infra_valid and result.score == 100
    assert result.ticks == len(applied) == 4
    assert calls == ([0] if consumer == ConsumerKind.MODEL else [0, 4])
    assert result.termination == (Termination.COMPLETED if consumer == ConsumerKind.MODEL
                                 else Termination.TRAJECTORY_EXHAUSTED)


def test_later_physics_cannot_change_t0_score(tmp_path):
    spec, task, ctx, actor, _, _ = window_case(tmp_path, complete_at=4, fall_at=5)
    result = run_episode(spec, task, ctx, actor, limits=EpisodeLimits(4, 10))
    assert result.infra_valid and result.ticks == 4 and result.score == 100


def test_native_fault_before_t0_remains_infra(tmp_path):
    spec, task, ctx, actor, _, _ = window_case(tmp_path)
    def fail_step(*, render):
        raise RuntimeError('native simulation failed')
    ctx.sim.step = fail_step
    result = run_episode(spec, task, ctx, actor, limits=EpisodeLimits(4, 10))
    assert not result.infra_valid and result.score is None
    assert result.termination == Termination.ERROR
