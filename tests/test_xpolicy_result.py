"""Synthetic runner test verifies adapter provenance reaches actual result.json."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mind2act_eval.adapters.xpolicylab import create
from mind2act_eval.contracts import BuildContext, ClockSpec, ConsumerKind, Health, InstanceSpec, NativeEvidence, Score
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.sim.demo import DemoRobot, DemoSimulator
from mind2act_eval.tasks import TaskDefinition
from test_xpolicylab import model_service, observation, pack, unpack


def synthetic_episode(tmp_path, clock=ClockSpec()):
    class Scene:
        def observe(self, tick):
            return replace(observation(), tick=tick, sim_time_s=tick/clock.physics_hz)
        def evidence(self, tick):
            return NativeEvidence('synthetic','one',tick,{})
        def health(self):
            return Health(True)
        def is_complete(self):
            return False
        def close(self):
            pass
    spec = InstanceSpec('synthetic','one','easy','synthetic-source','v1','none','v1',seed=42,clock=clock)
    task = TaskDefinition('synthetic',lambda spec,ctx:Scene(),lambda e:Score(25.))
    ctx = BuildContext(DemoSimulator(),DemoRobot(),None,tmp_path)
    return spec, task, ctx


def test_native_chunks_interpolate_without_early_inference_or_future_observation(tmp_path):
    requests = []
    def policy(ws):
        ws.send(pack({'action_steps': 2}))
        row = 0
        for blob in ws:
            request = unpack(blob)
            if request.get('__reset__'):
                row = 0
                ws.send(pack({'__reset__': True}))
                continue
            requests.append(request)
            ws.send(pack({'action': {'left_arm': np.full(6, row * 4, np.float32),
                                     'right_arm': np.zeros(6),
                                     'left_gripper': np.array([row % 2], np.float32),
                                     'right_gripper': np.ones(1)}, 'need_obs': row % 2 == 1}))
            row += 1
    spec, task, ctx = synthetic_episode(tmp_path)
    original_build = task.build
    def build(spec, ctx):
        scene = original_build(spec, ctx)
        scene.observe = lambda tick: replace(observation(), tick=tick, sim_time_s=tick/120,
                                              instruction=f"tick-{tick}")
        return scene
    task = TaskDefinition('synthetic', build, task.scorer)
    applied = []
    original = ctx.robot.apply
    def apply(action):
        applied.append(action)
        original(action)
    ctx.robot.apply = apply
    with model_service(policy) as url:
        result = run_episode(spec, task, ctx, create(dict(url=url, action_steps=2)),
                             consumer_kind=ConsumerKind.MODEL, limits=EpisodeLimits(16, 10))
    assert result.infra_valid, result.error
    assert result.model_tested
    assert [a.left_arm[0] for a in applied] == [0., 1., 2., 3.] + [4.] * 4 + [8., 9., 10., 11.] + [12.] * 4
    assert [a.left_gripper for a in applied] == [0., .25, .5, .75] + [1.] * 4 + [0., .25, .5, .75] + [1.] * 4
    assert len(requests) == 4
    assert requests[1] == requests[3] == {}
    assert requests[0]['state']['left_arm'][0] == 0.
    assert requests[0]['task'] == 'tick-0'
    assert requests[2]['task'] == 'tick-8'


@pytest.mark.parametrize('bad_action', [False, True])
def test_native_adapter_records_checkpoint_and_seed_in_persisted_result(tmp_path, bad_action):
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            request = unpack(blob)
            if request.get('__reset__'):
                ws.send(pack({'__reset__': True}))
            else:
                parts = {} if bad_action else {'left_arm':np.zeros(6), 'right_arm':np.zeros(6),
                    'left_gripper':np.ones(1), 'right_gripper':np.ones(1)}
                ws.send(pack({'action':parts, 'need_obs':True}))
    spec, task, ctx = synthetic_episode(tmp_path)
    with model_service(policy) as url:
        # Same entrypoint as --adapter mind2act_eval.adapters.xpolicylab:create.
        adapter = create(dict(url=url,action_steps=1,checkpoint='synthetic-test-only',checkpoint_sha256='a'*64))
        result = run_episode(spec,task,ctx,adapter,consumer_kind=ConsumerKind.DIAGNOSTIC,
                             limits=EpisodeLimits(4,10))
    assert result.infra_valid is (not bad_action), result.error
    stored = json.loads((tmp_path/'result.json').read_text())
    assert stored['score'] == (None if bad_action else 25.)
    assert stored['model_tested'] is False
    metadata = stored['policy_metadata']
    assert metadata['transport'] == 'g05-native'
    assert metadata['episode_seed'] == 42
    assert metadata['checkpoint'] == 'synthetic-test-only'
    assert metadata['checkpoint_sha256'] == 'a'*64
    assert metadata['capabilities']['episode_seed_confirmed'] is False
    assert metadata['capabilities']['remote_model_identity_verified'] is False
    assert adapter.result_metadata()['episode_seed'] == 42


@pytest.mark.parametrize('clock,accepted', [
    (ClockSpec(120,30,30), True),
    (ClockSpec(960,30,30), True),
    (ClockSpec(120,120,30), False),
    (ClockSpec(960,60,30), False),
    (ClockSpec(960,30,15), False),
])
def test_native_runner_rejects_clock_mismatch_before_first_action(tmp_path, clock, accepted):
    requests = []
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            request = unpack(blob)
            if request.get('__reset__'):
                ws.send(pack({'__reset__': True}))
            else:
                requests.append(request)
                ws.send(pack({'action': {'left_arm':np.zeros(6), 'right_arm':np.zeros(6),
                    'left_gripper':np.ones(1), 'right_gripper':np.ones(1)}, 'need_obs':True}))
    spec, task, ctx = synthetic_episode(tmp_path, clock)
    with model_service(policy) as url:
        adapter = create(dict(url=url,action_steps=1,checkpoint='synthetic-test-only'))
        result = run_episode(spec,task,ctx,adapter,consumer_kind=ConsumerKind.DIAGNOSTIC,
                             limits=EpisodeLimits(clock.ticks_per_action,10))
    stored = json.loads((tmp_path/'result.json').read_text())
    if accepted:
        assert result.infra_valid, result.error
        assert len(requests) == 1
        assert requests[0]['frequency'] == 30
        assert stored['policy_metadata']['clock']['validated'] is True
    else:
        assert not result.infra_valid
        assert result.score is None and stored['score'] is None
        assert 'clock' in result.error.lower()
        assert requests == []  # No action request may enter the model service.
        assert stored['policy_metadata']['checkpoint'] == 'synthetic-test-only'
        assert stored['policy_metadata']['episode_seed'] == 42
        assert stored['model_tested'] is False
        assert stored['policy_metadata']['clock']['requested'] == {
            'physics_hz':clock.physics_hz, 'control_hz':clock.control_hz, 'rgb_hz':clock.rgb_hz}
        assert stored['policy_metadata']['clock']['validated'] is False


@pytest.mark.parametrize('physics_hz,ticks_per_row', [(120,4), (960,32)])
def test_native_two_rows_each_keep_one_thirtieth_second(tmp_path, physics_hz, ticks_per_row):
    requests = []
    def policy(ws):
        ws.send(pack({'action_steps':1}))
        for blob in ws:
            request = unpack(blob)
            if request.get('__reset__'):
                ws.send(pack({'__reset__':True}))
                continue
            requests.append(request)
            ws.send(pack({'action':{'left_arm':np.full(6,len(requests),np.float32),
                'right_arm':np.zeros(6), 'left_gripper':np.ones(1), 'right_gripper':np.ones(1)},
                'need_obs':True}))
    spec, task, ctx = synthetic_episode(tmp_path, ClockSpec(physics_hz,30,30))
    applied = []
    class Robot(DemoRobot):
        def apply(self, action):
            applied.append(action.left_arm[0])
            super().apply(action)
    ctx = replace(ctx,robot=Robot())
    with model_service(policy) as url:
        adapter = create(dict(url=url,action_steps=1))
        result = run_episode(spec,task,ctx,adapter,consumer_kind=ConsumerKind.DIAGNOSTIC,
            limits=EpisodeLimits(2*ticks_per_row,10))
    assert result.infra_valid, result.error
    assert result.ticks == 2*ticks_per_row
    assert applied == [1.]*ticks_per_row + [2.]*ticks_per_row
    assert len(requests) == 2 and all(r['frequency']==30 for r in requests)
    metadata = json.loads((tmp_path/'result.json').read_text())['policy_metadata']['clock']
    assert metadata['ticks_per_action'] == ticks_per_row
    assert metadata['action_period_s'] == pytest.approx(1/30)
