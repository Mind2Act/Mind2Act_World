"""Real canonical WsModelClient/PolicyServer, with a deterministic model only."""
import asyncio
from contextlib import contextmanager
from threading import Thread

import numpy as np
import pytest

pytest.importorskip('client_server.ws.model_server', reason='install the pinned xpolicy extra')
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
from test_xpolicylab import observation

IDENTITY = dict(model='deterministic-test-only',
    protocol_version='xpolicylab-ws@128e626278ef753fa8a86b16e3425a6359388b4c',
    source_commit='a'*40, checkpoint_sha256='b'*64, stats_sha256='c'*64,
    config_sha256='d'*64, action_horizon=2)


class DeterministicPolicy:
    def get_metadata(self):
        return dict(IDENTITY, ready=True, action_space='joint14', arm_units='rad',
                    gripper_range=[0,1], control_hz=30, raw_state_dim=14, raw_action_dim=14,
                    environment_ticks_per_row=4, output_row_shift=0, cache_owner='environment_client')

    prepared = False

    def prepare_episode(self, payload):  # legacy contract: must never be called
        self.prepared = True

    def reset(self):
        self.obs = None
        return {}

    def update_obs(self, payload):
        assert set(payload) == {'images', 'state', 'task', 'frequency', 'embodiment_type'}
        np.testing.assert_array_equal(payload['images']['cam_high'], [[[11,44]], [[22,55]], [[33,66]]])
        assert payload['task'] == 'Move the cube'
        self.obs = payload

    def get_action(self):
        assert self.obs is not None
        self.obs = None
        return dict(actions=np.array([[.5]*6+[.25]+[-.5]*6+[.75],
                                       [1.5]*6+[.5]+[-1.5]*6+[1.]], np.float32),
                    need_obs=True, model_wall_s=.01)


@contextmanager
def rpc_service(policy):
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = PolicyServer(policy, PolicyServerConfig(host='127.0.0.1', port=0))
    asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
    try:
        yield server.url
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_official_rpc_adapter_consumes_whole_chunk_and_resets_cache():
    with rpc_service(DeterministicPolicy()) as url:
        adapter = XPolicyLabAdapter.xpolicy_rpc(url, expected_identity=IDENTITY)
        try:
            adapter.reset(19)
            assert adapter.act(observation()).left_arm == (.5,)*6
            assert adapter.act(observation()).left_arm == (1.5,)*6
            adapter.act(observation())
            adapter.reset(23)
            assert adapter.act(observation()).left_arm == (.5,)*6
        finally:
            adapter.close()


@pytest.mark.parametrize('field,value', [('model','wrong-model'), ('checkpoint_sha256','e'*64),
    ('cache_owner','model_service'), ('control_hz',120)])
def test_rpc_refuses_different_model_or_action_contract(field, value):
    class WrongPolicy(DeterministicPolicy):
        def get_metadata(self):
            return dict(super().get_metadata(), **{field: value})
    with rpc_service(WrongPolicy()) as url:
        adapter = XPolicyLabAdapter.xpolicy_rpc(url, expected_identity=IDENTITY)
        try:
            with pytest.raises(ValueError, match=field):
                adapter.reset(1)
            with pytest.raises(RuntimeError, match='reset'):
                adapter.act(observation())
            flags = adapter.result_metadata()['capabilities']
            assert flags['remote_model_identity_verified'] is False
            assert flags['episode_seed_confirmed'] is False
        finally:
            adapter.close()


@pytest.mark.parametrize('kind', ['shape', 'nan', 'timing'])
def test_rpc_malformed_chunk_cannot_yield_even_its_first_row(kind):
    class WrongPolicy(DeterministicPolicy):
        def get_action(self):
            reply = super().get_action()
            if kind == 'shape':
                reply['actions'] = np.zeros((3,14), np.float32)
            elif kind == 'nan':
                reply['actions'][1,-1] = np.nan
            else:
                reply['model_wall_s'] = -1
            return reply
    with rpc_service(WrongPolicy()) as url:
        adapter = XPolicyLabAdapter.xpolicy_rpc(url, expected_identity=IDENTITY)
        try:
            adapter.reset(1)
            with pytest.raises(ValueError):
                adapter.act(observation())
            with pytest.raises(RuntimeError, match='reset'):
                adapter.act(observation())
        finally:
            adapter.close()


def test_rpc_does_not_send_a_seed_and_records_identity_only():
    policy = DeterministicPolicy()
    with rpc_service(policy) as url:
        adapter = XPolicyLabAdapter.xpolicy_rpc(url, expected_identity=IDENTITY)
        try:
            adapter.reset(2**40)
            metadata = adapter.result_metadata()
            assert policy.prepared is False
            assert metadata['episode_seed'] == 2**40 and metadata['reset_confirmed'] is True
            assert metadata['capabilities']['remote_model_identity_verified'] is True
            assert metadata['capabilities']['episode_seed_confirmed'] is False
        finally:
            adapter.close()
        assert adapter.result_metadata() == metadata


@pytest.mark.parametrize('physics_hz,ticks_per_row', [(120,4), (960,32)])
@pytest.mark.parametrize('tick_hint', [None,4,32])
def test_official_rpc_rows_are_timed_by_runner_not_legacy_tick_hint(tmp_path,physics_hz,ticks_per_row,tick_hint):
    import json
    from dataclasses import replace
    from mind2act_eval.contracts import ClockSpec,ConsumerKind
    from mind2act_eval.runner import run_episode,EpisodeLimits
    from mind2act_eval.sim.demo import DemoRobot
    from test_xpolicy_result import synthetic_episode
    spec,task,ctx = synthetic_episode(tmp_path,ClockSpec(physics_hz,30,30))
    applied = []
    class Robot(DemoRobot):
        def apply(self,action):
            applied.append(action.left_arm[0])
            super().apply(action)
    ctx = replace(ctx,robot=Robot())
    class Policy(DeterministicPolicy):
        def get_metadata(self):
            metadata = super().get_metadata()
            if tick_hint is None:
                metadata.pop('environment_ticks_per_row')
            else:
                metadata['environment_ticks_per_row'] = tick_hint
            return metadata
    with rpc_service(Policy()) as url:
        adapter = XPolicyLabAdapter.xpolicy_rpc(url,expected_identity=IDENTITY)
        result = run_episode(spec,task,ctx,adapter,consumer_kind=ConsumerKind.DIAGNOSTIC,
            limits=EpisodeLimits(2*ticks_per_row,10))
    assert result.infra_valid,result.error
    assert applied == [.5 + k / ticks_per_row for k in range(ticks_per_row)] + [1.5]*ticks_per_row
    metadata=json.loads((tmp_path/'result.json').read_text())['policy_metadata']
    assert metadata['clock']['ticks_per_action'] == ticks_per_row
    assert metadata['clock']['action_period_s'] == pytest.approx(1/30)
    assert metadata['service_environment_ticks_per_row_hint'] == tick_hint


@pytest.mark.parametrize('control_hz,rgb_hz', [(60,30),(30,15)])
def test_rpc_runner_rejects_wrong_io_rates_before_observation(tmp_path,control_hz,rgb_hz):
    from mind2act_eval.contracts import ClockSpec,ConsumerKind
    from mind2act_eval.runner import run_episode,EpisodeLimits
    from test_xpolicy_result import synthetic_episode
    requests=[]
    class Policy(DeterministicPolicy):
        def update_obs(self,payload):
            requests.append(payload)
            super().update_obs(payload)
    spec,task,ctx=synthetic_episode(tmp_path,ClockSpec(960,control_hz,rgb_hz))
    with rpc_service(Policy()) as url:
        adapter=XPolicyLabAdapter.xpolicy_rpc(url,expected_identity=IDENTITY)
        result=run_episode(spec,task,ctx,adapter,consumer_kind=ConsumerKind.DIAGNOSTIC,
                           limits=EpisodeLimits(32,10))
    assert not result.infra_valid and result.score is None
    assert result.model_tested is False and 'clock' in result.error.lower()
    assert requests == []
    assert result.policy_metadata['episode_seed'] == 42
