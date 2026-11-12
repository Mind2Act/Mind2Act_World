"""CPU-only seed boundary regressions through real adapters and runner."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mind2act_eval.adapters import g05_wire, xpolicy_rpc
from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter, create
from mind2act_eval.contracts import ConsumerKind, to_json
from mind2act_eval.runner import EpisodeLimits, run_episode
from test_xpolicy_result import synthetic_episode
from test_xpolicylab import pack, unpack


class NativeWire:
    """In-memory binary peer; real G05NativeTransport owns reset and codec."""
    def __init__(self):
        self.requests = []
        self.closed = False

    def send(self, blob):
        self.requests.append(unpack(blob))

    def recv(self, timeout):
        if not self.requests:
            return pack({'action_steps': 1})
        if self.requests[-1] == {'__reset__': True}:
            return pack({'__reset__': True})
        return pack({'action': {'left_arm': np.zeros(6), 'right_arm': np.zeros(6),
                     'left_gripper': np.ones(1), 'right_gripper': np.ones(1)}, 'need_obs': True})

    def close(self):
        self.closed = True


@pytest.mark.parametrize('seed', [0, 2**32 - 1, 2**32, 37896428376987, 2**63 - 1])
def test_native_runner_preserves_seed_identity_without_claiming_rng_seeded(tmp_path, monkeypatch, seed):
    wire = NativeWire()
    monkeypatch.setattr(g05_wire, 'connect', lambda *args, **kwargs: wire)
    spec, task, ctx = synthetic_episode(tmp_path)
    spec = replace(spec, seed=seed)
    adapter = create({'url': 'ws://in-memory.invalid', 'action_steps': 1})
    adapter.transport._row = 7
    adapter.transport.needs_observation = False
    result = run_episode(spec, task, ctx, adapter, consumer_kind=ConsumerKind.DIAGNOSTIC,
                         limits=EpisodeLimits(4, 10))

    assert result.infra_valid, result.error
    assert result.ticks == 4 and result.model_tested is False
    assert spec.seed == to_json(spec)['seed'] == seed
    stored = json.loads((tmp_path / 'result.json').read_text())
    for metadata in (result.policy_metadata, stored['policy_metadata'], adapter.result_metadata()):
        assert metadata['episode_seed'] == seed
        assert metadata['reset_confirmed'] is True
        assert metadata['capabilities']['episode_seed_confirmed'] is False
    assert wire.requests[0] == {'__reset__': True}  # No fabricated/derived wire seed.
    assert len(wire.requests) == 2 and wire.closed
    assert adapter.transport._row == 1 and adapter.transport.needs_observation is True


def test_native_large_seed_still_requires_cache_reset_acknowledgement(monkeypatch):
    wire = NativeWire()
    monkeypatch.setattr(g05_wire, 'connect', lambda *args, **kwargs: wire)
    adapter = create({'url': 'ws://in-memory.invalid', 'action_steps': 1})
    wire.recv = lambda timeout: pack({'__reset__': False} if wire.requests else {'action_steps': 1})
    try:
        with pytest.raises(ValueError, match='acknowledge cache reset'):
            adapter.reset(37896428376987)
        metadata = adapter.result_metadata()
        assert metadata['episode_seed'] == 37896428376987
        assert metadata['reset_confirmed'] is False
        assert metadata['capabilities']['episode_seed_confirmed'] is False
        assert wire.closed
    finally:
        adapter.close()


@pytest.mark.parametrize('seed', [True, False, -1, 1.0, '1', None, np.int64(1)])
def test_native_rejects_invalid_seed_before_reset_wire(monkeypatch, seed):
    wire = NativeWire()
    monkeypatch.setattr(g05_wire, 'connect', lambda *args, **kwargs: wire)
    adapter = create({'url': 'ws://in-memory.invalid', 'action_steps': 1})
    try:
        with pytest.raises(ValueError, match='seed'):
            adapter.reset(seed)
        assert wire.requests == []
        assert adapter.result_metadata()['reset_confirmed'] is False
    finally:
        adapter.close()


IDENTITY = dict(model='cpu-test-only', protocol_version=xpolicy_rpc.PROTOCOL_VERSION,
                source_commit='a'*40, checkpoint_sha256='b'*64, stats_sha256='c'*64,
                config_sha256='d'*64, action_horizon=1)


class RPCPeer:
    def __init__(self):
        self.calls = []

    def call(self, method, payload=None):
        self.calls.append((method, payload))
        if method == 'get_metadata':
            return dict(IDENTITY, **xpolicy_rpc.ACTION_CONTRACT, ready=True)

    def close(self):
        pass


@pytest.mark.parametrize('seed', [0, 2**32 - 1, 2**32, 37896428376987, 2**63 - 1])
def test_rpc_never_sends_the_episode_seed_and_keeps_it_as_identity(monkeypatch, seed):
    peer = RPCPeer()
    monkeypatch.setattr(xpolicy_rpc, 'create_client', lambda *args, **kwargs: peer)
    adapter = XPolicyLabAdapter.xpolicy_rpc('ws://in-memory.invalid', expected_identity=IDENTITY)
    try:
        adapter.reset(seed)
        assert peer.calls == [('get_metadata', None), ('reset', None)]
        metadata = adapter.result_metadata()
        assert metadata['episode_seed'] == seed and 'model_seed' not in metadata
        assert metadata['reset_confirmed'] is True
        assert metadata['capabilities']['episode_seed_confirmed'] is False
        assert metadata['capabilities']['episode_seed_identity_only'] is True
    finally:
        adapter.close()


@pytest.mark.parametrize('seed', [True, -1, 1.0, '1'])
def test_invalid_episode_seed_is_rejected_before_any_call(monkeypatch, seed):
    peer = RPCPeer()
    monkeypatch.setattr(xpolicy_rpc, 'create_client', lambda *args, **kwargs: peer)
    adapter = XPolicyLabAdapter.xpolicy_rpc('ws://in-memory.invalid', expected_identity=IDENTITY)
    try:
        with pytest.raises(ValueError, match='nonnegative integer'):
            adapter.reset(seed)
        assert peer.calls == []
        with pytest.raises(ValueError, match='nonnegative integer'):
            XPolicyLabAdapter(object()).reset(seed)
    finally:
        adapter.close()
