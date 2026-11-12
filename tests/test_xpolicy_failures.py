import numpy as np
import pytest

from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
from test_xpolicylab import model_service, observation, pack, unpack


@pytest.mark.parametrize('reply', [
    {'error': {'code': 500, 'message': 'model failed'}},
    {'action': {'left_arm': np.zeros(6)}, 'need_obs': True},
    {'action': {'left_arm': np.full(6, np.nan), 'right_arm': np.zeros(6),
                'left_gripper': np.ones(1), 'right_gripper': np.ones(1)}, 'need_obs': True},
    {'action': {}, 'need_obs': False},
])
def test_bad_reply_cannot_be_followed_by_another_action_without_reset(reply):
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            request = unpack(blob)
            ws.send(pack({'__reset__': True} if request.get('__reset__') else reply))
    with model_service(policy) as url:
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=1)
        try:
            adapter.reset(0)
            with pytest.raises((RuntimeError, ValueError)):
                adapter.act(observation())
            with pytest.raises(RuntimeError, match='reset'):
                adapter.act(observation())
        finally:
            adapter.close()


def test_wrong_horizon_is_rejected_on_connect():
    def policy(ws):
        ws.send(pack({'action_steps': 16}))
    with model_service(policy) as url:
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=2)
        try:
            with pytest.raises(ValueError, match='handshake'):
                adapter.reset(0)
        finally:
            adapter.close()


def test_reset_acknowledgement_is_required():
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            ws.send(pack({'__reset__': False}))
    with model_service(policy) as url:
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=1)
        try:
            with pytest.raises(ValueError, match='reset'):
                adapter.reset(0)
            with pytest.raises(RuntimeError, match='reset'):
                adapter.act(observation())
        finally:
            adapter.close()


def test_timeout_poisoned_connection_cannot_consume_a_late_reply():
    from threading import Event
    from websockets.exceptions import ConnectionClosed
    release = Event()
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            request = unpack(blob)
            if request.get('__reset__'):
                ws.send(pack({'__reset__': True}))
            else:
                release.wait(2)
                try:
                    ws.send(pack({'action': {}, 'need_obs': True}))
                except ConnectionClosed:
                    return
    with model_service(policy) as url:
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=1, timeout_s=.1)
        try:
            adapter.reset(0)
            with pytest.raises(TimeoutError):
                adapter.act(observation())
            release.set()
            with pytest.raises(RuntimeError, match='new connection'):
                adapter.reset(1)
        finally:
            release.set()
            adapter.close()


def test_smoke_command_exercises_service_without_scene_or_score(capsys):
    import json
    from mind2act_eval.adapters.xpolicy_smoke import main
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            request = unpack(blob)
            if request.get('__reset__'):
                ws.send(pack({'__reset__': True}))
            else:
                assert set(request) == {'images', 'state', 'task', 'frequency', 'embodiment_type'}
                assert request['images']['cam_high'].shape == (3,224,224)
                ws.send(pack({'action': {'left_arm': np.zeros(6), 'right_arm': np.zeros(6),
                    'left_gripper': np.ones(1), 'right_gripper': np.ones(1)}, 'need_obs': True}))
    with model_service(policy) as url:
        assert main(['--transport','g05-native','--url',url,'--action-steps','1']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['rows_checked'] == 3
    assert report['reset_count'] == 2
    assert report['synthetic_input'] is True and report['score'] is None
    assert report['capabilities']['episode_seed_confirmed'] is False


def test_result_provenance_records_configured_checkpoint_and_requested_seed(capsys):
    def policy(ws):
        ws.send(pack({'action_steps': 1}))
        for blob in ws:
            assert unpack(blob) == {'__reset__': True}
            ws.send(pack({'__reset__': True}))
    with model_service(policy) as url:
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=1,
            checkpoint='synthetic-test-only', checkpoint_sha256='a'*64)
        try:
            adapter.reset(42)
            metadata = adapter.result_metadata()
            assert metadata['checkpoint'] == 'synthetic-test-only'
            assert metadata['checkpoint_sha256'] == 'a'*64
            assert metadata['episode_seed'] == 42
            assert metadata['reset_confirmed'] is True
            assert metadata['capabilities']['episode_seed_confirmed'] is False
            assert metadata['capabilities']['remote_model_identity_verified'] is False
        finally:
            adapter.close()
        # Result construction happens during runner cleanup; preserve provenance.
        assert adapter.result_metadata()['episode_seed'] == 42
