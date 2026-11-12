"""Protocol E2E: a deterministic model boundary, real binary WebSocket I/O."""
from contextlib import contextmanager
from threading import Thread

import msgpack
import numpy as np
import pytest
from websockets.sync.server import serve

from mind2act_eval.contracts import Action, Observation
from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter


def pack(value):
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return dict(__ndarray__=True, data=obj.tobytes(), dtype=obj.dtype.str, shape=obj.shape)
        raise TypeError(type(obj))
    return msgpack.packb(value, default=encode)


def unpack(blob):
    def decode(obj):
        if '__ndarray__' in obj:
            return np.frombuffer(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        return obj
    return msgpack.unpackb(blob, object_hook=decode)


def observation():
    # Distinct RGB channels detect HWC/CHW and RGB/BGR mistakes.
    image = np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)
    return Observation(0, 0., (0.,)*6, (1.,)*6, .25, .75,
                       {name: image for name in ('head', 'left_wrist', 'right_wrist')},
                       'Move the cube')


@contextmanager
def model_service(handler):
    errors = []
    def checked(ws):
        try:
            handler(ws)
        except Exception as exc:
            errors.append(exc)
    with serve(checked, '127.0.0.1', 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'ws://127.0.0.1:{server.socket.getsockname()[1]}'
        finally:
            server.shutdown()
            thread.join(timeout=5)
    if errors:
        raise errors[0]


def test_g05_native_observation_actions_and_episode_cache_reset():
    def policy(ws):
        ws.send(pack({'action_steps': 2}))
        row = 0
        for blob in ws:
            payload = unpack(blob)
            if payload == {'__reset__': True}:
                row = 0
                ws.send(pack({'__reset__': True}))
                continue
            if row % 2 == 0:
                assert set(payload) == {'images', 'state', 'task', 'frequency', 'embodiment_type'}
                assert payload['frequency'] == 30
                assert payload['embodiment_type'] == 'robodojo'
                assert payload['task'] == 'Move the cube'
                np.testing.assert_array_equal(payload['images']['cam_high'], [[[11,44]], [[22,55]], [[33,66]]])
                np.testing.assert_array_equal(payload['state']['left_gripper'], [.25])
                np.testing.assert_array_equal(payload['state']['right_arm'], [1]*6)
            else:
                assert payload == {}
            ws.send(pack({'action': {'left_arm': np.full(6, row + .5, np.float32),
                                     'right_arm': np.full(6, -row - .5, np.float32),
                                     'left_gripper': np.array([1.1], np.float32),
                                     'right_gripper': np.array([-.1], np.float32)},
                          'need_obs': row % 2 == 1}))
            row += 1
    with model_service(policy) as url:
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=2)
        try:
            adapter.reset(7)
            assert adapter.act(observation()) == Action((.5,)*6, (-.5,)*6, 1., 0.)
            assert adapter.act(observation()) == Action((1.5,)*6, (-1.5,)*6, 1., 0.)
            # Reset while remote chunk still has rows; no stale action crosses episodes.
            adapter.act(observation())
            adapter.reset(8)
            assert adapter.act(observation()).left_arm == (.5,)*6
        finally:
            adapter.close()
