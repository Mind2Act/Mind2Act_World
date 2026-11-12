"""G0.5 native binary WebSocket profile, independent of model/CUDA packages.

The ndarray envelope follows G0.5's g05.utils.websocket.msgpack. It is not
msgpack-numpy and is not the canonical XPolicyLab method-RPC envelope.
"""
from __future__ import annotations

import math

import msgpack
import numpy as np
from websockets.sync.client import connect


def _encode(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ('V', 'O', 'c'):
        raise ValueError('unsupported ndarray dtype')
    if isinstance(value, np.ndarray):
        return {'__ndarray__': True, 'data': value.tobytes(), 'dtype': value.dtype.str, 'shape': value.shape}
    if isinstance(value, np.generic):
        return {'__npgeneric__': True, 'data': value.item(), 'dtype': value.dtype.str}
    raise TypeError('unsupported msgpack value')


def _decode(value):
    # Some G0.5 peers write byte keys; normalize just as the native codec does.
    value = {(k.decode() if isinstance(k, bytes) else k): v for k, v in value.items()}
    if '__ndarray__' in value or '__npgeneric__' in value:
        dtype = np.dtype(value['dtype'])
        if dtype.kind in ('V', 'O', 'c'):
            raise ValueError('unsupported ndarray dtype')
        if '__ndarray__' in value:
            return np.frombuffer(value['data'], dtype=dtype).reshape(tuple(value['shape'])).copy()
        return dtype.type(value['data'])
    return value


class G05NativeTransport:
    capabilities = {'cache_owner': 'model_service', 'episode_seed_confirmed': False,
                    'episode_seed_identity_only': True,
                    'remote_model_identity_verified': False, 'model_timing_available': False}

    def __init__(self, url: str, *, action_steps: int, timeout_s: float = 180.):
        if type(action_steps) is not int or not 1 <= action_steps <= 32:
            raise ValueError('explicit execution chunk must be between 1 and trained horizon 32')
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite timeout required')
        self.timeout_s = timeout_s
        self.action_steps = action_steps
        self.needs_observation = True
        self._row = 0
        self._broken = False
        self.url = url
        self.ws = None

    def _connect(self):
        # Adapter construction precedes scene build. Native build/reset can hold
        # the GIL long enough to starve the websocket receiver's ping replies.
        # Open only on the first RPC, after runner scene/robot/sim initialization.
        self.ws = connect(self.url, open_timeout=self.timeout_s, close_timeout=5,
                          max_size=64*1024*1024, compression=None)
        handshake = self._receive()
        if type(handshake.get('action_steps')) is not int or handshake['action_steps'] != self.action_steps:
            raise ValueError('G0.5 action_steps handshake differs from explicit execution chunk')

    def _receive(self):
        blob = self.ws.recv(timeout=self.timeout_s)
        if not isinstance(blob, bytes):
            raise ValueError('G0.5 requires binary msgpack replies')
        value = msgpack.unpackb(blob, object_hook=_decode, strict_map_key=False)
        if not isinstance(value, dict):
            raise ValueError('G0.5 reply must be a mapping')
        if 'error' in value:
            raise RuntimeError('G0.5 service returned an inference error')
        return value

    def _request(self, value):
        if self._broken:
            raise RuntimeError('transport failed; create a new connection')
        try:
            if self.ws is None:
                self._connect()
            self.ws.send(msgpack.packb(value, default=_encode))
            return self._receive()
        except Exception:
            # After a timeout a late reply must never be mistaken for reset/action.
            self.close()
            raise

    def reset(self, seed: int):
        reply = self._request({'__reset__': True})
        if reply != {'__reset__': True}:
            self.close()
            raise ValueError('G0.5 did not acknowledge cache reset')
        self._row = 0
        self.needs_observation = True

    def action(self, observation):
        reply = self._request(observation)
        expected = (self._row + 1) % self.action_steps == 0
        if type(reply.get('need_obs')) is not bool or reply['need_obs'] != expected:
            raise ValueError('G0.5 need_obs differs from negotiated chunk schedule')
        self._row += 1
        self.needs_observation = reply['need_obs']
        return reply['action']

    def close(self):
        self._broken = True
        if self.ws is not None:
            self.ws.close()
