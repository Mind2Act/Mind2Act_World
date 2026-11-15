"""Canonical pinned XPolicyLab RPC: one inference yields a full chunk."""
from __future__ import annotations

from collections import deque
import math
import re

import numpy as np

from .xpolicylab import STATE_PARTS
from .xpolicy_pin import PROTOCOL_VERSION, create_client

IDENTITY_KEYS = {'model', 'protocol_version', 'source_commit', 'checkpoint_sha256',
                 'stats_sha256', 'config_sha256', 'action_horizon'}
ACTION_CONTRACT = dict(action_space='joint14', arm_units='rad', gripper_range=[0, 1],
    control_hz=30, raw_state_dim=14, raw_action_dim=14,
    output_row_shift=0, cache_owner='environment_client')


class XPolicyRPCTransport:
    @property
    def capabilities(self):
        return {'cache_owner': 'environment_client',
                'episode_seed_confirmed': False,
                'episode_seed_identity_only': True,
                'remote_model_identity_verified': self._identity_confirmed,
                'model_timing_available': True}

    def __init__(self, url, *, expected_identity, source_root=None, timeout_s=180.):
        if not isinstance(expected_identity, dict) or set(expected_identity) != IDENTITY_KEYS:
            raise ValueError('explicit model/protocol/source/checkpoint/stats/config/horizon identity required')
        identity = dict(expected_identity)
        for key in ('checkpoint_sha256', 'stats_sha256', 'config_sha256'):
            if not isinstance(identity[key], str) or not re.fullmatch('[0-9a-f]{64}', identity[key]):
                raise ValueError('expected SHA256 required: ' + key)
        if not isinstance(identity['source_commit'], str) or not re.fullmatch('[0-9a-f]{40}', identity['source_commit']):
            raise ValueError('full source commit required')
        if identity['protocol_version'] != PROTOCOL_VERSION:
            raise ValueError('canonical protocol pin required')
        if not isinstance(identity['model'], str) or not identity['model'].strip():
            raise ValueError('nonempty model identity required')
        if type(identity['action_horizon']) is not int or identity['action_horizon'] < 1:
            raise ValueError('positive pinned action horizon required')
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite timeout required')
        self.identity = identity
        self.queue = deque()
        self._reset_done = False
        self._identity_confirmed = False
        self.service_environment_ticks_per_row_hint = None
        self.model_wall_s = 0.
        self.rpc = create_client(url, source=source_root, evaluation_id='mind2act-eval',
            trial_id='episode', connect_timeout_s=timeout_s, handshake_timeout_s=timeout_s,
            request_timeout_s=timeout_s, max_connect_attempts=1, close_timeout_s=5.)

    @property
    def needs_observation(self):
        return not self.queue

    def reset(self, seed):
        # The episode seed is local case identity only (as in RoboDojo, which sends
        # no per-episode seed to the policy). The model's own randomness is the
        # service's business; record its checkpoint/config identity instead.
        self.queue.clear()
        self._reset_done = False
        self._identity_confirmed = False
        self.service_environment_ticks_per_row_hint = None
        self.model_wall_s = 0.
        metadata = self.rpc.call('get_metadata')
        if not isinstance(metadata, dict) or metadata.get('ready') is not True:
            raise ValueError('policy service is not ready')
        for key, expected in dict(self.identity, **ACTION_CONTRACT).items():
            if metadata.get(key) != expected:
                raise ValueError('policy identity/action contract mismatch: ' + key)
        # Legacy Pi0.5 advertises its old environment's four substeps here.
        # Official RPC forwards this metadata without interpreting it; model
        # inference consumes frequency=30, not a simulation physics clock.
        self.service_environment_ticks_per_row_hint = metadata.get('environment_ticks_per_row')
        self._identity_confirmed = True
        self.rpc.call('reset')
        self._reset_done = True

    def action(self, observation):
        if not self._reset_done:
            raise RuntimeError('reset required')
        if not self.queue:
            self.rpc.call('update_obs', observation)
            reply = self.rpc.call('get_action')
            if not isinstance(reply, dict):
                raise ValueError('action reply must be a mapping')
            if reply.get('need_obs') is not True:
                raise ValueError('chunk must request a fresh observation after consumption')
            raw = np.asarray(reply['actions'], dtype=np.float32)
            if raw.shape != (self.identity['action_horizon'], 14) or not np.isfinite(raw).all():
                raise ValueError('action chunk must match finite pinned horizon x14')
            duration = reply.get('model_wall_s')
            if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
                raise ValueError('invalid model_wall_s')
            self.model_wall_s += duration
            self.queue.extend(raw.copy())
        row = self.queue.popleft()
        return dict(zip(STATE_PARTS, (row[:6], row[6:7], row[7:13], row[13:14])))

    def close(self):
        self.queue.clear()
        self._reset_done = False
        self.rpc.close()
