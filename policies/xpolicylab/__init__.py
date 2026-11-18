"""Public observation/action boundary for XPolicyLab and native G0.5.

No simulator, task registry, evidence, scorer, or robot driver is imported here.
Native G0.5 owns its chunk cache; canonical XPolicyLab RPC returns whole chunks.
Select the transport explicitly: these wire formats are not interchangeable.
"""
from __future__ import annotations

from typing import Mapping
import re

import numpy as np

from mind2act_eval.contracts import Action, ClockSpec, Observation, to_json
from mind2act_eval.model_text import require_english_model_text

CAMERAS = {'head': 'cam_high', 'left_wrist': 'cam_left_wrist', 'right_wrist': 'cam_right_wrist'}
STATE_PARTS = ('left_arm', 'left_gripper', 'right_arm', 'right_gripper')
POLICY_CONTROL_HZ = 30
POLICY_RGB_HZ = 30


def public_observation(obs: Observation, camera_map: Mapping[str, str]) -> dict:
    """Rebuild an allowlist, converting public RGB HWC to model RGB CHW."""
    if obs.protocol_version != 'v1':
        raise ValueError('unsupported observation protocol')
    if not isinstance(obs.instruction, str) or not obs.instruction.strip():
        raise ValueError('nonempty public instruction required')
    require_english_model_text(obs.instruction)
    images = {}
    for source, target in camera_map.items():
        pixels = np.asarray(obs.rgb[source])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
            raise ValueError('expected uint8 RGB HWC: ' + source)
        images[target] = np.ascontiguousarray(pixels.transpose(2, 0, 1))
    return {'images': images,
            'state': {'left_arm': np.asarray(obs.left_arm, dtype=np.float32),
                      'left_gripper': np.asarray([obs.left_gripper], dtype=np.float32),
                      'right_arm': np.asarray(obs.right_arm, dtype=np.float32),
                      'right_gripper': np.asarray([obs.right_gripper], dtype=np.float32)},
            'task': obs.instruction, 'frequency': POLICY_CONTROL_HZ, 'embodiment_type': 'robodojo'}


def action_from_parts(parts: dict) -> Action:
    if not isinstance(parts, dict) or set(parts) != set(STATE_PARTS):
        raise ValueError('complete native joint action required; missing parts cannot be held silently')
    arrays = [np.asarray(parts[key], dtype=np.float32) for key in STATE_PARTS]
    if any(a.shape != (n,) or not np.isfinite(a).all() for a, n in zip(arrays, (6, 1, 6, 1))):
        raise ValueError('actions must be finite native absolute joint 6/1/6/1')
    # Only normalized grippers can overshoot. Arms stay absolute radians; no
    # delta transform, row shift, finger-coordinate conversion, or robot tuning.
    left, lg, right, rg = arrays
    return Action(tuple(left), tuple(right), float(np.clip(lg[0], 0, 1)), float(np.clip(rg[0], 0, 1)))


class XPolicyLabAdapter:
    """Adapter contract: seeded local episode lifecycle and one action per act.

    Native G0.5 reset only clears the remote chunk; it does NOT seed model RNG.
    This capability difference is exposed in transport.capabilities.
    """
    def __init__(self, transport, *, camera_map=None, result_identity=None):
        self.transport = transport
        self.camera_map = dict(CAMERAS if camera_map is None else camera_map)
        if len(self.camera_map) != 3 or set(self.camera_map.values()) != set(CAMERAS.values()):
            raise ValueError('camera map must bind exactly the three native cameras')
        self._result_identity = dict(result_identity or {})
        self._requested_clock = None
        self._clock_validated = False
        self._ticks_per_action = None
        self._episode_seed = None
        self._reset_confirmed = False
        self._active = False
        self._closed = False
        self._pending_action = None

    @classmethod
    def g05_native(cls, url: str, *, action_steps: int, timeout_s=180., camera_map=None,
                   checkpoint: str | None = None, checkpoint_sha256: str | None = None):
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ValueError('checkpoint must be a nonempty configured identifier')
        if checkpoint_sha256 is not None and (not isinstance(checkpoint_sha256, str) or
                not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256)):
            raise ValueError('checkpoint_sha256 must be a full SHA256')
        from .g05_wire import G05NativeTransport
        return cls(G05NativeTransport(url, action_steps=action_steps, timeout_s=timeout_s),
                   camera_map=camera_map, result_identity={'transport': 'g05-native',
                       'checkpoint': checkpoint, 'checkpoint_sha256': checkpoint_sha256,
                       'action_steps': action_steps})

    @classmethod
    def xpolicy_rpc(cls, url: str, *, expected_identity: dict, source_root=None,
                    timeout_s=180., camera_map=None):
        from .rpc import XPolicyRPCTransport
        return cls(XPolicyRPCTransport(url, expected_identity=expected_identity,
                   source_root=source_root, timeout_s=timeout_s), camera_map=camera_map,
                   result_identity={'transport': 'xpolicy-rpc', **expected_identity})

    def validate_clock(self, clock: ClockSpec) -> None:
        """Runner capability hook: reject retiming instead of converting rows."""
        self._clock_validated = False
        self._ticks_per_action = None
        self._requested_clock = to_json(clock) if isinstance(clock, ClockSpec) else None
        was_active, self._active = self._active, False
        if not isinstance(clock, ClockSpec):
            raise ValueError('policy clock must be a ClockSpec')
        # Revalidate the declared rates at this I/O boundary: positive integer
        # physics, divisible control/RGB clocks. Physics substeps are not wire data.
        validated = ClockSpec(**self._requested_clock)
        if validated.control_hz != POLICY_CONTROL_HZ or validated.rgb_hz != POLICY_RGB_HZ:
            raise ValueError('policy clock mismatch: requires control/RGB 30/30 Hz')
        self._ticks_per_action = validated.ticks_per_action
        self._clock_validated = True
        self._active = was_active

    def reset(self, seed: int) -> None:
        self._active = False
        self._pending_action = None
        if self._closed:
            raise RuntimeError('adapter is closed')
        if type(seed) is not int or seed < 0:
            raise ValueError('episode seed must be a nonnegative integer')
        self._episode_seed = seed
        self._reset_confirmed = False
        # The seed is case identity only; no transport sends it to a model service.
        self.transport.reset(seed)
        self._reset_confirmed = True
        self._active = True

    @property
    def requires_rgb(self):
        """A prefetched row can finish a chunk without consuming its next observation."""
        return self._pending_action is None and self.transport.needs_observation

    def act(self, observation: Observation) -> Action:
        if not self._active:
            raise RuntimeError('successful reset required before act')
        try:
            if self._pending_action is not None:
                action, self._pending_action = self._pending_action, None
                return action
            payload = public_observation(observation, self.camera_map) if self.transport.needs_observation else {}
            return action_from_parts(self.transport.action(payload))
        except Exception:
            self._active = False
            raise

    def peek_next_action(self):
        """Prefetch only inside the current chunk, never across a fresh-observation boundary."""
        if not self._active:
            raise RuntimeError('successful reset required before peek')
        if self._pending_action is None and not self.transport.needs_observation:
            try:
                self._pending_action = action_from_parts(self.transport.action({}))
            except Exception:
                self._active = False
                raise
        return self._pending_action

    def result_metadata(self) -> dict:
        """Runner copies this JSON-compatible provenance into Result on exit.

        Checkpoint values are caller declarations on native G0.5. Neither the
        local seed nor the cache reset acknowledgement confirms model RNG state.
        Preserve them across close(), including failed attempts, for archiving.
        """
        return {**self._result_identity,
                **({'service_environment_ticks_per_row_hint': self.transport.service_environment_ticks_per_row_hint}
                   if hasattr(self.transport, 'service_environment_ticks_per_row_hint') else {}),
                'episode_seed': self._episode_seed,
                'reset_confirmed': self._reset_confirmed,
                'clock': {'required': {'control_hz': POLICY_CONTROL_HZ, 'rgb_hz': POLICY_RGB_HZ},
                          'requested': dict(self._requested_clock) if self._requested_clock is not None else None,
                          'validated': self._clock_validated,
                          'ticks_per_action': self._ticks_per_action,
                          'action_period_s': 1 / POLICY_CONTROL_HZ if self._clock_validated else None},
                'capabilities': dict(self.transport.capabilities)}

    def close(self) -> None:
        self._active = False
        self._pending_action = None
        if not self._closed:
            self._closed = True
            self.transport.close()


def create(config: dict) -> XPolicyLabAdapter:
    """CLI adapter factory; native msgpack is the default transport."""
    if not isinstance(config, dict):
        raise ValueError('adapter configuration must be a mapping')
    values = dict(config)
    transport = values.pop('transport', 'g05-native')
    if transport == 'g05-native':
        return XPolicyLabAdapter.g05_native(**values)
    if transport == 'xpolicy-rpc':
        return XPolicyLabAdapter.xpolicy_rpc(**values)
    raise ValueError('unknown policy transport: ' + str(transport))
