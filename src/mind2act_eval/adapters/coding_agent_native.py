"""Schedule a provisioned native CodingAgent policy through the shared runner.

This module imports no legacy runtime, scene, scorer, planner or provider.
The integration owns the pinned SDK, public robot FK/planner, request ledger,
I/O deadlines and transcript. The adapter owns chunk timing and row validation.
"""
from collections import deque
from collections.abc import Mapping
from importlib import import_module
from math import isfinite

from mind2act_eval.adapters.coding_agent import public_observation
from mind2act_eval.contracts import Action, PolicyStop, freeze, to_json


class NativeCodingAgentAdapter:
    def __init__(self, integration):
        # observe receives only a projected public Observation. Public FK may
        # derive flange poses from those measured joints, never task evidence.
        for name in ("reset", "observe", "act", "confirm_executed", "close", "result_metadata"):
            if not callable(getattr(integration, name, None)):
                raise TypeError("native integration must implement " + name)
        self.integration = integration
        self._closed = False
        self._ready = False
        self._clock = None
        self._rows = deque()
        self._pending = None
        self._stop = None
        self._trace = []
        self._cleanup_error = None
        self._chunk_index = 0

    def reset(self, seed):
        if self._closed:
            raise RuntimeError("adapter is closed")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._ready = False
        self._clock = None
        self._rows.clear()
        self._pending = self._stop = None
        self._trace.clear()
        self._chunk_index = 0
        self.integration.reset(seed=seed)
        self._seed = seed
        self._ready = True

    def validate_clock(self, clock):
        if (clock.physics_hz, clock.control_hz, clock.rgb_hz) != (120, 30, 30):
            raise ValueError("native CodingAgent bridge requires explicit 120/30/30 clock")
        self._clock = clock

    def _load_chunk(self, observation):
        chunk = self.integration.act(self.integration.observe(public_observation(observation)))
        hz = getattr(chunk, "control_hz", None)
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not isfinite(hz) or hz != self._clock.control_hz:
            raise ValueError("native chunk control_hz does not match runner clock")
        rows = list(chunk.actions)
        if not rows:
            raise ValueError("native chunk must contain at least one action")
        validated = []
        for index, row in enumerate(rows):
            data = row.data
            if not isinstance(data, Mapping) or set(data) != {
                "left_arm_joint_state", "left_ee_joint_state",
                "right_arm_joint_state", "right_ee_joint_state",
            }:
                raise ValueError("native row must contain exactly four public joint channels")
            import numpy as np
            values = {name: np.asarray(value, dtype=float) for name, value in data.items()}
            if any(value.shape != ((6,) if "arm" in name else (1,)) for name, value in values.items()):
                raise ValueError("native row channel dimensions must be 6/1/6/1")
            action = Action(values["left_arm_joint_state"], values["right_arm_joint_state"],
                            values["left_ee_joint_state"][0], values["right_ee_joint_state"][0])
            meta = to_json(freeze(row.meta))
            if not isinstance(meta, dict):
                raise ValueError("native row metadata must be a JSON mapping")
            stop = None
            if "request_stop" in meta and type(meta["request_stop"]) is not bool:
                raise ValueError("request_stop must be boolean")
            if meta.get("request_stop"):
                if index != len(rows) - 1:
                    raise ValueError("native stop must be the final row of its chunk")
                stop = PolicyStop(meta.get("stop_reason"), meta.get("stop_detail", ""))
            validated.append((action, meta, stop))
        # Validate the entire chunk before returning any row to the robot.
        self._chunk_index += 1
        self._rows.extend(validated)

    def validate_limits(self, limits):
        validate = getattr(self.integration, "validate_limits", None)
        if callable(validate):
            validate(limits)

    def bind_decision_budget(self, budget):
        bind = getattr(self.integration, 'bind_decision_budget', None)
        if not callable(bind):
            raise ValueError('native integration must bind the framework decision budget')
        bind(budget)

    @property
    def requires_rgb(self):
        return not self._rows and self._stop is None

    def act(self, observation):
        if not self._ready or self._closed or self._clock is None:
            raise RuntimeError("reset and clock validation required before native act")
        if self._pending is not None:
            raise RuntimeError("previous native row has not been confirmed by the runner")
        if self._stop is not None:
            raise self._stop
        try:
            if not self._rows:
                self._load_chunk(observation)
            action, meta, stop = self._rows.popleft()
            self._pending = (action, meta, stop)
            self._trace.append({"chunk": self._chunk_index, "observation_tick": observation.tick,
                                "action": to_json(action), "native_metadata": meta,
                                "executed_physics_ticks": 0})
            return action
        except BaseException as error:
            self._ready = False
            try:
                self.close()
            except BaseException as cleanup:
                self._cleanup_error = type(cleanup).__name__ + ": " + str(cleanup)
            # The pinned native InfrastructureFailure derives from BaseException;
            # normalize it so the shared runner can publish its invalid Result.
            if not isinstance(error, (Exception, SystemExit, KeyboardInterrupt)):
                raise RuntimeError(type(error).__name__ + ": " + str(error)) from error
            raise

    def peek_next_action(self):
        # The shared runner interpolates only adjacent rows of this same chunk.
        return self._rows[0][0] if self._rows else None

    def confirm_action_executed(self, physics_ticks):
        if self._pending is None:
            raise RuntimeError("no native row awaits execution confirmation")
        if type(physics_ticks) is not int or not 0 < physics_ticks <= self._clock.ticks_per_action:
            raise ValueError("invalid native row execution duration")
        self._trace[-1]["executed_physics_ticks"] = physics_ticks
        self._stop = self._pending[2]
        self._pending = None
        if physics_ticks != self._clock.ticks_per_action:
            # A truncated row ends this episode. Never replan after truncation;
            # the pinned confirm_executed does not reconcile native goal latches.
            self._ready = False
        if not self._rows:
            self.integration.confirm_executed(sum(1 for row in self._trace if row["chunk"] == self._chunk_index))

    def result_metadata(self):
        return {"bridge_protocol": "mind2act-native-coding_agent-chunk-v1",
                "native_coding_agent_verified": False,
                "episode_seed": getattr(self, "_seed", None),
                "integration": to_json(freeze(self.integration.result_metadata())),
                "policy_stop": None if self._stop is None else {"reason": self._stop.reason, "detail": self._stop.detail},
                "returned_rows": len(self._trace), "unconsumed_rows": len(self._rows),
                "action_trace": self._trace, "cleanup_error": self._cleanup_error}

    def close(self):
        if not self._closed:
            self._closed = True
            self._ready = False
            self.integration.close()


def create(config):
    """Explicit trusted integration factory; no default model/credentials/budget."""
    if not isinstance(config, dict) or set(config) - {"integration_factory", "integration_config"}:
        raise ValueError("expected integration_factory and optional integration_config")
    factory = config.get("integration_factory")
    if not isinstance(factory, str) or len(factory.split(":")) != 2 or not all(factory.split(":")):
        raise ValueError("integration_factory must be module:factory")
    options = config.get("integration_config", {})
    if not isinstance(options, dict):
        raise ValueError("integration_config must be a JSON object")
    module, name = factory.split(":")
    integration = getattr(import_module(module), name)(to_json(freeze(options)))
    try:
        return NativeCodingAgentAdapter(integration)
    except BaseException:
        close = getattr(integration, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass
        raise
