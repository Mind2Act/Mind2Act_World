"""Application bridge, not the upstream CodingAgent model/tool wire protocol.

Use ``create`` for CLI JSON configuration. An external integration must adapt
the native policy/tool/planner loop to this single-Action interface itself.
Transport factories are trusted code, not a filesystem sandbox. Provision
external tools without private task inputs and with their own I/O timeouts.
"""
from collections.abc import Mapping
from importlib import import_module

from mind2act_eval.contracts import Action, Observation


def public_observation(observation):
    """Copy only the frozen public fields before invoking trusted plugin code."""
    if not isinstance(observation, Observation):
        raise TypeError("expected public Observation")
    if not isinstance(observation.instruction, str) or not isinstance(observation.protocol_version, str):
        raise TypeError("public instruction and protocol_version must be strings")
    images = {}
    if observation.rgb:
        import numpy as np
        for name, image in observation.rgb.items():
            if not isinstance(name, str) or not name:
                raise ValueError("camera names must be nonempty strings")
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("public RGB requires uint8 HWC arrays with three channels")
            images[str(name)] = np.array(image, copy=True, subok=False)
    # Construct a base object explicitly: never forward subclasses,
    # __dict__, evidence, assets, task config or scorer references.
    return Observation(
        observation.tick, float(observation.sim_time_s),
        tuple(observation.left_arm), tuple(observation.right_arm),
        observation.left_gripper, observation.right_gripper,
        rgb=images,
        instruction=str(observation.instruction),
        protocol_version=str(observation.protocol_version),
    )


class CodingAgentAdapter:
    requires_rgb = True  # Each act call is a new public tool/model decision.

    def __init__(self, transport):
        if not all(callable(getattr(transport, name, None)) for name in ("reset", "call_tool", "close")):
            raise TypeError("transport must implement reset/call_tool/close; CLI JSON uses coding_agent:create")
        self.transport = transport
        self._ready = False
        self._closed = False
        self._seed = None
        self._cleanup_error = None

    def reset(self, seed: int):
        self._ready = False
        if self._closed:
            raise RuntimeError("adapter is closed")
        try:
            if type(seed) is not int:
                raise ValueError("seed must be an integer")
            self._seed = seed
            self.transport.reset(seed=seed)
            self._ready = True
        except BaseException:
            self._close_after_failure()
            raise

    def act(self, observation: Observation) -> Action:
        if not self._ready or self._closed:
            raise RuntimeError("successful reset required before act")
        try:
            public = public_observation(observation)
            response = self.transport.call_tool("act", observation=public)
            if isinstance(response, Action):
                # Revalidate even preconstructed/subclassed responses.
                response = {name: getattr(response, name) for name in (
                    "left_arm", "right_arm", "left_gripper", "right_gripper")}
            if not isinstance(response, Mapping) or set(response) != {
                "left_arm", "right_arm", "left_gripper", "right_gripper"
            }:
                raise ValueError("transport must return exactly one 14D Action mapping")
            return Action(**response)
        except BaseException:
            self._ready = False
            self._close_after_failure()
            raise

    def _close_after_failure(self):
        # The runner calls close again during cleanup. Make that a no-op and
        # preserve the primary reset/act error even if transport.close fails.
        try:
            self.close()
        except BaseException as exc:
            self._cleanup_error = f"{type(exc).__name__}: {exc}"

    def close(self):
        if not self._closed:
            self._closed = True
            self._ready = False
            self.transport.close()

    def result_metadata(self):
        return {"bridge_protocol": "mind2act-public-action-v1",
                "native_coding_agent_verified": False,
                "episode_seed": self._seed,
                "seed_sent": self._seed is not None,
                "seed_confirmed": False,
                "cleanup_error": self._cleanup_error}


def create(config):
    """CLI factory. No default transport, model, network or private inputs."""
    if not isinstance(config, dict) or set(config) - {
        "transport_factory", "transport_config"
    }:
        raise ValueError("expected transport_factory and optional transport_config")
    factory = config.get("transport_factory")
    if not isinstance(factory, str) or len(factory.split(":")) != 2 or not all(factory.split(":")):
        raise ValueError("transport_factory must be module:factory; native CodingAgent is not configured")
    options = config.get("transport_config", {})
    if not isinstance(options, dict):
        raise ValueError("transport_config must be a JSON object")
    from mind2act_eval.contracts import freeze, to_json
    options = to_json(freeze(options))  # finite JSON only; no Python objects
    module, name = factory.split(":")
    transport = getattr(import_module(module), name)(options)
    try:
        return CodingAgentAdapter(transport)
    except BaseException:
        close = getattr(transport, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass
        raise
