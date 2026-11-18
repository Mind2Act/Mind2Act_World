"""Pinned upstream EEF tool loop, with explicitly provisioned public capabilities.

Client and planner factories are trusted integration code. Neither receives
scene/scorer objects, manifests, truth, phases, trajectories or robot handles.
The client owner must enforce the existing durable upstream-attempt ledger.
"""
from importlib import import_module

import numpy as np

from mind2act_eval.adapters.roboprobe.kinematics import X5Kinematics, URDF_SHA256
from mind2act_eval.adapters.roboprobe.native import NativeRoboProbeAdapter
from mind2act_eval.adapters.roboprobe.pin import COMMIT, load
from mind2act_eval.adapters.agent_payload import validate_input
from mind2act_eval.contracts import freeze, to_json

_ENV_KEYS = {"L3_INSPECT_BASE_URL", "L3_INSPECT_PLANNER", "L3_INSPECT_MODEL",
             "L3_INSPECT_API_STYLE", "L3_INSPECT_API_VERSION", "L3_INSPECT_API_KEY_ENV",
             "L3_INSPECT_MAX_LLM_CALLS", "L3_INSPECT_MAX_RETRIES", "L3_INSPECT_REASONING_EFFORT",
             "L3_INSPECT_IMAGE_HORIZON", "L3_INSPECT_KEEP_ALL_IMAGES", "L3_INSPECT_USE_RECIPE",
             "L3_INSPECT_DEPTH", "L3_INSPECT_ALLOW_DONE", "L3_INSPECT_HARD_TIMEOUT_S",
             "L3_INSPECT_TIMEOUT_S"}

# Pinned SDK joint deploy.py's measured tracking bounds at 30 Hz.
DEFAULT_MAX_STEP = ([0.05] * 6 + [0.25]) * 2


def policy_defaults(env):
    env = dict(env)
    if env.get("L3_INSPECT_DEPTH") in (None, "0"):
        env["L3_INSPECT_DEPTH"] = "off"
    env.setdefault("L3_INSPECT_TIMEOUT_S", "300")
    return env


def action_defaults(action):
    action = dict(action)
    steps = action.get("max_step", [None] * 14)
    if not isinstance(steps, (tuple, list)) or len(steps) != 14:
        raise ValueError("native max_step requires fourteen joint/gripper entries")
    action["max_step"] = [default if value is None else value for value, default in zip(steps, DEFAULT_MAX_STEP)]
    return action


def _factory(name, options):
    if not isinstance(name, str) or len(name.split(":")) != 2 or not all(name.split(":")):
        raise ValueError("capability factory must be module:factory")
    if not isinstance(options, dict):
        raise ValueError("capability config must be a JSON object")
    module, factory = name.split(":")
    return getattr(import_module(module), factory)(to_json(freeze(options)))


class EefIntegration:
    def __init__(self, config, client, planner, *, policy_factory=None):
        self.client, self.planner = client, planner
        joint, self.types, eef = load(config["source_root"])
        self.kinematics = X5Kinematics(config["urdf"], config["mounts"])
        self.config = to_json(freeze(config))
        env = config["policy_env"]
        required = {"L3_INSPECT_BASE_URL", "L3_INSPECT_MODEL", "L3_INSPECT_MAX_LLM_CALLS", "L3_INSPECT_MAX_RETRIES",
                    "L3_INSPECT_REASONING_EFFORT", "L3_INSPECT_API_STYLE"}
        if (not isinstance(env, dict) or set(env) - _ENV_KEYS or not required <= set(env)
                or any(not isinstance(value, str) for value in env.values())):
            raise ValueError("explicit allowlisted native policy_env required; no credentials or history paths")
        self.physics_ticks = config["physics_ticks"]
        if type(self.physics_ticks) is not int or self.physics_ticks <= 0:
            raise ValueError("explicit positive physics budget required")
        if not 1 <= int(env["L3_INSPECT_MAX_LLM_CALLS"]) <= 150:
            raise ValueError("native LLM decision call limit must be between 1 and 150")
        if int(env["L3_INSPECT_MAX_RETRIES"]) < 0:
            raise ValueError("invalid declared native request budget")
        self.camera_map = config["camera_map"]
        if (not isinstance(self.camera_map, dict) or len(self.camera_map) != 3
                or set(self.camera_map.values()) != {"cam_head", "cam_left_wrist", "cam_right_wrist"}):
            raise ValueError("exact three public camera aliases required")
        action = config["action_spec"]
        labels = tuple(label for arm in ("left", "right") for label in
                       (*[f"{arm}_joint{i}" for i in range(1, 7)], f"{arm}_gripper"))
        spec = joint.RoboDojoActionSpec(labels, np.asarray(action["low"], dtype=float),
            np.asarray(action["high"], dtype=float), 30., action["docs"], tuple(action["max_step"]))
        self._state = None
        native_client = getattr(client, "native_client", client)
        for field, key in (("model", "L3_INSPECT_MODEL"), ("reasoning_effort", "L3_INSPECT_REASONING_EFFORT"),
                           ("api_style", "L3_INSPECT_API_STYLE"), ("azure_endpoint", "L3_INSPECT_BASE_URL")):
            value = getattr(native_client, field, None)
            if value is not None and value != env[key]:
                raise ValueError("native client differs from the declared policy profile: " + field)
        self.policy = (policy_factory or eef.EefAgentPolicy)(action_spec=spec, env=env,
            client=native_client, planner=self._plan)
        self._guard_client()

    def _guard_client(self):
        native_client = self.policy._client
        complete = native_client.complete
        def guarded_complete(messages, tools):
            validate_input(messages, tools)
            return complete(messages, tools)
        native_client.complete = guarded_complete

    def _plan(self, *, arm, target_pose):
        if self._state is None:
            raise RuntimeError("planner requires the current public measured state")
        # A real provisioned planner receives only public joints and flange goal.
        return self.planner(arm=arm, q=np.array(self._state[arm + "_arm_joint_state"], copy=True),
                            target_pose=np.array(target_pose, copy=True))

    def reset(self, *, seed):
        self.seed, self._state = seed, None
        self.policy.reset()

    def validate_limits(self, limits):
        if limits.physics_ticks != self.physics_ticks:
            raise ValueError("native observation budget differs from shared runner physics budget")

    def bind_decision_budget(self, budget):
        bind = getattr(self.client, 'bind_decision_budget', None)
        if not callable(bind):
            raise ValueError('agent client must enforce framework call budgets')
        bind(budget)
        self._guard_client()

    def observe(self, public):
        if public.tick % 4:
            raise ValueError("native observations require a complete 30Hz control boundary")
        if not set(self.camera_map) <= set(public.rgb):
            raise ValueError("all three native public RGB cameras are required")
        state = {}
        for arm in ("left", "right"):
            radians = np.array(getattr(public, arm + "_arm"), dtype=float)
            state[arm + "_arm_joint_state"] = radians
            state[arm + "_ee_joint_state"] = np.array([getattr(public, arm + "_gripper")])
            state[arm + "_ee_pose"] = self.kinematics.flange_pose(arm, radians)
        self._state = state
        return self.types.Observation(images={native: public.rgb[name] for name, native in self.camera_map.items()},
            state=state, instruction=public.instruction, step=public.tick // 4,
            remaining_steps=max(0, (self.physics_ticks - public.tick + 3) // 4), extra={})

    def act(self, observation):
        return self.policy.act(observation)

    def confirm_executed(self, played):
        self.policy.confirm_executed(played)

    def result_metadata(self):
        metadata = getattr(self.client, "result_metadata", None)
        return {"native_source_commit": COMMIT, "public_urdf_sha256": URDF_SHA256,
                "model": self.config["policy_env"]["L3_INSPECT_MODEL"],
                "action_spec_max_step": self.config["action_spec"]["max_step"],
                "policy_env": self.config["policy_env"], "episode_seed": getattr(self, "seed", None),
                "seed_confirmed": False, "native_calls": self.policy.calls,
                "native_transcript": self.policy.transcript(),
                "client": metadata() if callable(metadata) else {"provenance_verified": False}}

    def close(self):
        errors = []
        for resource in (self.client, self.planner):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    errors.append(type(error).__name__ + ": " + str(error))
        if errors:
            raise RuntimeError("; ".join(errors))


def create(config):
    if isinstance(config, dict) and 'source_root' not in config:
        from mind2act_eval.adapters.roboprobe.pin import resolve_source_root
        config = dict(config, source_root=str(resolve_source_root()))
    required = {"source_root", "urdf", "mounts", "physics_ticks", "camera_map", "action_spec", "policy_env",
                "client_factory", "client_config", "planner_factory", "planner_config"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("native EEF config requires exactly: " + ", ".join(sorted(required)))
    config = to_json(freeze(config))
    config["policy_env"] = policy_defaults(config["policy_env"])
    config["action_spec"] = action_defaults(config["action_spec"])
    if config["client_factory"] == "mind2act_eval.adapters.roboprobe_client:create":
        config["client_config"]["policy_env"] = policy_defaults(config["client_config"]["policy_env"])
    client = planner = None
    try:
        client = _factory(config["client_factory"], config["client_config"])
        planner = _factory(config["planner_factory"], config["planner_config"])
        return NativeRoboProbeAdapter(EefIntegration(config, client, planner))
    except BaseException:
        for resource in (client, planner):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass
        raise
