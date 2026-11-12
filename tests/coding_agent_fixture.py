"""Deterministic public test plugin; errors are synthetic, no I/O transport."""
import json
from pathlib import Path


class Transport:
    def __init__(self, config):
        self.config = config

    def record(self, event, **fields):
        with Path(self.config["trace"]).open("a") as stream:
            stream.write(json.dumps(dict(event=event, **fields)) + "\n")

    def reset(self, *, seed):
        self.record("reset", seed=seed)
        if self.config.get("mode") == "reset_error":
            raise RuntimeError("original reset failure")

    def call_tool(self, name, *, observation):
        self.record("call_tool", name=name, tick=observation.tick,
                    fields=sorted(vars(observation)))
        mode = self.config.get("mode")
        if mode == "timeout":
            raise TimeoutError("synthetic transport deadline exceeded")
        if mode == "act_error":
            raise RuntimeError("synthetic tool failure")
        action = dict(left_arm=[1.] + [0.] * 5, right_arm=[0.] * 6,
                      left_gripper=0., right_gripper=1.)
        if mode == "width":
            action["left_arm"] = [0.] * 5
        elif mode == "nan":
            action["left_arm"][0] = float("nan")
        elif mode == "gripper":
            action["right_gripper"] = 2.
        elif mode == "extra":
            action["private"] = "must reject"
        elif mode == "none":
            return None
        self.record("action", width=sum(len(action[k]) for k in ("left_arm", "right_arm")) + 2)
        return action

    def close(self):
        self.record("close")
        if self.config.get("mode") in ("reset_error", "close_error"):
            raise RuntimeError("synthetic cleanup failure")


def create(config):
    return Transport(config)


def invalid(config):
    return {}
