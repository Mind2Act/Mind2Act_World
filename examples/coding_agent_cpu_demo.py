"""Public zero-model CLI demonstration. This is not a CodingAgent policy."""


class DemoTransport:
    def __init__(self):
        self.ready = False

    def reset(self, *, seed):
        self.ready = True

    def call_tool(self, name, *, observation):
        if name != "act" or not self.ready:
            raise RuntimeError("demo requires reset then act")
        return {"left_arm": [1., 0., 0., 0., 0., 0.], "right_arm": [0.] * 6,
                "left_gripper": 1., "right_gripper": 1.}

    def close(self):
        self.ready = False


def create(config):
    if config:
        raise ValueError("public demo accepts no configuration")
    return DemoTransport()
