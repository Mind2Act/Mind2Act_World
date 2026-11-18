"""CPU-only deterministic runtime, explicitly selected; never an Isaac fallback."""
class DemoSimulator:
    def __init__(self):
        self.world = self
        self.ticks = 0

    def step(self, *, render):
        self.ticks += 1

    def close(self):
        pass


class DemoRobot:
    def __init__(self):
        from mind2act_eval.contracts import Action
        self.action = Action((0.,) * 6, (0.,) * 6, 1., 1.)

    def apply(self, action):
        self.action = action

    def observe(self):
        from mind2act_eval.contracts import to_json
        return {**to_json(self.action), "rgb": {}}
