"""Public synthetic counter scene for CPU integration."""
from mind2act_eval.contracts import Observation, NativeEvidence, Health


class DemoScene:
    def __init__(self, spec, ctx):
        self.spec, self.ctx = spec, ctx

    def observe(self, tick):
        return Observation(tick, tick / self.spec.clock.physics_hz,
                           **self.ctx.robot.observe(), instruction="Reach the public target",
                           protocol_version=self.spec.protocol_version)

    def before_step(self, tick):
        pass

    def after_step(self, tick):
        pass

    def evidence(self, tick):
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
                              {"reached": self.is_complete()})

    def health(self):
        return Health(True)

    def is_complete(self):
        return self.ctx.robot.observe()["left_arm"][0] >= 1.0

    def close(self):
        pass


def build(spec, ctx):
    return DemoScene(spec, ctx)
