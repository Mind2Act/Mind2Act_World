"""Deterministic public action source used by the CPU demo only."""
from mind2act_eval.contracts import Action


class DemoAdapter:
    def reset(self, seed):
        pass

    def act(self, observation):
        return Action((1., 0., 0., 0., 0., 0.), (0.,) * 6, 1., 1.)

    def close(self):
        pass


def create(config):
    return DemoAdapter()
