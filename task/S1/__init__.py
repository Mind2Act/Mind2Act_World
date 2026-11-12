"""S1 task definition. Isaac-dependent code is imported lazily by `build`."""
from mind2act_eval.tasks import TaskDefinition


def _build(spec, ctx):
    from .scene import build
    return build(spec, ctx)


def _score(evidence):
    from .scorer import score
    return score(evidence)


DEFINITION = TaskDefinition("S1", _build, _score)
