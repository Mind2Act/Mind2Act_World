"""PR4: original free FEM cloth; producer import does not load native scene."""


def __getattr__(name):
    if name != 'DEFINITION':
        raise AttributeError(name)
    from mind2act_eval.tasks import TaskDefinition
    from .scene import build
    from .scorer import score
    return TaskDefinition('PR4', build, score)
