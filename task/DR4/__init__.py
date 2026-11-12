"""DR4 (source/data alias PR7): one prepared scene and current staged scorer."""
from mind2act_eval.tasks import TaskDefinition
from .scene import build
from .scorer import score

DEFINITION = TaskDefinition('DR4', build, score)
