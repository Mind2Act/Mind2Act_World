"""PR5 prepared scene and staged score, shared by all evaluation consumers."""
from mind2act_eval.tasks import TaskDefinition
from .scene import build
from .scorer import score

DEFINITION = TaskDefinition('PR5', build, score)
