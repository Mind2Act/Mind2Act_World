"""PR6 public task; frozen source/data alias PR8, native task CP11."""
from mind2act_eval.tasks import TaskDefinition
from .scene import build
from .scorer import score

DEFINITION = TaskDefinition('PR6',build,score)
