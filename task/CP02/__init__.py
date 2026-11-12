from mind2act_eval.tasks import TaskDefinition
from mind2act_eval.tasks.CP02.scene import build
from mind2act_eval.tasks.CP02.scorer import score

DEFINITION = TaskDefinition("CP02", build, score)
