from mind2act_eval.tasks import TaskDefinition
from mind2act_eval.tasks.CP01.scene import build
from mind2act_eval.tasks.CP01.scorer import score

DEFINITION = TaskDefinition("CP01", build, score)
