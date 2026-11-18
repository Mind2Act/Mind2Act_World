from mind2act_eval.tasks import TaskDefinition
from mind2act_eval.tasks.CP03.scene import build
from mind2act_eval.tasks.CP03.scorer import score

DEFINITION = TaskDefinition("CP03", build, score)
