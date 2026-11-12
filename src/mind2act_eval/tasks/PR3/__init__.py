"""PR3 trajectory tracing. Importing this package needs neither pxr nor Isaac."""
from mind2act_eval.tasks import TaskDefinition
from .scene import build
from .scorer import score

DEFINITION = TaskDefinition("PR3", build, score)
