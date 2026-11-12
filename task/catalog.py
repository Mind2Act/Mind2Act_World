"""Explicit immutable catalog. Task scene imports happen only after Kit startup."""
from importlib import import_module
from types import MappingProxyType
from mind2act_eval.tasks import TaskDefinition

TASK_MODULES = MappingProxyType({
    task_id: f"mind2act_eval.tasks.{task_id}"
    for task_id in ("PR1", "PR2", "PR3", "PR4", "PR5", "PR6", "DR1", "DR2", "DR4", "E1", "S1", "P1", "F1", "M1", "M2",
                    "CP01", "CP02", "CP03", "CP04", "CP05", "CP06", "CP07", "CP08", "CP09", "CP10", "E2", "DR3", "F2", "S2", "P2", "demo")
})


def load_task(task_id):
    if task_id not in TASK_MODULES:
        raise ValueError("unregistered task: " + task_id)
    module = TASK_MODULES[task_id]
    return TaskDefinition(task_id, import_module(module + ".scene").build,
                          import_module(module + ".scorer").score)
