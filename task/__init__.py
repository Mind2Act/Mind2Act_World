"""Explicit task catalog, with no process-global mutable registration."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from mind2act_eval.contracts import Builder, Scorer, InstanceSpec, BuildContext, Scene


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    build: Builder
    scorer: Scorer


class TaskRegistry:
    def __init__(self, definitions: Iterable[TaskDefinition]):
        entries = {}
        for definition in definitions:
            if definition.task_id in entries:
                raise ValueError("duplicate task: " + definition.task_id)
            entries[definition.task_id] = definition
        self._entries = MappingProxyType(entries)

    def get(self, task_id: str) -> TaskDefinition:
        try:
            return self._entries[task_id]
        except KeyError:
            raise ValueError("unregistered task: " + task_id) from None

    def build(self, requested: InstanceSpec, resolved: InstanceSpec, ctx: BuildContext) -> Scene:
        requested.assert_binding(resolved)
        return self.get(requested.task_id).build(resolved, ctx)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)
