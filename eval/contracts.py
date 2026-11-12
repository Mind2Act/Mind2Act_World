"""Version 1 contracts shared by task ports, runner, and model adapters.

All action joints are radians, all positions metres, all time seconds.
Grippers use 0=closed, 1=open. Evidence never crosses the Adapter interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from collections.abc import Sequence
from enum import Enum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .scoring import SCORING_VERSION


def freeze(value: Any) -> Any:
    if type(value) is FrozenHistory:
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("configuration keys must be strings")
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError("configuration must contain finite JSON values")


class FrozenHistory(Sequence):
    """Persistent JSON row snapshots; appending freezes only the new row.

    Each prefix is immutable and retains its own length. Traversal is iterative
    so a full physics trace can be serialized without recursion proportional to
    the number of ticks.
    """
    __slots__ = ('_previous', '_value', '_length')

    def __init__(self):
        object.__setattr__(self, '_previous', None)
        object.__setattr__(self, '_value', None)
        object.__setattr__(self, '_length', 0)

    def __setattr__(self, name, value):
        raise TypeError('FrozenHistory is immutable')

    def __delattr__(self, name):
        raise TypeError('FrozenHistory is immutable')

    def append(self, value):
        row = freeze(value)
        result = FrozenHistory()
        object.__setattr__(result, '_previous', self)
        object.__setattr__(result, '_value', row)
        object.__setattr__(result, '_length', self._length + 1)
        return result

    def __len__(self):
        return self._length

    def __iter__(self):
        rows, current = [], self
        while current._length:
            rows.append(current._value)
            current = current._previous
        return reversed(rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if not isinstance(index, int):
            raise TypeError('history indices must be integers or slices')
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError('history index out of range')
        current = self
        for _ in range(self._length - index - 1):
            current = current._previous
        return current._value


def to_json(value: Any) -> Any:
    """Convert frozen contracts to JSON primitives without copying image arrays."""
    if is_dataclass(value):
        result = {f.name: to_json(getattr(value, f.name)) for f in fields(value)}
        # Existing clean specs are hash-pinned across deployment/projection
        # receipts. Keep their serialized identity while parsing to typed clean.
        if isinstance(value, InstanceSpec) and value.variation.mode == "clean":
            result.pop("variation")
        return result
    if isinstance(value, Mapping):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, FrozenHistory)):
        return [to_json(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class ClockSpec:
    physics_hz: int = 120
    control_hz: int = 30
    rgb_hz: int = 30

    def __post_init__(self):
        for rate in (self.physics_hz, self.control_hz, self.rgb_hz):
            if type(rate) is not int or rate <= 0:
                raise ValueError("rates must be positive integers")
        if self.physics_hz % self.control_hz or self.physics_hz % self.rgb_hz:
            raise ValueError("control/RGB rates must divide physics_hz")

    @property
    def ticks_per_action(self) -> int:
        return self.physics_hz // self.control_hz


@dataclass(frozen=True)
class VariationSpec:
    """A frozen background overlay; clean has no implicit layout or fallback."""
    mode: str = "clean"
    layout_id: str | None = None
    layout_sha256: str | None = None
    base_spec_sha256: str | None = None
    asset_manifest_sha256: str | None = None

    def __post_init__(self):
        import re
        pins = (self.layout_sha256, self.base_spec_sha256, self.asset_manifest_sha256)
        if self.mode == "clean":
            if any(v is not None for v in (self.layout_id, *pins)):
                raise ValueError("clean variation must have no layout or pins")
        elif self.mode == "random":
            if not isinstance(self.layout_id, str) or not self.layout_id.strip():
                raise ValueError("random variation requires a layout_id")
            if any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in pins):
                raise ValueError("random variation requires three lowercase SHA256 pins")
        else:
            raise ValueError("variation mode must be clean or random")


@dataclass(frozen=True)
class InstanceSpec:
    task_id: str
    instance_id: str
    difficulty: str
    source_id: str
    config_version: str
    asset_version: str
    protocol_version: str
    seed: int = 0
    clock: ClockSpec = field(default_factory=ClockSpec)
    config: Mapping[str, Any] = field(default_factory=dict)
    assets: Mapping[str, str] = field(default_factory=dict)
    variation: VariationSpec = field(default_factory=VariationSpec)

    def __post_init__(self):
        for name in ("task_id", "instance_id", "source_id", "config_version", "asset_version", "protocol_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be easy, medium, or hard")
        if isinstance(self.clock, Mapping):
            object.__setattr__(self, "clock", ClockSpec(**self.clock))
        if not isinstance(self.clock, ClockSpec):
            raise ValueError("clock must be ClockSpec or a rate mapping")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "config", freeze(self.config))
        object.__setattr__(self, "assets", freeze(self.assets))
        if isinstance(self.variation, Mapping):
            object.__setattr__(self, "variation", VariationSpec(**self.variation))
        if not isinstance(self.variation, VariationSpec):
            raise ValueError("variation must be VariationSpec or a mapping")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InstanceSpec:
        values = dict(data)
        values["clock"] = ClockSpec(**values.get("clock", {}))
        return cls(**values)

    def assert_binding(self, resolved: InstanceSpec) -> None:
        """Never substitute another difficulty/recipe or resolve only by task slug."""
        if self != resolved:
            raise ValueError("requested/resolved instance binding mismatch")


def _joints(values) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != 6 or not all(isfinite(v) for v in result):
        raise ValueError("each X5 arm requires six finite joint radians")
    return result


def _gripper(value) -> float:
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gripper must be normalized [0,1], 0=closed, 1=open")
    return value


@dataclass(frozen=True)
class Action:
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    left_gripper: float
    right_gripper: float

    def __post_init__(self):
        for name in ("left_arm", "right_arm"):
            object.__setattr__(self, name, _joints(getattr(self, name)))
        for name in ("left_gripper", "right_gripper"):
            object.__setattr__(self, name, _gripper(getattr(self, name)))


@dataclass(frozen=True)
class Observation:
    tick: int
    sim_time_s: float
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    left_gripper: float
    right_gripper: float
    rgb: Mapping[str, Any] = field(default_factory=dict)  # camera -> uint8 HWC array
    instruction: str = ""
    protocol_version: str = "v1"

    def __post_init__(self):
        if type(self.tick) is not int or self.tick < 0 or not isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError("invalid observation clock")
        for name in ("left_arm", "right_arm"):
            object.__setattr__(self, name, _joints(getattr(self, name)))
        for name in ("left_gripper", "right_gripper"):
            object.__setattr__(self, name, _gripper(getattr(self, name)))
        object.__setattr__(self, "rgb", MappingProxyType(dict(self.rgb)))


class ConsumerKind(str, Enum):
    MODEL = "model"
    EXPERT_REPLAY = "expert_replay"
    DIAGNOSTIC = "diagnostic"
    SCENE_PROBE = "scene_probe"


class Termination(str, Enum):
    COMPLETED = "completed"
    PHYSICS_BUDGET = "physics_budget"
    TRAJECTORY_EXHAUSTED = "trajectory_exhausted"
    WALL_TIMEOUT = "wall_timeout"
    ERROR = "error"
    KILLED = "killed"
    RESOURCE_LIMIT = "resource_limit"
    POLICY_STOP = "policy_stop"
    CALL_BUDGET = "call_budget"


class PolicyStop(Exception):
    """A model ends its attempt; the shared scorer still decides the score."""

    def __init__(self, reason: str, detail: str = ""):
        if reason not in ("done", "give_up") or not isinstance(detail, str):
            raise ValueError("policy stop requires done/give_up and a string detail")
        self.reason, self.detail = reason, detail
        super().__init__(reason + (": " + detail if detail else ""))


@dataclass(frozen=True)
class NativeEvidence:
    """Task-produced native measurements; no expert phase labels or model input."""
    task_id: str
    instance_id: str
    tick: int
    values: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "values", freeze(self.values))


@dataclass(frozen=True)
class Score:
    value: float
    components: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isfinite(self.value):
            raise ValueError("score must be finite")
        if not isinstance(self.components, Mapping):
            raise ValueError("score components must be a mapping")
        object.__setattr__(self, "components", freeze(self.components))


def score_from_completion(success: bool, k: int, n: int, *, components: Mapping[str, Any] | None = None) -> Score:
    """Build the CP01-CP05 partial-credit score and report completion separately."""
    if type(success) is not bool:
        raise ValueError("success must be bool")
    if type(k) is not int or type(n) is not int or n <= 0 or not 0 <= k <= n:
        raise ValueError("completion requires integer 0 <= k <= n and n > 0")
    if success and k != n:
        raise ValueError("successful completion requires k == n")
    if components is None:
        components = {}
    if not isinstance(components, Mapping):
        raise ValueError("score components must be a mapping")
    if {"task_success", "completion"}.intersection(components):
        raise ValueError("task_success and completion are reserved score components")
    details = dict(components)
    details["task_success"] = success
    details["completion"] = {"k": k, "n": n, "ratio": k / n}
    return Score(100.0 if success else 20.0 * k / n, details)


@dataclass(frozen=True)
class Health:
    ok: bool
    reason: str = ""

    def __post_init__(self):
        if type(self.ok) is not bool:
            raise ValueError("health.ok must be boolean")


@dataclass(frozen=True)
class Result:
    task_id: str
    instance_id: str
    score: float | None
    termination: Termination
    infra_valid: bool
    consumer_kind: ConsumerKind
    model_tested: bool
    ticks: int = 0
    timings: Mapping[str, float] = field(default_factory=dict)
    error: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)
    score_components: Mapping[str, Any] = field(default_factory=dict)
    task_metadata: Mapping[str, Any] = field(default_factory=dict)
    video_metadata: Mapping[str, Any] = field(default_factory=dict)
    policy_kind: str | None = None
    budget_version: str | None = None
    calls_used: int | None = None
    calls_limit: int | None = None
    decision_retry_limit: int | None = None
    budget_limits: Mapping[str, Any] = field(default_factory=dict)
    scoring_version: str = SCORING_VERSION

    def __post_init__(self):
        from .budgets import POLICY_KINDS, E2E_BUDGET_VERSION, AGENT_BUDGET_VERSION
        versions = {'e2e': E2E_BUDGET_VERSION, 'coding_agent': AGENT_BUDGET_VERSION}
        if ((self.policy_kind is None) != (self.budget_version is None)
                or self.policy_kind is not None and (self.policy_kind not in POLICY_KINDS
                    or self.budget_version != versions[self.policy_kind])):
            raise ValueError('result policy kind and budget version differ')
        used, limit = self.calls_used, self.calls_limit
        if (used is None) != (limit is None) or used is not None and (
                type(used) is not int or type(limit) is not int or not 0 <= used <= limit):
            raise ValueError('invalid result call accounting')
        if self.decision_retry_limit is not None and (
                type(self.decision_retry_limit) is not int or self.decision_retry_limit != 12):
            raise ValueError('invalid per-decision retry allowance')
        if self.policy_kind == 'e2e' and (self.calls_limit is not None or self.decision_retry_limit is not None):
            raise ValueError('e2e has no decision call limit')
        if self.policy_kind == 'coding_agent' and self.infra_valid and (
                self.calls_used is None or self.calls_limit != 150 or self.decision_retry_limit != 12):
            raise ValueError('valid coding agent result requires 150 decisions and 12 retries per decision')
        object.__setattr__(self, 'budget_limits', freeze(self.budget_limits))
        if type(self.infra_valid) is not bool or type(self.model_tested) is not bool:
            raise ValueError("infra_valid/model_tested must be bool")
        if type(self.ticks) is not int or self.ticks < 0:
            raise ValueError("ticks must be a nonnegative integer")
        object.__setattr__(self, "termination", Termination(self.termination))
        object.__setattr__(self, "consumer_kind", ConsumerKind(self.consumer_kind))
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite or null")
        if not self.infra_valid and self.score is not None:
            raise ValueError("invalid infrastructure cannot publish a numeric score")
        if self.consumer_kind == ConsumerKind.SCENE_PROBE:
            if self.score is not None or self.model_tested or self.score_components:
                raise ValueError("scene probes cannot publish scores or claim model testing")
        if self.infra_valid and self.score is None and self.consumer_kind != ConsumerKind.SCENE_PROBE:
            raise ValueError("valid episode requires numeric score")
        if self.termination in (Termination.WALL_TIMEOUT, Termination.ERROR, Termination.KILLED, Termination.RESOURCE_LIMIT) and self.infra_valid:
            raise ValueError("infrastructure termination cannot be valid")
        if self.model_tested and self.consumer_kind != ConsumerKind.MODEL:
            raise ValueError("expert/diagnostic replay cannot claim a model test")
        object.__setattr__(self, "timings", freeze(self.timings))
        object.__setattr__(self, "artifacts", freeze(self.artifacts))
        if not isinstance(self.policy_metadata, Mapping):
            raise ValueError("policy_metadata must be a JSON mapping")
        object.__setattr__(self, "policy_metadata", freeze(self.policy_metadata))
        if not isinstance(self.score_components, Mapping):
            raise ValueError("score_components must be a JSON mapping")
        object.__setattr__(self, "score_components", freeze(self.score_components))
        if not isinstance(self.task_metadata, Mapping):
            raise ValueError("task_metadata must be a JSON mapping")
        object.__setattr__(self, "task_metadata", freeze(self.task_metadata))
        if not isinstance(self.video_metadata, Mapping):
            raise ValueError("video_metadata must be a JSON mapping")
        object.__setattr__(self, "video_metadata", freeze(self.video_metadata))


class Adapter(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    def close(self) -> None: ...


class Robot(Protocol):
    def mount(self, usd_path: Path, mounts: Any) -> None: ...
    def initialize(self, world: Any) -> None: ...
    def apply(self, action: Action) -> None: ...
    def observe(self) -> Mapping[str, Any]: ...


class Simulator(Protocol):
    @property
    def stage(self) -> Any: ...
    def open_stage(self, path: Path) -> None: ...
    @property
    def world(self) -> Any: ...
    def step(self, *, render: bool) -> None: ...
    def close(self) -> None: ...


class AssetResolver(Protocol):
    def resolve(self, asset_id: str) -> Path: ...


@dataclass(frozen=True)
class BuildContext:
    sim: Simulator
    robot: Robot
    assets: AssetResolver
    output_dir: Path
    variation_session: Any = field(default=None, repr=False, compare=False)

    def prepare_variation(self, targets) -> None:
        """Builder-owned seam after USD preparation, before its existing reset."""
        if self.variation_session is not None:
            self.variation_session.apply(self.sim.stage, targets)

    @property
    def world(self) -> Any:
        return self.sim.world


class Scene(Protocol):
    def before_step(self, tick: int) -> None: ...
    def after_step(self, tick: int) -> None: ...
    def observe(self, tick: int) -> Observation: ...
    def evidence(self, tick: int) -> NativeEvidence: ...
    def health(self) -> Health: ...
    def is_complete(self) -> bool: ...
    def close(self) -> None: ...


class Builder(Protocol):
    def __call__(self, spec: InstanceSpec, ctx: BuildContext) -> Scene: ...


class ResultScene(Scene, Protocol):
    """A true final success at a completed model action, never a violation flag."""
    def is_success(self) -> bool: ...


class EventScene(Scene, Protocol):
    """The public sequence is exhausted, independently of the earned score."""
    def events_complete(self) -> bool: ...


class Scorer(Protocol):
    def __call__(self, evidence: NativeEvidence) -> Score: ...
