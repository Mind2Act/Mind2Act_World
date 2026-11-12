"""Versioned policy budgets, separate from physical consumer identity."""
from collections.abc import Mapping
from math import isfinite
from threading import Lock

from .contracts import PolicyStop

E2E_BUDGET_VERSION = 'e2e-p90x1.2-v1'
AGENT_BUDGET_VERSION = 'coding-agent-e2e-x4-v1'
E2E_WALL_S = 7200
AGENT_WALL_S = 21600
POLICY_KINDS = ('e2e', 'coding_agent')


class CallBudgetExhausted(PolicyStop):
    def __init__(self, dimension):
        self.dimension = dimension
        super().__init__('give_up', dimension + ' exhausted')


class DecisionBudget:
    """Count completed model decisions; transport retries do not spend decisions."""
    retry_limit = 12

    def __init__(self, decision_limit=150, *, recorder=None):
        if type(decision_limit) is not int or decision_limit <= 0:
            raise ValueError('positive integer decision limit required')
        self.limit = decision_limit
        self.used = 0
        self._lock = Lock()
        self.recorder = recorder

    def ensure_available(self):
        with self._lock:
            if self.used >= self.limit:
                raise CallBudgetExhausted('decision_call_limit')

    def decision(self):
        """Record a complete reply, including one requiring action repair."""
        with self._lock:
            if self.used >= self.limit:
                raise CallBudgetExhausted('decision_call_limit')
            self.used += 1
            if self.recorder is not None:
                self.recorder(self.receipt())

    def receipt(self):
        return dict(calls_used=self.used, calls_limit=self.limit,
                    decision_retry_limit=self.retry_limit)


def validate_e2e_fields(row):
    """Old releases remain readable; a new row must bind both fields together."""
    limits, version = row.get('limits_e2e'), row.get('budget_version_e2e')
    if limits is None and version is None:
        return
    original = row.get('limits')
    if (not isinstance(original, Mapping) or set(original) != {'physics_ticks', 'wall_s', 'retries'}
            or any(type(original[k]) is not int or original[k] <= 0 for k in ('physics_ticks', 'wall_s'))
            or type(original['retries']) is not int or original['retries'] != 0):
        raise ValueError('versioned e2e row must retain its original published limits')
    if (version != E2E_BUDGET_VERSION or not isinstance(limits, Mapping)
            or set(limits) != {'physics_ticks', 'wall_s', 'retries'}
            or type(limits['physics_ticks']) is not int or limits['physics_ticks'] <= 0
            or limits['physics_ticks'] % 4 or type(limits['wall_s']) is not int
            or limits['wall_s'] < 1800 or type(limits['retries']) is not int
            or limits['retries'] != 0):
        raise ValueError('invalid versioned e2e limits')


def selected_limits(row, policy_kind=None, budget_version=None):
    """Selecting a new protocol is explicit; legacy/default limits never change."""
    if policy_kind is None:
        if budget_version is not None:
            raise ValueError('budget_version requires policy_kind')
        return row['limits']
    versions = {'e2e': E2E_BUDGET_VERSION, 'coding_agent': AGENT_BUDGET_VERSION}
    if policy_kind not in versions or budget_version != versions[policy_kind]:
        raise ValueError('unknown policy kind or budget version')
    validate_e2e_fields(row)
    if row.get('budget_version_e2e') != E2E_BUDGET_VERSION:
        raise ValueError('case has no frozen e2e budget for this policy')
    ticks = row['limits_e2e']['physics_ticks']
    return dict(physics_ticks=ticks * (4 if policy_kind == 'coding_agent' else 1),
                wall_s=AGENT_WALL_S if policy_kind == 'coding_agent' else E2E_WALL_S,
                retries=0)


def validate_execution_limits(actual, expected, *, allow_longer_wall=False):
    for key, value in expected.items():
        supplied = getattr(actual, key, None)
        if key == 'wall_s':
            if (isinstance(supplied, bool) or not isinstance(supplied, (int, float))
                    or not isfinite(supplied)
                    or (supplied < value if allow_longer_wall else supplied != value)):
                raise ValueError('execution wall_s differs from the frozen infrastructure allowance')
        elif type(supplied) is not type(value) or supplied != value:
            raise ValueError('execution limits differ from selected policy budget')
