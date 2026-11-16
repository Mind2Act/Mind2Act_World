"""Measured E1 batch completion.

Each batch opens for submissions in phase `batch_execute`. Every item that is really placed inside the
tray and released is registered for that batch (removing it later does not clear it). The batch seals at
the first tick where any item completes a continuous 1.0 s window (inside, floor-supported, released,
position range < 3 mm relative to the tray). There is no per-batch deadline, so a batch
never sealed before the episode budget ends contributes 0. All events of
that physical tick are collected first. A sealed batch is valid only if exactly one item is registered,
it is the batch's unique repeated item, and its window is complete. Errors never stop later batches.
Measurements only: no expert labels.
"""
from collections import deque
import numpy as np
from mind2act_eval.contracts import NativeEvidence
from .scorer import N_BY_DIFFICULTY
from .submission import occupied_items

WINDOW_TICKS = 121  # 120 intervals = 1.0 s at 120 Hz
RANGE_M = .003
LIFT_M = .025       # diagnostic only; not a scoring condition
SUBMISSION_RULE = 'e1-g05-submission-current-score-v1'


class Evidence:
    def __init__(self, answer_identities, difficulty):
        if difficulty not in N_BY_DIFFICULTY:
            raise ValueError("E1 unknown difficulty: %r" % (difficulty,))
        self.answers, self.difficulty = answer_identities, difficulty
        self.windows, self.lifted, self.baselines = {}, set(), {}
        self.registered = {}  # round -> set(item paths)
        self.registration_ticks = {}  # diagnostic: first released occupancy, not completion
        self.results = []
        self.last_tick = 0

    def _window(self, key):
        return self.windows.setdefault(key, deque(maxlen=WINDOW_TICKS))

    @staticmethod
    def _stable(window):
        if len(window) < WINDOW_TICKS:
            return False
        points = np.asarray(window)
        return bool(np.linalg.norm(points[:, None] - points[None, :], axis=-1).max() < RANGE_M)

    def sample(self, row):
        """Feed one physics tick; return True when a batch was sealed at this tick."""
        if row["physics_tick"] != self.last_tick + 1:
            raise ValueError("E evidence clock gap")
        self.last_tick = row["physics_tick"]
        rn = row["round"]
        if row["phase"] != "batch_execute":
            return False
        if rn != len(self.results) + 1:
            if rn <= len(self.results):
                return False  # already sealed; departure is in progress
            raise ValueError("E batches must be sealed in order")
        items = row["items"]
        bounds = {i["path"]: (i["center"], i["dimensions"]) for i in items}
        contacts = {tuple(r["paths"]): dict(active=True) for r in row["contacts"]}
        occupied = occupied_items(bounds, np.asarray(row["goal_xy"]), np.asarray(row["inner_xy"]),
                                  row["z_bounds"], contacts, rn)
        registered = self.registered.setdefault(rn, set())
        registration_ticks = self.registration_ticks.setdefault(rn, {})
        goal = np.asarray(row["goal_xy"], float)
        stable = {}
        for item in items:
            key = item["path"]
            center = np.asarray(item["center"], float)
            dims = np.asarray(item["dimensions"], float)
            if (center.shape != (3,) or dims.shape != (3,) or not np.isfinite(np.r_[center, dims]).all()
                    or (dims <= 0).any() or type(item['support']) is not bool
                    or type(item['touch_robot']) is not bool):
                raise ValueError("Invalid measured E geometry")
            self.baselines.setdefault(key, item["baseline_z"])
            if item["bilateral"] and center[2] > self.baselines[key] + LIFT_M:
                self.lifted.add(key)
            inside = bool(np.all(abs(center[:2] - goal) + dims[:2] / 2 < row["inner_xy"]))
            released = not item["touch_robot"]
            if key in occupied and released:
                registered.add(key)
                registration_ticks.setdefault(key, row['physics_tick'])
            window = self._window(key)
            if inside and item["support"] and released:
                window.append(np.r_[center[:2] - goal, center[2]])
            else:
                window.clear()
            stable[key] = self._stable(window)
        if not any(stable.values()):
            return False
        identity = self.answers[rn]
        target = next(i for i in items if i["identity"] == identity)["path"]
        submitted = sorted(registered)
        passed = bool(len(submitted) == 1 and submitted[0] == target and stable[target])
        first_stable = next(key for key, ready in stable.items() if ready)
        points = np.asarray(self.windows[first_stable])
        self.results.append(dict(round=rn, **{"pass": passed}, identity=identity, target=target, submitted=submitted,
                                 sealed_by="stable_window", tick=row["physics_tick"],
                                 target_lifted=target in self.lifted,
                                 submission_locked_tick=row['physics_tick'],
                                 stable_start_tick=row['physics_tick'] - WINDOW_TICKS + 1,
                                 stable_window_ticks=WINDOW_TICKS,
                                 stable_position_range_m=float(np.linalg.norm(points[:, None] - points[None, :], axis=-1).max()),
                                 stable_candidate=first_stable, registration_ticks=dict(registration_ticks),
                                 submission_rule=SUBMISSION_RULE))
        return True

    def native_evidence(self, task_id, instance_id, tick):
        return NativeEvidence(task_id, instance_id, tick, dict(difficulty=self.difficulty, rounds=list(self.results),
                              submission_rule=SUBMISSION_RULE, stable_window_s=1.0, stable_range_m=RANGE_M,
                              source_g05_evidence_sha256='ce4d469d62efbfcda6c91a6afdd93db3a4f50d1266bc0ec905ed82ea9d59a9bb'))
