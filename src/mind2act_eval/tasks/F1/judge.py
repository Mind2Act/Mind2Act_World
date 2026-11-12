"""F1 physical-edge judge: completed probe history, reset/reveal and the shortest-route success check.

Ported from source `task_evidence.Evidence` (source_id f-v4-original-train-camera-v1). Reads only native rows:
joints, button depression/contact, slider pose, goal visibility, environment stage. Per CommonInput snapshot 20261101
F1 section 5: every button needs at least one real press-release in exploration, in any order, repeats allowed
(the source demanded the instance's recorded expert order); L* is a BFS over the real transition graph. Used online by the scene for termination and
offline by the scorer; both see the same rows.
"""
from collections.abc import Mapping

import numpy as np

from .space import moved, shortest, slider_xy


def plain(value):
    """Frozen evidence uses tuples/mappingproxy; positions compare as lists."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class Judge:
    def __init__(self, case, protocol, layout):
        self.case, self.protocol, self.layout = plain(case), plain(protocol), layout
        case = self.case
        self.last_tick = 0
        self.failures = set()
        self.probes, self.executions = [], []
        self.pending = False
        self.reset_ticks = self.reveal_ticks = self.ready_ticks = self.final_ticks = 0
        self.previous_stage = 'exploration'
        self.reset_complete = self.goal_seen = False
        self.armed = {b: True for b in case['buttons']}
        self.hold_ticks = round(case['timing']['hold_seconds'] * 120)

    def sample(self, row):
        row = plain(row)
        case, protocol = self.case, self.protocol
        if row['tick'] != self.last_tick + 1:
            raise ValueError('F evidence clock gap')
        self.last_tick, stage = row['tick'], row['stage']
        if stage not in ('exploration', 'reset', 'goal_reveal', 'execution'):
            raise ValueError('Unknown F stage')
        q = np.asarray(row['q'], float).reshape(2, 6)
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite F measured joints')
        depths, touch = row['button_depths'], row['button_contacts']
        if set(depths) != set(case['buttons']) or not np.isfinite(list(depths.values())).all():
            raise ValueError('Missing F cap evidence')
        released = all(depths[b] < protocol['release_threshold_m']  for b in case['buttons'])
        actual = np.asarray(row['slider_position'], float)
        if actual.shape != (2,) or not np.isfinite(actual).all():
            raise ValueError('Missing measured F slider transform')
        if row['slider_motion_complete'] and not np.allclose(
                actual, slider_xy(case['layout'], case['space'], row['pos'], self.layout['slider_end_margin_m']), rtol=0, atol=1e-5):
            self.failures.add('displayed_slider_disagrees_with_logical_state')
        for b in case['buttons']:
            if depths[b] < protocol['release_threshold_m'] :
                self.armed[b] = True
        for event in row['events']:
            self._edge(event, row, stage, depths, touch)
        if self.pending and released and row['slider_motion_complete']:
            self.pending = False
        all_probed = self.all_probed()
        if stage == 'exploration':
            if row['goal_visible']:
                self.failures.add('goal_leaked_during_exploration')
            ok = all_probed and released and not self.pending and row['slider_motion_complete']
            self.ready_ticks = self.ready_ticks + 1 if ok else 0
        elif stage == 'reset':
            if not all_probed:
                self.failures.add('reset_reveal_without_full_probe_release_history')
            self.reset_ticks += 1
            if (all_probed and not self.pending and released and row['slider_motion_complete']
                    and row['pos'] == case['execution']['start']):
                self.reset_complete = True
        else:
            if not self.reset_complete or not row['goal_visible']:
                self.failures.add('missing_visible_goal_history')
            else:
                self.goal_seen = True
            if stage == 'goal_reveal':
                self.reveal_ticks += 1
        # Historical traces may carry the retired animation/reveal wait failures.
        failures = set(row['mechanism_failures']) - {'new_probe_before_release_and_slider_completion', 'press_during_environment_reset'}
        if stage == 'goal_reveal':
            failures.discard('press_during_environment_reset')
        self.failures.update(failures)
        last_key_returned = (not self.executions or
                             depths[self.executions[-1]['button']] < protocol['release_threshold_m'])
        arrived = (stage in ('execution', 'goal_reveal') and row['pos'] == case['execution']['target']
                   and row['slider_motion_complete'] and last_key_returned)
        self.final_ticks = self.final_ticks + 1 if arrived else 0
        self.previous_stage = stage
        self.last_released, self.last_pos = last_key_returned, row['pos']
        self.last_actual, self.last_motion = actual, bool(row['slider_motion_complete'])
        return self.grade()

    def all_probed(self):
        return set(self.case['buttons']) <= set(self.probes)

    def _edge(self, event, row, stage, depths, touch):
        case, protocol = self.case, self.protocol
        b = event['button']
        if (b not in case['buttons'] or event['tick'] != row['tick']
                or event['depression_m'] < protocol['press_threshold_m']
                or depths.get(b, 0) < protocol['press_threshold_m']):
            self.failures.add('edge_without_current_physical_travel')
            return
        if not self.armed[b]:
            self.failures.add('repeated_edge_without_measured_release')
        self.armed[b] = False
        if event['after'] != moved(case['space'], case['mapping'], event['before'], b):
            self.failures.add('incorrect_slider_effect')
        if stage == 'exploration':
            self.pending = True
            self.probes.append(b)  # repeated probes are allowed; each button needs at least one
        elif stage in ('execution', 'goal_reveal'):
            if not self.reset_complete or not row['goal_visible']:
                self.failures.add('execution_before_complete_reveal')
            expected = case['execution']['start'] if not self.executions else self.executions[-1]['after']
            if event['before'] != expected:
                self.failures.add('execution_state_discontinuity')
            self.executions.append(event)
        # A reset-phase press affects the mechanism but is not a formal answer.

    def at_target(self):
        """Arrived = the logical position is the goal AND the displayed slider has actually finished moving onto it."""
        ex = self.case['execution']
        if self.last_pos != ex['target'] or not self.last_motion or self.last_actual is None:
            return False
        goal = slider_xy(self.case['layout'], self.case['space'], ex['target'], self.layout['slider_end_margin_m'])
        return bool(np.allclose(self.last_actual, goal, rtol=0, atol=1e-5))

    def grade(self):
        ex = self.case['execution']
        short = shortest(self.case['space'], self.case['mapping'], ex['start'], ex['target'])
        complete = bool(not self.failures and self.all_probed() and self.reset_complete
                        and self.goal_seen and len(self.executions) == short
                        and self.final_ticks >= 1)
        return dict(task_success=complete, at_target=self.at_target(), final_released=bool(self.last_released),
                    prerequisites=bool(self.all_probed() and self.reset_complete
                                       and self.goal_seen), execution_presses=len(self.executions), shortest_presses=short,
                    path_redundancy=max(0, len(self.executions) - short), effective_probes=len(set(self.probes)),
                    protocol_violations=sorted(self.failures), reset_ticks=self.reset_ticks,
                    reveal_ticks=self.reveal_ticks, final_stable_ticks=self.final_ticks)
