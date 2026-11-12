"""CP02 native evidence: measured cake/lid/button schedule and terminal placements.

Sources (cp02-hard-support-v3; Easy/Medium identical in cp02-em-qualified-batch):
cases_unified/CP02/timeline.py (Timeline), physical.py (MeasuredTask, gates) and
the StablePlacement tracker from the shared coupling module. scipy Rotation is
replaced by equivalent numpy quaternion maths. No expert phase label is an input.
"""
import numpy as np


def quat_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def roll_deg(q):
    """First angle of scipy's extrinsic 'xyz' Euler decomposition (rotation about x)."""
    r = quat_matrix(q)
    return float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))


class StablePlacement:
    """Current supported placement; history fields remain diagnostic."""
    def __init__(self, *args, **kwargs):
        self.grasped = self.lifted = self.complete = False

    def update(self, position, *, bilateral, elevated, released, supported, correct):
        self.grasped |= bool(bilateral)
        self.lifted |= bool(bilateral and elevated)
        self.complete = bool(supported and correct)


class Timeline:
    def __init__(self, config, world=None):
        from mind2act_eval.scoring import PressEdge
        self.spec = config
        self.level = config['difficulty']
        self.phase = 'observation'
        self.clock = dict(cue_finished=False, max_speed_error=0.)
        self.controls = dict(started=self.level == 'easy', stopped=False,
                             lid_opened=self.level == 'easy', start_tick=0,
                             lid_tick=None, start_order_valid=self.level == 'easy', failures=[])
        self.start_edge, self.stop_edge = PressEdge(.0025), PressEdge(.0025)
        self.speed = 0.

    def advance(self, step, measured):
        state = self.controls
        if not self.clock['cue_finished'] and step >= round(self.spec.get('cue_seconds', 3)*120):
            self.clock['cue_finished'] = True
            self.phase = 'execution'
            if state['started'] and not state['stopped']:
                self.speed = float(self.spec['speed_deg_s'])
        if self.level != 'easy':
            if measured['lid_angle_deg'] > 90 and not state['lid_opened']:
                state['lid_opened'], state['lid_tick'] = True, step
            if self.start_edge.update(measured['start_depth_m']):
                state.update(started=True, stopped=False, start_tick=step,
                             start_order_valid=bool(state['lid_tick'] is not None and state['lid_tick'] < step))
                self.speed = float(self.spec['speed_deg_s'])
            if self.level == 'hard' and self.stop_edge.update(measured['stop_depth_m']):
                state['stopped'] = True
                self.speed = 0.
                state['stop_tick'] = step
                state['stop_after_placements'] = bool(measured['all_decorations_stable'])
        return dict(phase=self.phase, cue_visible=not self.clock['cue_finished'], speed_deg_s=self.speed)


def decoration_correct(cue_finished, started, difficulty, lid_angle_deg, local, target, up,
                       radius):
    """Center in the visible circle, with no height or orientation gate."""
    return bool(cue_finished and np.linalg.norm(np.asarray(local)[:2]-np.asarray(target)[:2]) <= radius)


def cover_correct(stopped, angular_velocity_z, center_inside):
    return bool(stopped and abs(angular_velocity_z) <= .01 and center_inside)


class MeasuredTask:
    """Actual mechanism events plus terminal geometry, FINAL-VERSION.md."""
    def __init__(self, spec, config, initial):
        self.spec, self.config = spec, config
        self.timeline = Timeline(spec)
        self.initial = [np.asarray(row['position']) for row in initial['bodies']]
        self.gates = [StablePlacement() for _ in spec['items']]
        self.cover = StablePlacement()
        self.stop_tick = self.lid_tick = self.cover_first = None
        self.item_first = {}
        self.admitted = [False]*len(self.gates)
        self.previous_inside = [False]*len(self.gates)
        self.core_correct = [False]*len(self.gates)
        self.wrong = [False]*len(self.gates)
        self.cover_wrong = False
        self.last_omega_z = 0.
        self.latest = None

    def update(self, tick, raw):
        from copy import deepcopy
        self.latest = deepcopy(raw)
        rotor = raw['rotor']; rotation = quat_matrix(rotor['quaternion_wxyz'])
        lid_angle = -roll_deg(raw['lid']['quaternion_wxyz'])
        def depth(name):
            row = raw['buttons'].get(name)
            return float(row['reference_z_m']-row['position'][2]) if row else 0.
        previous_cue = self.timeline.clock['cue_finished']
        event = self.timeline.advance(tick, dict(lid_angle_deg=lid_angle,
            start_depth_m=depth('Start'), stop_depth_m=depth('Stop'),
            all_decorations_stable=all(g.complete for g in self.gates)))
        state = self.timeline.controls
        self.lid_tick = state['lid_tick']
        self.stop_tick = state.get('stop_tick')
        self.last_omega_z = float(rotor['angular_velocity_rad_s'][2])
        if len(raw['bodies']) != len(self.gates):
            raise ValueError('CP02 body count mismatch')
        radius = float(self.spec['region_inner_outer_radius_m'][0])
        for i, (row, item, gate) in enumerate(zip(raw['bodies'], self.spec['items'], self.gates)):
            if row['identity'] != item['identity']:
                raise ValueError('CP02 body identity mismatch')
            p = np.asarray(row['position'])
            center = p+quat_matrix(row['quaternion_wxyz'])@np.asarray(row.get('local_center_m', [0,0,0]))
            local = rotation.T@(center-np.asarray(rotor['position']))
            # Core credit is terminal geometry; cue/START/motion qualify the
            # full task flow separately and never suppress a correct macaron.
            core_inside = bool(row['supported'] and decoration_correct(True,
                state['started'], self.spec['difficulty'], lid_angle, local,
                item['target_cake_local_m'], None, radius))
            self.core_correct[i] = core_inside
            inside = bool(core_inside and self.timeline.clock['cue_finished'])
            if not inside:
                self.admitted[i] = False
            elif not self.previous_inside[i]:
                self.admitted[i] = bool(state['started'] and self.last_omega_z != 0.)
                if self.admitted[i]: self.item_first.setdefault(i, tick)
            self.previous_inside[i] = inside
            gate.update(local, bilateral=row['bilateral'], elevated=False, released=True,
                        supported=row['supported'], correct=inside and self.admitted[i])
            self.wrong[i] = bool(row['supported'] and not core_inside)
        if self.spec['difficulty'] == 'hard':
            row = raw['cover']
            ok = cover_correct(state['stopped'], self.last_omega_z,
                               raw['cover_contains_cake_center'])
            self.cover.update(row['position'], bilateral=False, elevated=False, released=True,
                              supported=row['supported'], correct=ok)
            if self.cover.complete and self.cover_first is None: self.cover_first = tick
            self.cover_wrong = bool(row['supported'] and not ok)
        return dict(event, cue_transition=not previous_cue and self.timeline.clock['cue_finished'])

    def steps(self):
        level, state = self.spec['difficulty'], self.timeline.controls
        required = len(self.gates)+(2 if level != 'easy' else 0)+(2 if level == 'hard' else 0)
        satisfied = sum(g.complete for g in self.gates)
        if level != 'easy':
            satisfied += int(state['lid_opened']) + int(state['started'] and state['start_order_valid'])
        if level == 'hard':
            stopped = bool(state['stopped'] and state.get('stop_after_placements', False)
                           and abs(self.last_omega_z) <= .01)
            satisfied += int(stopped)+int(self.cover.complete and self.cover_first is not None
                                          and self.stop_tick is not None and self.stop_tick <= self.cover_first)
        return dict(required_steps=required, steps_satisfied=int(satisfied),
                    credited_macarons=int(sum(self.core_correct)),
                    core_wrong_or_extra_items=int(sum(self.wrong)),
                    wrong_or_extra_items=int(sum(self.wrong)+self.cover_wrong))

    def grade(self):
        from mind2act_eval.scoring import SCORING_VERSION
        steps = self.steps()
        success = bool(steps['steps_satisfied'] == steps['required_steps'] and not steps['wrong_or_extra_items'])
        return dict(steps, physical_task_success=success, task_process_pass=success,
                    task_evidence_complete=True, scoring_version=SCORING_VERSION,
                    controls=dict(self.timeline.controls), cover_stable=self.cover.complete,
                    cue_finished=self.timeline.clock['cue_finished'], terminal_measurements=self.latest,
                    placements=[dict(grasped=g.grasped, lifted=g.lifted, stable=g.complete) for g in self.gates])
