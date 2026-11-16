"""PR6 rules from FINAL-VERSION.md and native poses/contact manifolds.

Setup/propagation thresholds and whole-batch permissions retain the selected
CP11 source. Current §5–6 adds historical setup and the 0/10/20/100 stages.
No expert phase, action index, reported milestone or reference path is input.
"""
import numpy as np
from .contact_policy import contact_actions
from .curve_setup import advance_setup
from .measurements import pose, task_contact, robot_touch
from .evidence import Body


class Dominos:
    def __init__(self, config):
        self.config = config
        self.tick = -1
        self.phase, self.status, self.reason = 'setup', 'running', None
        self.ready_ticks, self.pushes, self.last_progress = 0, 0, 0
        self.chain, self.setup_assignment = [], []
        self.activated, self.fallen, self.edges, self.robot_contacts = set(), set(), set(), set()
        self.setup_tick = None
        self.bell_contact = self.clapper_struck = False
        self.bell_angle = 0.
        self.events = []
        self.motion = {}
        self.pending_pushes = {}
        self.repair_contacts = set()
        self.repaired_upright_contacts = set()

    @property
    def action_enabled(self):
        return self.tick >= self.config['cue_ticks']

    def event(self, kind, **fields):
        # Bounded by setup, at most nine push events, N edges/falls and violations.
        self.events.append(dict(tick=self.tick, kind=kind, **fields))
        if kind == 'physical_domino_setup_ready':
            self.setup_tick = self.tick

    def fail(self, reason):
        self.status, self.reason = 'failure', reason

    def sample(self, frame):
        self.tick = frame['tick']
        self.bodies = {name: Body.parse(row) for name, row in frame['bodies'].items()}
        previous = self.motion
        self.motion = {name: dict(moving=bool(np.linalg.norm(b.angular) > .15 or np.linalg.norm(b.linear) > .02),
                                  up=float(pose(frame,name)[1].apply([0,0,1])[2])) for name,b in self.bodies.items()}
        if self.status == 'failure':
            return  # Validate/audit subsequent native frames, preserve first terminal event.
        if not self.action_enabled:
            return  # Prompt actions change physics but never create violations.
        for name in ('bell', 'clapper'):
            if robot_touch(frame, name):
                self.fail(name + '_directly_touched_by_robot')
                return
        if self.phase == 'setup':
            advance_setup(self, frame)
            return
        names, c = self.chain, self.config
        contacts = {n for n in names if robot_touch(frame, n)}
        entered = contacts - self.robot_contacts
        actions = contact_actions(names, entered, self.fallen, self.pushes,
                                  self.tick-self.last_progress, c['stall_ticks'], contacts=contacts)
        # Keep a quiet-fallen repair distinct from any upright push candidate.
        # A held push does not authorize touching the tile once it starts falling.
        self.repair_contacts.intersection_update(contacts)
        self.repaired_upright_contacts.intersection_update(contacts)
        for name in entered:
            before, state = previous[name], self.motion[name]
            if (name != names[0] and not before['moving']
                    and (before['up'] < .99 or (not state['moving'] and state['up'] < .99))):
                self.repair_contacts.add(name)
        for name in list(self.repair_contacts):
            state = self.motion[name]
            # Repair motion is not a push. Authorize a pending restart only
            # after the repaired tile is both upright and quiet.
            if not state['moving'] and state['up'] > .99:
                self.repair_contacts.remove(name)
                self.pending_pushes[name] = 'restart'
        falling_touch = False
        for name in contacts:
            state = self.motion[name]
            falling = state['moving'] and (state['up'] < .99
                       or (name in entered and name in self.activated))
            before = previous[name]
            # A newly authorized first push/restart can cause motion within
            # this sampled tick. It is legal when the preceding native sample
            # was upright and quiet. A held candidate can first move later.
            new_legal_push = ((name in actions or name in self.pending_pushes) and
                              not before['moving'] and before['up'] > .99)
            # A quiet fallen tile may begin moving because the hand is picking
            # it up to repair it. That is not contact with a pre-existing fall.
            static_repair = (name in self.repair_contacts
                             and name not in self.repaired_upright_contacts)
            if falling and not (new_legal_push or static_repair):
                falling_touch = True
        if falling_touch:
            self.fail('domino_contact_before_repair_window')
            return
        # Preserve a quiet-fallen repair's permission in its first upright
        # sample, then prevent that exemption from covering a later held fall.
        self.repaired_upright_contacts.update(
            name for name in self.repair_contacts if self.motion[name]['up'] > .99)
        for name, action in actions.items():
            if action in ('push', 'restart') and name not in self.repair_contacts:
                # Contact alone is not a push. Keep this contact episode until
                # its measured native motion responds, or it ends without motion.
                self.pending_pushes[name] = action
        for name, action in list(self.pending_pushes.items()):
            if self.motion[name]['moving'] and (name in contacts or name in self.robot_contacts):
                restart = self.pushes > 0
                self.pushes += 1
                idle_ticks = self.tick-self.last_progress
                # Only the first domino may be pushed; a stalled
                # chain is stood back up and restarted from the first domino.
                if name != names[0]:
                    self.event('physical_chain_restart_rejected', item=name, pushes=self.pushes,
                               reason='push_not_first_domino')
                    del self.pending_pushes[name]
                    self.fail('push_not_first_domino')
                    return
                if restart and idle_ticks < c['stall_ticks']:
                    self.event('physical_chain_restart_rejected', item=name, pushes=self.pushes,
                               idle_ticks=idle_ticks, required_idle_ticks=c['stall_ticks'])
                    del self.pending_pushes[name]
                    self.fail('restart_before_stall_wait')
                    return
                self.activated.add(name)
                self.last_progress = self.tick
                self.event('physical_chain_push', item=name, restart=restart, pushes=self.pushes)
                del self.pending_pushes[name]
                if self.pushes > c['push_budget']:
                    self.fail('push_budget_exceeded')
                    return
            elif name not in contacts:
                del self.pending_pushes[name]
        self.robot_contacts = contacts
        for left, right in zip(names, names[1:]):
            responding = np.linalg.norm(self.bodies[right].angular) > .15 or np.linalg.norm(self.bodies[right].linear) > .02
            if left in self.activated and right not in self.activated and responding and task_contact(frame, left, right):
                self.edges.add((left, right))
                self.activated.add(right)
                self.last_progress = self.tick
                self.event('domino_contact_propagation', source=left, target=right)
        for name in names:
            if name in self.activated and name not in self.fallen and pose(frame, name)[1].apply([0,0,1])[2] < .65:
                self.fallen.add(name)
                self.last_progress = self.tick
                self.event('domino_fell', item=name)
        if names[-1] in self.activated and task_contact(frame, names[-1], 'bell'):
            self.bell_contact = True
        if self.bell_contact and task_contact(frame, 'bell', 'clapper') and not self.clapper_struck:
            self.clapper_struck = True
            self.event('physical_clapper_strike')
        self.bell_angle = float(np.degrees(np.arccos(np.clip(pose(frame, 'bell')[1].apply([0,0,1])[2], -1, 1))))
        if self.bell_contact and all(name in self.fallen for name in names) and self.status != 'success':
            self.status = 'success'
            self.event('physical_bell_ring', angle_deg=self.bell_angle)

    def facts(self):
        prefix = 0
        for name in self.chain:
            if name not in self.fallen:
                break
            prefix += 1
        violation = self.status == 'failure'
        success = self.status == 'success' and not violation
        value = (0 if violation or self.setup_tick is None else 100 if success
                 else 20 if prefix >= (len(self.chain)+1)//2 else 10)
        return dict(score=value, success=success, status=self.status, reason=self.reason,
            setup_tick=self.setup_tick, ordered_tiles=self.chain, valid_fallen_prefix=prefix,
            pushes=self.pushes, fallen=sorted(self.fallen), activated=sorted(self.activated),
            contact_edges=[list(e) for e in sorted(self.edges)], bell_contact=self.bell_contact,
            clapper_struck=self.clapper_struck, measured_bell_angle_deg=self.bell_angle,
            events=self.events)
