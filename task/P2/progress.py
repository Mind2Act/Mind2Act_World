"""Terminal continuous bottom prefix; process history never grants eligibility."""
from types import SimpleNamespace
from .measurements import supports, validate_frame, tilt
from .binding import geometry
import math


class Progress:
    def __init__(self, source, initial):
        self.targets = [f'Tier{i+1}' for i in range(len(source['target_foods_bottom_to_top']))]
        objects = self.targets + ['Lid'] + list(source['decoy_xyz'])
        validate_frame(initial, objects)
        if initial['physics_tick'] != 0:
            raise ValueError('P2 native reset sample must be tick0')
        self.final = geometry(source)['final_xyz']
        self.facts = SimpleNamespace(tick=0, last=initial, objects=objects, decoys=list(source['decoy_xyz']))
        self.previous_time = initial['simulation_time_s']

    def sample(self, row):
        validate_frame(row, self.facts.objects)
        if row['physics_tick'] != self.facts.tick+1 or abs(row['simulation_time_s']-self.previous_time-1/120) > 1e-6:
            raise ValueError('P2 consecutive 120Hz native samples required')
        self.facts.tick, self.facts.last = row['physics_tick'], row
        self.previous_time = row['simulation_time_s']

    def placed(self, name, below):
        row = self.facts.last
        position, expected = row['poses'][name]['position'], self.final[name]
        precise = (math.dist(position[:2], expected[:2]) <= .01+1e-12
                   and abs(position[2]-expected[2]) <= .003+1e-12
                   and tilt(row['poses'][name]['quaternion_wxyz']) <= 5.+1e-12)
        return precise and below in supports(row, name)

    def prefix(self, row=None):
        prefix = 0
        for i, name in enumerate(self.targets):
            if not self.placed(name, 'Stove' if i == 0 else self.targets[i-1]):
                break
            prefix += 1
        return prefix

    def terminal_stack(self):
        row = self.facts.last
        stack = {'Stove'}
        while True:
            added = {n for n in self.facts.objects if any(p in stack for p in supports(row, n))}
            if added <= stack:
                return stack-{'Stove'}
            stack.update(added)

    def result(self):
        prefix = self.prefix()
        lid = prefix == len(self.targets) and self.placed('Lid', self.targets[-1])
        stack = self.terminal_stack()
        success = lid and stack == set(self.targets+['Lid'])
        return dict(valid_disassembly=False, restored_prefix=prefix, target_count=len(self.targets),
                    final_conditions_met=bool(success), terminal_stack=sorted(stack), task_success=bool(success),
                    lid_replaced=bool(lid), final_stable=None)
