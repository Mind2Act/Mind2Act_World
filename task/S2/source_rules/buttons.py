"""Measured press/release drives a real XY stage; the old board safety stays intact."""
import math
from copy import deepcopy
from .board import Task as BoardTask

class Task(BoardTask):
    def start(self):
        super().start()
        self.stage_target=[0.,0.];self.button_armed=True;self.press_counts={k:0 for k in self.config['direction_control']['buttons']}
        self.stage_busy=False;self.commands=0
        self.return_keys=()
        self.motion_from=[0.,0.]

    def advance(self,frame):
        d=self.config['direction_control']; buttons=frame['direction_buttons']
        if set(buttons)!=set(d['buttons']):raise ValueError('Missing directional button measurements')
        if any(not math.isfinite(v['depth_m']) or type(v['robot_contact']) is not bool for v in buttons.values()):
            raise ValueError('Invalid directional button measurement')
        p=self.bodies['slider'].center; initial=self.config['objects']['slider']['position']
        offset=[p[i]-initial[i] for i in range(2)]
        super().advance(frame)
        if not self.action_enabled:return
        arrived=math.dist(offset,self.stage_target)<=d['arrival_tolerance_m']
        if arrived and self.bodies['slider'].quiet(self.config):
            self.stage_busy=False;self.motion_from=list(self.stage_target)
        pressed=[k for k,v in buttons.items() if v['depth_m']>=d['press_depth_m']]
        released=all(buttons[k]['depth_m']<d['release_depth_m'] for k in self.return_keys)
        if released:self.button_armed=True
        for k in buttons:
            self.press_counts[k]=self.press_counts[k]+1 if k in pressed and self.action_enabled else 0
        if len(pressed)>1:
            self.return_keys=tuple(pressed)
            self.button_armed=False;self.press_counts={k:0 for k in buttons};return
        if self.button_armed and len(pressed)==1 and self.press_counts[pressed[0]]>=1:
            k=pressed[0]; delta=d['cell_delta'][k]
            self.motion_from=list(self.stage_target)
            self.stage_target=[self.stage_target[i]+delta[i]*self.config['board']['cell_size'] for i in range(2)]
            self.return_keys=(k,)
            self.stage_busy=True;self.button_armed=False;self.commands+=1
            self.event('stage_command',button=k,target_m=self.stage_target)

    def public(self):
        p=super().public();p.update(instruction=self.config['instruction'],board_failed=self.status=='failure')
        return p

    def result(self):
        r=super().result();r['diagnostics']['button_commands']=self.commands;return r
