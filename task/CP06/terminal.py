"""Incremental terminal candidate without transport, release or time gates."""
from collections import deque
from mind2act_eval.scoring import PressEdge
from .physical import terminal_units,DropFacts


class TerminalFacts:
    def __init__(self,config,initial):
        self.config=config;self.tail=deque(maxlen=360)
        self.zero=config['difficulty']=='easy';self.edge=PressEdge(.004)
        self.drops=DropFacts(config,initial);self.tick=0

    def append(self,state):
        self.tick+=1;self.drops.append(state,self.tick)
        self.tail.append(state)
        if self.config['difficulty']!='easy' and self.edge.update(state['control']['button_travel_m']):
            if not any(any(o['pan_supported'].values()) for o in state['objects'].values()):self.zero=True

    def candidate(self):
        return bool(self.tail and not self.drops.dropped
                    and all(terminal_units(self.config,self.tail[-1],self.zero,self.drops.dropped)[0]))
