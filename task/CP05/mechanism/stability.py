from collections import deque
import numpy as np

class StablePlacement:
    def __init__(self, ticks=120, tolerance=.003, require_manipulation=True):
        self.ticks=ticks;self.tolerance=tolerance;self.positions=deque(maxlen=ticks)
        self.grasped=False;self.lifted=False;self.require_manipulation=require_manipulation

    def update(self, position, *, bilateral, elevated, released, supported, correct):
        self.grasped |= bool(bilateral)
        self.lifted |= bool(bilateral and elevated)
        self.positions.clear()
        if supported and correct:
            self.positions.append(np.asarray(position).copy())

    @property
    def complete(self):
        return bool(self.positions)
