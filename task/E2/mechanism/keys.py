"""Password input from mechanical depression/return edges, independent of contact."""
import math

PRESS_M = -.003
RELEASE_M = -.001
PRESS_TICKS = RELEASE_TICKS = 1


class Keys:
    def __init__(self):
        self.armed = [True] * 9
        self.pending = {}
        self.events = []
        self.last_tick = 0
        self.serial_violation = False
        self.travel = [0.] * 9

    def sample(self, tick, travel, contacts, *, enabled):
        if type(tick) is not int or tick != self.last_tick + 1:
            raise ValueError('E2 measurements must cover consecutive physics ticks from 1')
        if len(travel) != 9 or len(contacts) != 9:
            raise ValueError('E2 needs nine measured keys')
        if any(type(q) not in (float, int) or not math.isfinite(q) for q in travel):
            raise ValueError('E2 key travel must be finite')
        if any(type(c) is not bool for c in contacts) or type(enabled) is not bool:
            raise ValueError('E2 physical contacts and input gate must be boolean')
        self.last_tick, self.travel = tick, list(travel)
        edges = []
        for slot, q in enumerate(travel):
            if q > RELEASE_M:
                self.armed[slot] = True
                if slot in self.pending:
                    self.pending.pop(slot)['release_tick'] = tick
            if q <= PRESS_M and self.armed[slot]:
                self.armed[slot] = False
                if enabled:
                    edges.append(slot)
        # Out-of-turn edges do not enter the string. The cap must physically
        # return before that key can produce a new, admissible edge.
        if len(edges) != 1 or self.pending:
            self.serial_violation |= bool(edges)
            return
        slot = edges[0]
        event = dict(start_tick=tick, press_tick=tick, slot=slot, digit=slot+1)
        self.events.append(event)
        self.pending[slot] = event

    @property
    def released(self):
        return not self.events or self.travel[self.events[-1]['slot']] > RELEASE_M

    @property
    def digits(self):
        return [event['digit'] for event in self.events]

    def complete(self, target):
        return self.digits == list(target) and self.released
