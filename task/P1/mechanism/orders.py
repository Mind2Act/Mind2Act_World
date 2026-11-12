"""P1 order publication clock and terminal-screen state (P-148 baseline world.update_orders).

An order is published at tick == round(at_seconds * 120). The terminal screen texture follows the baseline:
order i card while published, hard_idle between, hard_all_sent after the last order ends. Missing card
textures are asset gaps and fail at screen set time; there is no fallback imagery.
"""
HZ = 120


class OrderClock:
    def __init__(self, orders):
        self.orders = list(orders)
        self.published = set()
        self.events = []
        self.last_end = max((round((o["at_seconds"] + o["seconds"]) * HZ) for o in self.orders), default=0)

    def step(self, tick):
        """Return the screen card name to set at this tick, or None to leave it unchanged."""
        screen = None
        if any(tick == round((o["at_seconds"] + o["seconds"]) * HZ) for o in self.orders):
            screen = "hard_all_sent.png" if tick >= self.last_end else "hard_idle.png"
        for i, o in enumerate(self.orders):
            if i not in self.published and tick == round(o["at_seconds"] * HZ):
                self.published.add(i)
                self.events.append(dict(kind="order", tick=tick, index=i, order=o))
                screen = "hard_order%d.png" % i
        return screen

    def published_demand(self):
        demand = {}
        for i in sorted(self.published):
            for cup, kinds in self.orders[i]["delta"].items():
                for kind, count in kinds.items():
                    demand[cup + "/" + kind] = demand.get(cup + "/" + kind, 0) + count
        return demand

    def all_published(self, tick):
        """Publication state after step(); card visibility does not gate fulfilment."""
        return len(self.published) == len(self.orders)
