"""Record every measured replay press, including invalid tool contacts."""


class ReplayEvents:
    def __init__(self):
        self.events = []
        self.pending = {}

    def observe(self, step, *, phase, before_down, after_down, depths,
                tips, finger_keys, tool_category):
        # Retain the native depth dictionary's deterministic key order.
        for key in depths:
            if phase == 'replay' and key in after_down and key not in before_down:
                valid = key in tips and key not in finger_keys
                event = {'index': len(self.events), 'phase': 'replay', 'key': int(key),
                         'tool_id': tool_category(key, valid), 'nib_contact': bool(valid),
                         'press_tick': int(step), 'release_tick': None}
                self.events.append(event)
                self.pending[int(key)] = event
        for key in before_down - after_down:
            event = self.pending.pop(int(key), None)
            if event is not None:
                event['release_tick'] = int(step)
