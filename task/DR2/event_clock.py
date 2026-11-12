"""Native button depression edges and independent cue credit; FINAL-VERSION.md."""


class EventClock:
    def __init__(self, sequence, press_ticks, release_ticks, gap_ticks, release_depression=.0008, press_depression=.0025):
        if not sequence or any(b not in (0, 1, 2) for b in sequence):
            raise ValueError('Expected a nonempty three-button sequence')
        if min(press_ticks, release_ticks, gap_ticks) <= 0:
            raise ValueError('Positive timing budgets required')
        self.sequence = list(sequence)
        self.press_ticks, self.release_ticks, self.gap_ticks = press_ticks, release_ticks, gap_ticks
        self.release_depression, self.press_depression = release_depression, press_depression
        self.armed = [True] * 3
        self.records, self.events = [], []
        self.current = None
        self.next_due = 1
        self.unlit_presses = []
        self.last_tick = 0

    @property
    def done(self):
        """Every scheduled cue has resolved; there is no post-sequence gap."""
        return (self.current is None and len(self.records) == len(self.sequence)
                and all(r['status'] in ('valid', 'failed') for r in self.records))

    @property
    def valid_count(self):
        return sum(r['status'] == 'valid' for r in self.records)

    def _fail(self, record, reason, tick):
        if record['status'] != 'failed':
            record.update(status='failed', reason=reason)
        if record is self.current:
            self.current = None
        self.events.append({'kind': 'fail', 'index': record['index'], 'reason': reason, 'tick': tick})

    def advance(self, tick, depression, contact):
        if tick != self.last_tick + 1:
            raise ValueError('Clock requires every physics tick exactly once')
        self.last_tick = tick
        for b in range(3):
            if depression[b] < self.release_depression:
                self.armed[b] = True
        for b in range(3):  # presses: only an armed (released) button can press, so holding never repeats
            if not (self.armed[b] and depression[b] >= self.press_depression):
                continue
            self.armed[b] = False
            cue = self.current
            self.events.append({'kind': 'press', 'button': b, 'tick': tick, 'cue': cue['button'] if cue else None})
            if cue is not None:
                if b == cue['button'] and cue['press_tick'] is None:
                    if tick - cue['cue_tick'] > self.press_ticks:
                        self._fail(cue, 'press_timeout', tick)  # the window closed before this press
                    else:
                        cue['press_tick'] = tick
                else:
                    self.unlit_presses.append(dict(button=b, tick=tick, cue=cue['index']))
            else:
                self.unlit_presses.append(dict(button=b, tick=tick, cue=None))
        cue = self.current
        if cue is not None:
            if cue['press_tick'] is None and tick - cue['cue_tick'] > self.press_ticks:
                self._fail(cue, 'press_timeout', tick)
            elif cue['press_tick'] is not None:
                if self.armed[cue['button']] and cue['release_tick'] is None:
                    cue['release_tick'] = tick
                    if tick - cue['press_tick'] > self.release_ticks:
                        self._fail(cue, 'release_timeout', tick)
                    else:
                        cue['status'] = 'valid'
                        self.current = None
                        self.events.append({'kind': 'release', 'index': cue['index'], 'button': cue['button'], 'tick': tick})
                elif cue['release_tick'] is None and tick - cue['press_tick'] > self.release_ticks:
                    self._fail(cue, 'release_timeout', tick)
        if self.current is None and len(self.records) < len(self.sequence):
            if self.next_due is not None and not all(self.armed):
                self.next_due = None  # any button down inside the gap restarts it after the release
            if self.next_due is None and all(self.armed):
                self.next_due = tick + self.gap_ticks
            if self.next_due is not None and tick >= self.next_due:
                i = len(self.records)
                self.current = {'index': i, 'button': self.sequence[i], 'cue_tick': tick, 'press_tick': None,
                                'release_tick': None, 'status': None, 'reason': None}
                self.records.append(self.current)
                self.next_due = None
                self.events.append({'kind': 'cue', 'index': i, 'button': self.current['button'], 'tick': tick})
