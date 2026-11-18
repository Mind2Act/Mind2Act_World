"""Native DR2 contact table and per-tick rows (source: mind2act.backends.isaac.world.attach contacts,
`touch` semantics preserved: active pair with impulse > 1e-7 or negative separation)."""
import math
from collections import defaultdict


def matches(path, root):
    return path == root or path.startswith(root + '/')


class ContactTable:
    def __init__(self):
        self.pairs, self._index, self._seen = {}, defaultdict(set), set()

    def report(self, pair, active, impulse, separation):
        pair = tuple(sorted(pair))
        if pair not in self._seen:
            self._seen.add(pair)
            for path in pair:
                parts = path.split('/')
                for end in range(2, len(parts) + 1):
                    self._index['/'.join(parts[:end])].add(pair)
        row = self.pairs.setdefault(pair, dict(active=False, max_impulse_Ns=0., min_separation_m=1.))
        row.update(active=bool(active), current_impulse_Ns=impulse, current_separation_m=separation)
        row['max_impulse_Ns'] = max(row['max_impulse_Ns'], impulse)
        row['min_separation_m'] = min(row['min_separation_m'], separation)

    def touch(self, root, *, robot_only=False, link=None):
        for pair in self._index.get(root, ()):
            row = self.pairs[pair]
            if not row['active'] or not (row['current_impulse_Ns'] > 1e-7 or row['current_separation_m'] < 0):
                continue
            for p, q in (pair, pair[::-1]):
                if not matches(p, root):
                    continue
                if robot_only and '/Robot' not in q:
                    continue
                if link is not None and not ('/' + link + '/' in q or q.endswith('/' + link)):
                    continue
                return True
        return False

    def unexpected(self):
        """Diagnostic only (not scored): robot contact that is neither finger-on-button nor base-on-pedestal."""
        result = []
        for pair, row in self.pairs.items():
            if not row['active'] or not (row['current_impulse_Ns'] > 1e-5 or row['current_separation_m'] < -.0005):
                continue
            if not any('/Robot' in p for p in pair):
                continue
            button = any('/CallPanel/Button' in p and '/E_button_01_13/' in p for p in pair)
            hand = any('/Robot' in p and any('/' + l + '/' in p for l in ('link7', 'link8')) for p in pair)
            if button and hand:
                continue
            if any('/base_link/' in p for p in pair) and any('/Pedestal' in p for p in pair):
                continue
            result.append(dict(pair=list(pair), impulse_Ns=row['current_impulse_Ns'], separation_m=row['current_separation_m']))
        return result


class Recorder:
    """One row per physics tick: button depression (m) and finger contact per button."""

    def __init__(self, depression, contacts_of, clock):
        self._depression, self._contacts_of, self.clock = depression, contacts_of, clock
        self.ticks = []

    def record(self, tick):
        row = dict(tick=tick, button_depression_m=[float(x) for x in self._depression()],
                   button_contacts=[bool(x) for x in self._contacts_of()])
        if not all(math.isfinite(x) for x in row['button_depression_m']):
            raise ValueError('button depression is not finite')
        self.clock.advance(tick, row['button_depression_m'], row['button_contacts'])
        self.ticks.append(row)
        return row
