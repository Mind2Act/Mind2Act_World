"""Native F1 contacts and per-tick rows (source: development_world.py contact_report/cap_contact)."""
import numpy as np

CONTACT_EPS = 1e-7


class ContactTable:
    def __init__(self, finger_links):
        self.pairs = {}
        self.links = tuple('/' + l + '/' for l in finger_links)

    def report(self, pair, active, impulse, separation):
        pair = tuple(sorted(pair))
        if not any('/Robot' in v or '/Button' in v for v in pair):
            return
        rec = self.pairs.setdefault(pair, dict(active=False, max_impulse_Ns=0., min_separation_m=1.))
        rec.update(active=bool(active), current_impulse_Ns=impulse, current_separation_m=separation)
        rec['max_impulse_Ns'] = max(rec['max_impulse_Ns'], impulse)
        rec['min_separation_m'] = min(rec['min_separation_m'], separation)

    def finger_pairs(self, button, side=None):
        """Active finger-on-cap pairs for one button, optionally restricted to one arm."""
        cap = '/Button' + button + '/Asset/E_button_01_13/'
        robot = '/Robot' + side + '/' if side else '/Robot'
        return [list(pair) for pair, r in self.pairs.items()
                if r['active'] and (r['current_impulse_Ns'] > CONTACT_EPS or r['current_separation_m'] < 0)
                and any(cap in p for p in pair)
                and any(robot in p and any(l in p for l in self.links) for p in pair)]

    def cap_contact(self, button):
        return bool(self.finger_pairs(button))


def measure_row(tick, stage, q, slider, depths, contacts, goal_visible, events, failures):
    row = dict(tick=tick, stage=stage, q=[float(x) for x in q], pos=slider.pos,
               slider_motion_complete=slider.motion_complete(), slider_position=[float(x) for x in slider.xy_world()],
               button_depths={b: float(d) for b, d in depths.items()}, button_contacts={b: bool(c) for b, c in contacts.items()},
               goal_visible=bool(goal_visible), events=events, mechanism_failures=failures)
    if not np.isfinite(row['q']).all():
        raise ValueError('nonfinite measured joints')
    return row
