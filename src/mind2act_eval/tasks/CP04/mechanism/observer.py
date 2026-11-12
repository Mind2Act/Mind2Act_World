"""Measured CP04 evidence for the candidate pilot, no policy phases or oracles.

All poses/speeds are native physics samples. This observer is deliberately
conservative: task failures remain valid negatives, missing measurements raise.
Stability is observed for 0.5 s; the thresholds are recorded in the run manifest.
"""
import math
from collections import deque
from copy import deepcopy

IDS = ('chili', 'garlic', 'ginger', 'onion')


class Observer:
    def __init__(self, contract, initial):
        self.contract = contract
        self.recipe = contract['recipe']
        self.initial = initial
        self.hidden = None
        self.drawer = None
        self.lid = None
        self.lid_was_supported = False
        self.ingredients = []
        self.added = set()
        self.thermal = []
        self.band = None
        self.peak = None
        self.hold = None
        self.warm_start = None
        self.tick = 0
        self.windows = {name: deque(maxlen=16) for name in IDS}
        self.latest = None
        self.latest_sample_tick = None
        self.faults = set()

    def thermal_sample(self, tick, row):
        self.tick = tick
        mode = row['mode'].replace(' ', '_')
        temp = float(row['temperature'])
        if not math.isfinite(temp):
            raise ValueError('nonfinite measured thermal state')
        if not self.thermal or mode != self.thermal[-1]['mode']:
            self.thermal.append(dict(mode=mode, tick=tick, temperature_c=temp,
                                     display_temperature_c=temp))
        if self.band is None and row['window_entered'] is not None:
            self.band = min(tick, int(round(row['window_entered'] * 120)))
        if mode == 'KEEP_WARM':
            self.warm_start = self.warm_start or tick
            self.peak = max(temp, self.peak or temp)
            if tick - self.warm_start >= 120 and row['valid_switch'] and self.hold is None:
                self.hold = tick
        elif self.peak is not None:
            self.peak = max(self.peak, temp)

    @staticmethod
    def contained(xyz):
        return math.hypot(xyz[0] - .106139, xyz[1] - .024681) < .089 and .826 < xyz[2] < .910

    @staticmethod
    def speed_ok(sample, prefix):
        return sample[prefix + '_speed'] < .015 and sample[prefix + '_angular_speed'] < .15

    def sample(self, tick, measured):
        self.tick = tick
        self.latest = measured
        self.latest_sample_tick = tick
        if self.drawer is None and abs(measured['drawer'][1] + .36) < .01:
            self.drawer = tick
        for name in IDS:
            row = measured['jars'][name]
            if not all(math.isfinite(x) for key in ('body', 'cap') for x in row[key]):
                raise ValueError('nonfinite native jar pose')
            if row['body'][2] < .70 or row['cap'][2] < .70:
                self.faults.add('object_fell_below_table:' + name)
            self.windows[name].append((tick, deepcopy(row)))
            food = measured['food'][name]
            # Preserve wrong/extra ingredients too, so recipe scoring can fail.
            if name not in self.added and len(food) == len(measured['food_in_pot'][name]) == 3 and all(measured['food_in_pot'][name]):
                # One native sample proves a group, not an order within it.
                self.ingredients.append(dict(ingredient_id=name, tick=tick,
                    whole_portion=True, entered_pot=True, cap_removed=bool(row['detached'])))
                self.added.add(name)
        group = [event for event in self.ingredients if event['tick'] == tick]
        for event in group:
            event['sample_tick'] = tick
            event['order_ambiguous'] = len(group) > 1
        # Do not credit an initially pre-seated Easy lid, or a lid before pours.
        if measured['pot_lid_supported'] and not self.lid_was_supported:
            self.lid = tick
        self.lid_was_supported = bool(measured['pot_lid_supported'])

    def retention_measurement(self):
        """Current measured containment; entry events remain immutable prefix credit."""
        portions = {}
        for name in sorted(self.added):
            positions = self.latest['food'][name] if self.latest else []
            finite = all(len(p) == 3 and all(math.isfinite(x) for x in p)
                         for p in positions)
            portions[name] = dict(piece_count=len(positions), finite=finite,
                contained=bool(len(positions) == len(self.latest['food_in_pot'][name]) == 3 and finite and
                               all(self.latest['food_in_pot'][name])))
        fresh = bool(self.latest is not None and self.latest_sample_tick is not None
                     and 0 <= self.tick - self.latest_sample_tick <= 3)
        return dict(sample_tick=self.latest_sample_tick, observed_through_tick=self.tick,
                    sample_fresh=fresh, portions=portions,
                    all_added_remain_in_pot=bool(fresh and all(p['contained'] for p in portions.values())))

    def evidence(self):
        if self.latest is None or self.hidden is None:
            # Pre-observation idle is still negative evidence; hide tick is in
            # the future and no task event can earn full-flow credit yet.
            hidden = 420
        else:
            hidden = self.hidden
        returned = []
        displacement = 0.
        for name in IDS:
            if self.latest:
                displacement = max(displacement, math.dist(self.latest['jars'][name]['body'], self.initial[name]))
        for name in self.recipe:
            window = self.windows[name]
            stable_body = bool(len(window) == 16 and window[-1][0]-window[0][0] >= 60 and all(self.speed_ok(r, 'body') for _, r in window))
            stable_cap = bool(len(window) == 16 and window[-1][0]-window[0][0] >= 60 and all(self.speed_ok(r, 'cap') for _, r in window))
            row = self.latest['jars'][name] if self.latest else {}
            returned.append(dict(ingredient_id=name,
                jar_returned=bool(row.get('in_drawer')),
                jar_released=bool(row.get('body_released')),
                jar_stable=stable_body,
                cap_in_tray=bool(row.get('in_tray') and row.get('detached')),
                cap_released=bool(row.get('cap_released')),
                cap_stable=stable_cap))
        portions=self.retention_measurement()['portions']
        events=deepcopy(self.ingredients)
        for event in events:
            event['terminal_portion_contained']=bool(portions[event['ingredient_id']]['contained'])
        return dict(schema='cp04-cooking-evidence.v2', producer='native_evaluator',
            thermal_profile_version=self.contract['thermal_profile_version'], tick_hz=120,
            recipe_hidden_tick=hidden, drawer_open_tick=self.drawer, pot_lid_release_tick=self.lid,
            ingredient_events=events,
            thermal=dict(events=deepcopy(self.thermal), response_window_start_tick=self.band,
                max_temperature_after_keep_warm_c=self.peak, history_complete=True,
                hold_confirmed_tick=self.hold, observed_through_tick=self.tick),
            terminal_measurements=deepcopy(self.latest),
            returns=returned, controls_released=bool(self.latest and self.latest['controls_released']),
            physical_flow_valid=bool(self.latest and
                self.retention_measurement()['all_added_remain_in_pot']), jar_displacement_m=displacement)
