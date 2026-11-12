"""M1: first N actual supported station arrivals, plus terminal endpoint for full credit."""
import numpy as np

from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION

from .instances import layout, plain
from .ledger import Ledger


class Tracker:
    """Incremental native analysis (uncover, extraction, placements); shared by scorer and scene termination."""

    def __init__(self, v, lay):
        self.lay, self.start = lay, v['physical_start']
        self.baseline, self.initial_z = v['closed_baseline'], v['target_initial_z']
        self.ledger = Ledger(plain(v['submission_stations']), v['target_dimensions'], lay['ledger'])
        self.uncovery, self.extracted_at, self._held = [], None, 0
        self.n = len(v['expected_path'])
        self.hold_ticks = round(lay['extraction_hold_s'] * 120)  # diagnostics only

    def observe(self, r):
        step = r['tick']
        # Retain truthful legacy diagnostics without making them score gates.
        if self.start is not None and step >= self.start:
            seen = {event['cover_id'] for event in self.uncovery}
            for cover, baseline in self.baseline.items():
                lift = r['cover_centers'][cover][2]-baseline
                if cover not in seen and r['cover_fingers'].get(cover, False) and lift > self.lay['uncover_lift_m']:
                    self.uncovery.append(dict(step=step, cover_id=cover, lift_m=float(lift)))
            if self.extracted_at is None:
                lifted = r['target_center'][2]-self.initial_z > self.lay['extraction_lift_m'] and r.get('target_finger', False)
                self._held = self._held+1 if lifted else 0
                if self._held >= self.hold_ticks:
                    self.extracted_at = step
        self.ledger.observe(step, r['target_center'], r['target_linear_velocity'], r['target_angular_velocity'],
                            r.get('target_robot_contact'), r['station_support'],
                            self.start is not None and step >= self.start)

    @property
    def route(self):
        """The answer A: the first N submissions (the round ends at submission N)."""
        return [p['station'] for p in self.ledger.placements][:self.n]


def analyse(v, lay=None):
    lay = lay or layout()
    ticks = v['ticks']
    if not ticks or any(r['tick'] != i for i, r in enumerate(ticks, 1)):
        raise ValueError('M1 requires every physics tick, first tick = 1')
    for e in v['arrivals']:
        # A policy stop freezes the observed prefix; future arrivals have not
        # happened and cannot prove divergence. Observed arrivals stay strict.
        if e['step'] > len(ticks):
            continue
        if np.linalg.norm(np.asarray(ticks[e['step'] - 1]['target_center'][:2])
                          - np.asarray(v['station_xy'][e['target_station']])) >= lay['arrival_tolerance_m']:
            raise ValueError('environment exchange path diverged from the compiled schedule')
    if v['physical_start'] is None:
        return dict(uncovery=[], extracted_at=None, route=[], placements=[])
    tracker = Tracker(v, lay)
    for r in ticks:
        tracker.observe(r)
    return dict(uncovery=tracker.uncovery, extracted_at=tracker.extracted_at, route=tracker.route,
                placements=tracker.ledger.placements[:tracker.n])


def lcs(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
    return table[-1][-1]


def score(evidence: NativeEvidence) -> Score:
    v = plain(evidence.values)
    a = analyse(v)
    expected, answer = v['expected_path'], a['route']
    n = len(expected)
    first_ok = bool(a['uncovery'] and a['uncovery'][0]['cover_id'] == v['target_cover_id'])
    premise = bool(first_ok and a['extracted_at'] is not None)
    c = lcs(answer, expected)
    w = len(answer) - c
    m = (n - c) + w
    terminal = v['ticks'][-1]
    ledger = Ledger(v['submission_stations'], v['target_dimensions'], layout()['ledger'])
    endpoint = ledger.station_at(terminal['target_center'], terminal['station_support'])
    success = bool(answer == expected and endpoint == expected[-1])
    level = v.get('difficulty') or {4:'easy',5:'medium',6:'hard'}[n]
    cap = {'easy':30.,'medium':22.,'hard':15.}[level]
    if success:
        value = 100.
    elif m == 1:
        value = cap/2
    elif m == 2:
        value = cap/4
    else:
        value = 0.
    return Score(value, dict(scoring_version=SCORING_VERSION,
        success=success, premise=premise, correct_first_uncover=first_ok, extracted=a['extracted_at'] is not None,
        route=answer, expected_route=expected, lcs=c, wrong_submissions=w, missing_or_wrong=m, completion=c / n,
        placements=a['placements'], uncover_attempts=a['uncovery'], terminal_station=endpoint))
