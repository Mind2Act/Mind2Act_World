"""DR1 T0 target support and center containment; FINAL-VERSION.md."""
import math
import re

import numpy as np

from mind2act_eval.contracts import NativeEvidence, Score, score_from_completion
from mind2act_eval.scoring import SCORING_VERSION

SETTLE_TICKS, RELEASE_TICKS, ROUND_QUIET_TICKS = 120, 60, 60
MAX_DRIFT_M, MAX_SPEED_M_S = .003, .03


def belt_continuous(ticks, speed):
    """Belt surface velocity is exactly (speed, 0, 0) in float32 on every physics tick."""
    expected = np.array([speed, 0., 0.], dtype=np.float32)
    return bool(ticks) and all(
        np.array_equal(np.asarray(row['belt_velocity'], dtype=np.float32), expected) for row in ticks)


def _stable(tail):
    """The scorer's existing drift and final-speed gates, shared with termination."""
    pos = np.asarray(tail[-1]['position'])
    drift = max(float(np.linalg.norm(np.asarray(p['position']) - pos)) for p in tail)
    return bool(drift < MAX_DRIFT_M and np.linalg.norm(tail[-1]['velocity']) < MAX_SPEED_M_S)


def parcel_state(ticks, index, tray):
    """T0 center in the authored tray cavity, with real tray support."""
    if not ticks:
        raise ValueError('missing DR1 final physical row')
    final = ticks[-1]['parcels'][index]
    pos = final['position']
    inside = bool(final['active'] and tray[0] <= pos[0] <= tray[1]
                  and tray[2] <= pos[1] <= tray[3])
    if type(final.get('tray_supported')) is not bool:
        raise ValueError('DR1 final measured tray support is required')
    supported = final['tray_supported']
    return dict(inside=inside, supported=supported, placed=inside and supported)


def round_over(ticks, n):
    """No quiet-parcel early stop: the actor retains its chance to recover."""
    return False


def score(evidence: NativeEvidence) -> Score:
    v = evidence.values
    ticks, n = v['ticks'], v['count']
    if evidence.task_id != 'DR1' or not ticks or ticks[-1]['tick'] != evidence.tick:
        raise ValueError('DR1 requires its T0 row')
    if not ticks or any(row['tick'] != i for i, row in enumerate(ticks, 1)):
        raise ValueError('DR1 requires every physics tick, first tick = 1')
    if any(len(row['parcels']) != n for row in ticks):
        raise ValueError('DR1 requires every parcel on every tick')
    states = [parcel_state(ticks, i, v['tray_bounds_m']) for i in range(n)]
    k = sum(s['placed'] for s in states)
    belt = belt_continuous(ticks, v['belt_speed_m_s'])
    level = v.get('difficulty') or {3:'easy',5:'medium',7:'hard'}.get(n)
    if level is None:
        found = set(re.split(r'[_/\-]+', evidence.instance_id)) & {'easy','medium','hard'}
        if len(found)!=1:raise ValueError('DR1 evidence requires difficulty')
        level=found.pop()
    cap = {'easy':30.,'medium':22.,'hard':15.}[level]
    base = score_from_completion(k == n, k, n, components=dict(belt_continuous=belt, parcels=states, scoring_version=SCORING_VERSION))

    return Score(100. if k==n else cap*k/n, dict(base.components, difficulty=level, partial_ceiling=cap))
