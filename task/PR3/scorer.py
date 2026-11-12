"""PR3 actual final ink coverage/RMSE; FINAL-VERSION.md."""
import numpy as np

from mind2act_eval.contracts import NativeEvidence, Score
from .mechanism.paths import reference
from mind2act_eval.scoring import SCORING_VERSION

COVERAGE_MIN = 0.98
RMSE_MAX_M = 0.002
POINT_TOLERANCE_M = 0.003
COVERAGE_FLOOR = 0.5
KEYS = frozenset(('parameters', 'paper', 'samples', 'board_contact_at_end', 'measured_ticks',
                  'direction_gate', 'normal_completion'))


def segment_distances(points, starts, ends):
    best = np.full(len(points), np.inf)
    best_index = np.zeros(len(points), dtype=int)
    for index, (start, end) in enumerate(zip(starts, ends)):
        vector = end - start
        denom = float(np.dot(vector, vector))
        if denom == 0:
            projection = np.broadcast_to(start, points.shape)
        else:
            u = np.clip(np.sum((points - start) * vector, axis=1) / denom, 0.0, 1.0)
            projection = start + u[:, None] * vector
        distance = np.linalg.norm(points - projection, axis=1)
        update = distance < best
        best[update] = distance[update]
        best_index[update] = index
    return best, best_index


def split_strokes(ticks, stroke_ids, points):
    strokes, current, previous = [], [], None
    for tick, stroke, point in zip(ticks, stroke_ids, points):
        if current and (tick != previous[0] + 1 or stroke != previous[1]):
            strokes.append(np.asarray(current))
            current = []
        current.append(point)
        previous = (tick, stroke)
    if current:
        strokes.append(np.asarray(current))
    return strokes


def resample(stroke, spacing=.0005):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(stroke, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    arc, stroke = arc[keep], stroke[keep]
    if len(arc) < 2:
        return stroke
    ticks = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1] / spacing)) + 1))
    return np.column_stack([np.interp(ticks, arc, stroke[:, j]) for j in range(3)])


def metrics(evidence: NativeEvidence) -> dict:
    v = evidence.values
    if set(v) != KEYS:
        raise ValueError('PR3 evidence needs exactly ' + ', '.join(sorted(KEYS)))
    if v['measured_ticks'] != evidence.tick:
        raise ValueError('PR3 trace does not cover every physics tick')
    s = v['samples']
    if not (len(s['tick']) == len(s['stroke']) == len(s['point'])):
        raise ValueError('inconsistent PR3 samples')
    p = v['parameters']
    xy = reference(dict(level=p['level'], mirror=p['mirror'], angle_delta_deg=p['angle_delta_deg'],
                        offset_xy_mm=list(p['offset_xy_mm']))) / 1000.0
    paper = v['paper']
    target = np.column_stack((xy[:, 0], xy[:, 1] + paper['center_y_m'], np.full(len(xy), paper['top_z_m'])))
    target_length = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    strokes = [resample(x) for x in split_strokes(s['tick'], s['stroke'], [tuple(q) for q in s['point']])]
    real_contact = bool(strokes)
    lifted = not v['board_contact_at_end']
    out = dict(real_contact_trace=real_contact, nonzero_trace=False, lifted_finish=lifted, coverage=0.,
               arclength_weighted_rmse_m=None, ordered_continuous_stroke=False, target_length_m=target_length,
               actual_contact_strokes=len(strokes))
    if not strokes:
        return out
    actual = np.concatenate(strokes)
    starts = np.concatenate([x[:-1] if len(x) > 1 else x for x in strokes])
    ends = np.concatenate([x[1:] if len(x) > 1 else x for x in strokes])
    target_to_actual, _ = segment_distances(target, starts, ends)
    actual_to_target, nearest = segment_distances(actual, target[:-1], target[1:])
    weights = np.r_[np.linalg.norm(np.diff(target, axis=0), axis=1), 0.0]
    coverage = float(np.sum(weights[target_to_actual <= POINT_TOLERANCE_M]) / np.sum(weights[:-1]))
    actual_weights = np.concatenate([np.r_[0.0, np.linalg.norm(np.diff(x, axis=0), axis=1)] for x in strokes])
    nonzero = bool(np.sum(actual_weights) > 0)
    rmse = float(np.sqrt(np.sum(actual_weights * actual_to_target ** 2) / np.sum(actual_weights))) if nonzero else None
    parameter = nearest / max(1, len(target) - 2)
    out.update(nonzero_trace=nonzero, coverage=coverage, arclength_weighted_rmse_m=rmse,
               ordered_continuous_stroke=bool(np.all(np.maximum.accumulate(parameter) - parameter <= 0.015)),
               maximum_actual_to_target_error_m=float(np.max(actual_to_target)),
               actual_length_m=float(sum(np.linalg.norm(np.diff(x, axis=0), axis=1).sum() for x in strokes)))
    return out


def score(evidence: NativeEvidence) -> Score:
    if evidence.task_id != 'PR3':
        raise ValueError('not PR3 evidence')
    m = metrics(evidence)
    v = evidence.values
    if type(v['direction_gate']) is not bool or type(v['normal_completion']) is not bool:
        raise ValueError('direction_gate/normal_completion must be boolean')
    coverage_pass = m['coverage'] >= COVERAGE_MIN
    rmse_pass = m['arclength_weighted_rmse_m'] is not None and m['arclength_weighted_rmse_m'] <= RMSE_MAX_M
    valid_trace = m['real_contact_trace'] and m['nonzero_trace']
    both = coverage_pass and rmse_pass
    goal = both and valid_trace
    value = 100. if goal else {'easy':30.,'medium':22.,'hard':15.}[v['parameters']['level']] if valid_trace and (coverage_pass or rmse_pass) else 0.
    return Score(value, dict(m, coverage_pass=coverage_pass, arclength_weighted_rmse_pass=rmse_pass,
                             task_success=goal, task_goal_satisfied=goal,
                             scoring_version=SCORING_VERSION, completion_percent=m['coverage'] * 100))
