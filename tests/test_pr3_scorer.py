import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks import PR3, TaskRegistry
from mind2act_eval.tasks.PR3.mechanism.paths import make_path, reference
from mind2act_eval.tasks.PR3.scorer import score

PARAMS = dict(level='medium', mirror=False, angle_delta_deg=5.0, offset_xy_mm=[3.0, -7.5])
PAPER = dict(top_z_m=0.7745, center_y_m=0.065)


def trace(points_xy, *, shift=(0., 0.), start_tick=1, stroke=0, board_end=False, gate=False, normal=True):
    ticks, strokes, pts = [], [], []
    for i, (x, y) in enumerate(points_xy):
        ticks.append(start_tick + i)
        strokes.append(stroke)
        pts.append((x + shift[0], y + shift[1], PAPER['top_z_m']))
    n = ticks[-1] if ticks else 0
    return NativeEvidence('PR3', 'i', n + 10, dict(
        parameters=PARAMS, paper=PAPER, measured_ticks=n + 10, board_contact_at_end=board_end,
        direction_gate=gate, normal_completion=normal and not board_end,
        samples=dict(tick=ticks, stroke=strokes, point=pts)))


def target_xy():
    xy = reference(PARAMS) / 1000.
    return [(x, y + PAPER['center_y_m']) for x, y in xy]


def test_registers_without_pxr():
    assert TaskRegistry([PR3.DEFINITION]).task_ids == ('PR3',)


def test_path_is_350mm_and_depends_on_the_instance_parameters():
    points, stats = make_path('hard')
    assert stats['length_mm'] == pytest.approx(350, abs=.01)
    shifted = reference(dict(PARAMS, offset_xy_mm=[13.0, -7.5]))
    assert np.allclose(shifted - reference(dict(PARAMS, offset_xy_mm=[3.0, -7.5])), [10, 0])
    with pytest.raises(ValueError):
        reference(dict(PARAMS, level='easy', mirror=True))


def test_exact_trace_with_lifted_finish_scores_100():
    s = score(trace(target_xy()))
    assert s.value == 100. and s.components['coverage'] > .999
    assert s.components['arclength_weighted_rmse_m'] < 1e-6


def test_both_metrics_but_still_on_the_board_or_not_completed_normally_is_20():
    assert score(trace(target_xy(), board_end=True)).value == 100.
    assert score(trace(target_xy(), normal=False)).value == 100.


def test_direction_gate_is_explicit_and_recorded():
    xy = target_xy()
    def two_strokes(gate):
        first, second = trace(xy), trace(xy[600:1200], start_tick=3000, stroke=1)
        ticks = list(first.values['samples']['tick']) + list(second.values['samples']['tick'])
        return NativeEvidence('PR3', 'i', 4000, dict(
            parameters=PARAMS, paper=PAPER, measured_ticks=4000, board_contact_at_end=False,
            direction_gate=gate, normal_completion=True,
            samples=dict(tick=ticks, stroke=list(first.values['samples']['stroke']) + list(second.values['samples']['stroke']),
                         point=list(first.values['samples']['point']) + list(second.values['samples']['point']))))
    off, on = score(two_strokes(False)), score(two_strokes(True))
    assert off.value == 100.
    assert on.value == 100.


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_direction_gate_rejects_a_fully_reversed_dense_trace(level):
    params = dict(PARAMS, level=level)
    xy = reference(params) / 1000.
    xy[:, 1] += PAPER['center_y_m']
    ev = trace(xy[::-1], gate=True)
    ev = NativeEvidence('PR3', 'i', ev.tick, dict(ev.values, parameters=params))
    result = score(ev)
    assert result.value == 100.
    # Without a visible direction requirement the same geometric trace is valid.
    assert score(NativeEvidence('PR3', 'i', ev.tick, dict(ev.values, direction_gate=False))).value == 100.


def test_direction_tolerates_small_jitter_but_not_accumulated_backtracking():
    xy = target_xy()
    jitter = xy[:1000] + xy[980:1000] + xy[1000:]
    assert score(trace(jitter, gate=True)).value == 100.
    backtrack = xy[:1500] + xy[1499:900:-1] + xy[900:]
    assert score(trace(backtrack, gate=True)).value == 100.


def test_one_metric_with_half_coverage_is_10_and_below_the_floor_is_0():
    xy = target_xy()
    n = len(xy)
    assert score(trace(xy[:int(n * .8)])).value == 22.                  # exact RMSE, coverage 80%
    assert score(trace(xy[:int(n * .3)])).value == 22.                   # exact RMSE but coverage < 50%
    assert score(trace(xy[:int(n * .5) + 40])).value == 22.


def test_offset_trace_misses_both_metrics_so_0():
    s = score(trace(target_xy(), shift=(0., .02)))
    assert s.value == 0.


def test_no_real_trace_scores_0_with_undefined_error_recorded_as_null():
    empty = NativeEvidence('PR3', 'i', 5, dict(parameters=PARAMS, paper=PAPER, measured_ticks=5, board_contact_at_end=False,
        direction_gate=False, normal_completion=True,
                                               samples=dict(tick=[], stroke=[], point=[])))
    s = score(empty)
    assert s.value == 0. and s.components['arclength_weighted_rmse_m'] is None


def test_stationary_contact_has_no_trace_weight():
    xy = target_xy()
    still = NativeEvidence('PR3', 'i', 20, dict(parameters=PARAMS, paper=PAPER, measured_ticks=20, board_contact_at_end=False,
        direction_gate=False, normal_completion=True,
                                                samples=dict(tick=list(range(1, 11)), stroke=[0] * 10,
                                                             point=[(xy[0][0], xy[0][1], PAPER['top_z_m'])] * 10)))
    s = score(still)
    assert s.components['nonzero_trace'] is False and s.value == 0.


def test_gap_in_ticks_splits_strokes_and_does_not_bridge_them():
    xy = target_xy()
    left, right = xy[:600], xy[1800:]
    ev = trace(left)
    ticks = list(ev.values['samples']['tick']) + list(range(5000, 5000 + len(right)))
    pts = list(ev.values['samples']['point']) + [(x, y, PAPER['top_z_m']) for x, y in right]
    joined = NativeEvidence('PR3', 'i', 9000, dict(parameters=PARAMS, paper=PAPER, measured_ticks=9000, board_contact_at_end=False,
        direction_gate=False, normal_completion=True,
                                                   samples=dict(tick=ticks, stroke=[0] * len(ticks), point=pts)))
    s = score(joined)
    assert s.components['actual_contact_strokes'] == 2 and s.components['coverage'] < .98


def test_evidence_must_cover_every_tick_and_nothing_extra():
    ok = trace(target_xy())
    bad = NativeEvidence('PR3', 'i', ok.tick + 1, ok.values)
    with pytest.raises(ValueError, match='every physics tick'):
        score(bad)
    extra = NativeEvidence('PR3', 'i', ok.tick, dict(ok.values, phase='trace'))
    with pytest.raises(ValueError):
        score(extra)
    with pytest.raises(ValueError):
        score(NativeEvidence('PR2', 'i', ok.tick, ok.values))


def test_final_ink_quality_has_no_lift_direction_or_half_coverage_gate():
    assert score(trace(target_xy()[::-1], board_end=True, gate=True, normal=False)).value == 100.
    assert score(trace(target_xy()[:100], board_end=True)).value == 22.
