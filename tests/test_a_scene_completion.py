"""A-line completion hooks use measured task success, never legacy failure/done."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mind2act_eval.tasks._precision.recorder import Recorder as PrecisionRecorder
from mind2act_eval.tasks.PR1.scene import RheostatScene
from mind2act_eval.tasks.PR2.scene import VoltageScene
from mind2act_eval.tasks.PR3.scene import TraceScene
from mind2act_eval.tasks.PR3.evidence import TipRecorder
from mind2act_eval.tasks.PR3.mechanism.paths import reference
from mind2act_eval.tasks.PR4.scene import ClothScene
from mind2act_eval.tasks.PR4.mechanism.terminal_shape import target_outline
from mind2act_eval.tasks.PR5.scene import StampScene
from mind2act_eval.tasks.PR6.scene import DominoScene
from mind2act_eval.tasks.DR1.scene import ConveyorScene
from mind2act_eval.tasks.DR2.scene import ButtonScene
from mind2act_eval.tasks.DR2.event_clock import EventClock


@pytest.mark.parametrize('cls,task,key,tolerance', [
    (RheostatScene, 'PR1', 'target_ohm', .2),
    (VoltageScene, 'PR2', 'target_volt', 1.),
])
def test_precision_success_uses_current_reading_while_held(cls, task, key, tolerance):
    current = [10.]
    scene = object.__new__(cls)
    scene.spec = NS(config={key: 10.}, difficulty='easy')
    scene._recorder = PrecisionRecorder(lambda: current[0], lambda: True, task, 'fixture', 10., 120, 121)
    assert not scene.is_success()
    scene._recorder.record(1)
    assert scene.is_success() and scene.is_complete()
    current[0] = 10. + tolerance + .001
    scene._recorder.record(2)
    assert not scene.is_success()  # No historical-success latch.


def test_trace_success_needs_both_final_ink_metrics_without_a_lift():
    params = dict(level='medium', mirror=False, angle_delta_deg=5., offset_xy_mm=[3., -7.5])
    paper = dict(top_z_m=.7745, center_y_m=.065)
    recorder = TipRecorder('PR3', 'fixture', params, paper,
        dict(valid_x_m=1., valid_y_half_m=1.), True, 120)
    scene = object.__new__(TraceScene)
    scene._recorder = recorder
    assert not scene.is_success()
    xy = reference(params) / 1000.
    for tick, (x, y) in enumerate(xy, 1):
        recorder.sample(tick, True, [x, y + paper['center_y_m'], paper['top_z_m']])
    assert scene.is_success() and scene.is_complete()
    for tick in range(len(xy)+1, len(xy)+1001):
        recorder.sample(tick, True, [.8, .8, paper['top_z_m']])
    assert not scene.is_success()  # Later bad ink can invalidate the current geometry.


def test_cloth_success_uses_current_fixed_target_outline_not_history():
    from tasks.PR4.support_fixture import support_geometry
    scene = object.__new__(ClothScene)
    scene.support_geometry = support_geometry()
    points = np.column_stack((target_outline('easy', [0., 0.], .4), np.full(4, .8)))
    scene.recorder = NS(rows=2, case=dict(level='easy', origin=dict(fold_target_center=[0., 0.], width=.4)),
                        triangles=np.array([[0, 1, 2], [0, 2, 3]]))
    scene._last_shape_points = points
    assert scene.is_success() and scene.is_complete()
    scene._last_shape_points = points + [0., 0., .02]
    scene.recorder.rows += 1
    assert not scene.is_success()  # Correct outline suspended above the board.
    scene._last_shape_points = points + [.1, 0., 0.]
    scene.recorder.rows += 1
    assert not scene.is_success()  # No best-fit translation back into the target.


def test_stamp_success_requires_actual_complete_imprint_and_return(tmp_path):
    from tasks.PR5.test_evidence import measured
    recorder = measured(tmp_path, home=False)
    scene = object.__new__(StampScene)
    scene.recorder = recorder
    assert not scene.is_success()
    recorder.record(77, [.02, -.2, .772], [1., 0., 0., 0.],
        {'Left': [0., 0., .77], 'Right': [0., 0., .77]},
        [dict(pair=['/World/Task/HolderA/Base', '/World/Task/StampA/Face'],
              impulse_ns=[0., 0., 0.], support_normal_z=1.)])
    assert scene.is_success() and scene.is_complete()
    recorder.close()


def test_domino_violation_is_not_completion_and_later_physics_is_recorded(tmp_path):
    from tasks.PR6.test_pr6_rules import case, frame, setup, contact
    from mind2act_eval.tasks.PR6.evidence import Recorder
    c = case()
    recorder = Recorder('fixture', c, trace_path=tmp_path/'domino.jsonl')
    setup(recorder)
    recorder.record(frame(c, 660, contacts=[contact('bell', 'robot')]))
    scene = object.__new__(DominoScene)
    scene.recorder = recorder
    scene.measure = lambda tick: frame(c, tick)
    assert recorder.rules.status == 'failure'
    assert not scene.is_success() and not scene.is_complete()
    scene.after_step(661)
    assert recorder.frames == 662
    assert not scene.is_success()
    recorder.close()


def test_parcel_success_requires_every_measured_center_and_target_support():
    from test_dr1_scorer import parcel
    scene = object.__new__(ConveyorScene)
    scene._bodies = [object(), object()]
    scene._tray = [-.34, .34, -.275, -.155, .768, .784]
    scene._recorder = NS(ticks=[])
    assert not scene.is_success()
    scene._recorder.ticks.append(dict(tick=1, parcels=[parcel(finger=True), parcel(x=.339)]))
    assert scene.is_success() and scene.is_complete()
    scene._recorder.ticks[-1]['parcels'][1]['tray_supported'] = False
    assert not scene.is_success()


def test_button_sequence_finishes_even_when_all_ten_responses_are_wrong():
    scene = object.__new__(ButtonScene)
    scene.clock = EventClock([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 3, 2, 1)
    assert not scene.events_complete()
    for tick in range(1, 100):
        scene.clock.advance(tick, [0., 0., 0.], [False, False, False])
        if scene.events_complete():break
    assert len(scene.clock.records) == 10 and scene.clock.valid_count == 0
    assert scene.events_complete() and scene.is_complete()
