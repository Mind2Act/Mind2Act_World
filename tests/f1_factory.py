"""Synthetic F1 instance row (easy line layout): baseline shape, no real ids, hashes or paths."""
import copy


def row():
    return copy.deepcopy(dict(
        task_id='F', difficulty='easy', case_id='F/easy/synthetic_0001', split='train',
        instruction='Probe both buttons, then reach the visible goal using the fewest presses.',
        config=dict(
            budget_ticks=43202,
            protocol=dict(press_threshold_m=.0025, release_threshold_m=.0008, quality=dict(parking_error=.02)),
            native_inputs=dict(
                manifest=dict(station=dict(world_support_z=.768)),
                config=dict(difficulty='Easy', space=dict(kind='line', positions=5), buttons=['A', 'B'],
                            mapping=dict(A='right', B='left'), exploration=['B', 'A'],
                            execution=dict(start=3, target=2, route=['B']),
                            timing=dict(button_feedback_seconds=.65, hold_seconds=.8),
                            layout=dict(button_centers_xy=[[-.07, -.145], [.07, -.145]],
                                        slider_center_xy=[0, .075], slider_size_xy=[.36, .1]))))))
