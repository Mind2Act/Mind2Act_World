"""Synthetic DR2 instance row: baseline shape, no real ids, hashes or paths."""
import copy

WINDOWS = {'easy': 4.0, 'medium': 3.2, 'hard': 2.6}


def row(difficulty='easy'):
    return copy.deepcopy(dict(
        task_id='FR2', difficulty=difficulty, case_id='FR2_%s_synthetic' % difficulty, split='train',
        instruction='Press the lit button within %.1f seconds and release within 1 second.' % WINDOWS[difficulty],
        config=dict(source_case=dict(
            case_id='FR2_%s_synthetic' % difficulty, sequence=[0, 1, 2, 1, 2, 1, 1, 0, 2, 2],
            press_window_s=WINDOWS[difficulty], release_window_s=1.0, inter_event_gap_s=0.3,
            layout=dict(button_centers_xy=[[-.17, 0], [0, 0], [.17, 0]]), instance_sha256='0' * 64),
            native_inputs=dict(manifest=dict(station=dict(robots=[
                dict(name='Left', position=[-.3, -.45, .765], euler_deg=[0, 0, 90]),
                dict(name='Right', position=[.3, -.45, .765], euler_deg=[0, 0, 90])]))))))
