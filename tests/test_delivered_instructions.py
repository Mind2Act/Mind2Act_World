"""Public model payload regressions for source-bound and diagnostic inputs."""
from types import SimpleNamespace

import pytest

from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.tasks._observation import observation
from mind2act_eval.adapters.xpolicylab import public_observation


@pytest.mark.parametrize('task', ['CP01','CP02','CP03','CP04','CP05','DR1','DR2','E1','F1','M1','P1','PR3','S1'])
@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_source_prose_cannot_replace_delivered_model_text(task, difficulty):
    spec = SimpleNamespace(task_id=task, difficulty=difficulty,
        config={'instruction': '测试源说明，不能作为英文模型指令。',
                'press_window_s': {'easy': 4., 'medium': 3.2, 'hard': 2.6}[difficulty],
                'release_window_s': 1., 'sequence': [0] * 10})
    text = public_instruction(spec)
    assert '<N>' not in text and '测试' not in text
    if task == 'CP02' and difficulty == 'easy':
        assert text == ('Remember each macaron and its place on the still cake shown for three seconds. '
                        'Recreate the arrangement as the cake rotates.')
    if task == 'F1' and difficulty == 'easy':
        assert text.startswith('Probe the two unmarked buttons')
    if task == 'CP03':
        assert text.startswith('Load %d upright plates' % {'easy': 3, 'medium': 5, 'hard': 7}[difficulty])
    if task == 'DR2':
        assert 'within %.1f seconds' % spec.config['press_window_s'] in text
    assert spec.config['instruction'] == '测试源说明，不能作为英文模型指令。'


def test_formal_f1_observation_reaches_model_with_delivered_english():
    import numpy as np
    from mind2act_eval.tasks.F1.scene import SliderScene
    from mind2act_eval.contracts import ClockSpec

    source_text = '探索两个按钮，复位后用最少按压到达绿色目标。'
    spec = SimpleNamespace(task_id='F1', difficulty='easy', config={'instruction': source_text},
                           clock=ClockSpec(), protocol_version='v1')
    robot = SimpleNamespace(observe=lambda: dict(left_arm=[0.] * 6, right_arm=[0.] * 6,
        left_gripper=1., right_gripper=1., rgb={'head': np.zeros((2, 2, 3), dtype=np.uint8)}))
    scene = SliderScene(spec, SimpleNamespace(robot=robot), {})
    payload = public_observation(scene.observe(0), {'head': 'cam_high'})
    assert payload['task'] == ('Probe the two unmarked buttons and remember how each moves the slider. '
        'After the slider resets and the goal appears, reach the visible goal using the fewest presses.')
    assert spec.config['instruction'] == source_text
