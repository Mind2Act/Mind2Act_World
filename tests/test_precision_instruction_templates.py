from types import SimpleNamespace
import pytest

from mind2act_eval.tasks._instructions import public_instruction


@pytest.mark.parametrize('difficulty', ['easy','medium','hard'])
@pytest.mark.parametrize('task,key,target,expected', [
    ('PR1', 'target_ohm', 20.5,
     'Adjust the resistance between the left terminal and wiper to 20.5 ohm, then release the slider and keep that reading for 0.5 seconds.'),
    ('PR2', 'target_volt', 140.,
     'Adjust the output voltage to 140 V, then release the knob and keep that reading for 1 second.'),
])
def test_precision_public_instruction_matches_approved_english(task,key,target,expected,difficulty):
    spec = SimpleNamespace(task_id=task, difficulty=difficulty, config={key:target})
    assert public_instruction(spec) == expected
