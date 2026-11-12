"""Delivered English E2E text, independent of source-case prose and recipes."""
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def _catalog():
    return json.loads(resources.files(__package__).joinpath('config/instructions.json').read_text())['instructions']


def public_instruction(spec):
    """Resolve only public quantity parameters; never include source descriptions."""
    if spec.task_id in ('PR1', 'PR2'):
        layout = json.loads(resources.files('mind2act_eval.tasks.' + spec.task_id)
                            .joinpath('config/layout.json').read_text())
        key = 'target_ohm' if spec.task_id == 'PR1' else 'target_volt'
        return layout['instruction'].replace('{TARGET}', f'{spec.config[key]:g}')
    rows = [row for row in _catalog()[spec.task_id] if spec.difficulty in row['d']]
    if len(rows) != 1:
        raise ValueError('ambiguous delivered instruction')
    text = rows[0]['en']
    if spec.task_id in ('CP03', 'DR1'):
        text = text.replace('<N>', str({'easy': 3, 'medium': 5, 'hard': 7}[spec.difficulty]))
    elif spec.task_id == 'F1' and spec.difficulty != 'easy':
        text = text.replace('<N>', str({'medium': 3, 'hard': 4}[spec.difficulty]))
    elif spec.task_id == 'DR2':
        config = spec.config
        for value in (f"{config['press_window_s']:.1f}", f"{config['release_window_s']:g}",
                      str(len(config['sequence']))):
            text = text.replace('<N>', value, 1)
    if '<N>' in text:
        raise ValueError('unresolved public instruction parameter')
    return text
