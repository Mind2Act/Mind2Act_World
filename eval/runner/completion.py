"""Task completion interfaces from FINAL-VERSION's two termination classes."""
from mind2act_eval.scoring import EVENT_SEQUENCE_TASKS, RESULT_TASKS


def completion_kind(task_id):
    if task_id in EVENT_SEQUENCE_TASKS:
        return 'events'
    if task_id in RESULT_TASKS:
        return 'result'
    return None


def completed(scene, task_id):
    kind = completion_kind(task_id)
    if kind is None:
        raise ValueError('task has no final scoring completion class')
    name = 'events_complete' if kind == 'events' else 'is_success'
    probe = getattr(scene, name, None)
    if not callable(probe):
        raise TypeError(f'{task_id} scene must implement {name}()')
    ready = probe()
    if type(ready) is not bool:
        raise TypeError(f'{task_id} {name}() must return bool')
    return ready
