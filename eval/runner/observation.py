"""Environment-owned observation allowance, independent of the actor budget.

Audited against the thirty task ports; see docs/runtime.md and the observation
window review. Values grant physical time, never success or threshold changes.
"""
from math import ceil, isfinite
from types import MappingProxyType

WINDOW_SECONDS = MappingProxyType({
    'CP01': 1., 'CP02': 1., 'CP03': 1., 'CP04': 1., 'CP05': 1.,
    'CP06': 3., 'CP07': 0., 'CP08': 5., 'CP09': 1.5, 'CP10': .5,
    'PR1': .5, 'PR2': 1., 'PR3': 1., 'PR4': 1., 'PR5': .5, 'PR6': .5,
    'DR1': 1., 'DR2': 1., 'DR3': 1., 'DR4': 1.,
    'M1': 1., 'M2': .9, 'S1': 1., 'S2': .5,
    'P1': 1., 'P2': 3., 'F1': .8, 'F2': .5, 'E1': 1., 'E2': 4 / 120,
})


def observation_ticks(spec):
    seconds = WINDOW_SECONDS.get(spec.task_id, 0.)
    if spec.task_id == 'F1':
        seconds = spec.config['case']['timing']['hold_seconds']
        if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
            raise ValueError('F1 observation hold must be a bounded positive duration')
    # Endpoint-based windows need N+1 fresh samples to span N intervals.
    # This also commits CP10's collected shot on the following physical tick.
    return ceil(seconds * spec.clock.physics_hz) + 1 if seconds else 0


def observation_wall_s(spec):
    """Separate infrastructure watchdog: at most one real second per extra tick.

    This is a hang guard, not a policy budget or a scored observation criterion.
    """
    ticks = observation_ticks(spec)
    return max(60., float(ticks)) if ticks else 0.


def task_deadline_with_observation(task_id, config):
    """Keep source global-clock cutoffs from freezing a pending final window.

    Online scene and offline scorer instantiate the same task machine. Actual
    cue/press/shot deadlines and experiment counts remain task-owned.
    """
    return config['max_ticks'] + ceil(WINDOW_SECONDS[task_id] * config['physics_hz']) + 2
