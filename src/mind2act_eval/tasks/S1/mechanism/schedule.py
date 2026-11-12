"""S1 rotation schedule: cue -> rotation program -> execution (120 Hz physics ticks).

Source: S-147 baseline src/mind2act/core/timeline.py (`table_angle`, `rotation_end`, `schedule`), verbatim.
The phase names are the task schedule itself (when execution is allowed), not expert labels.
"""
import numpy as np

TRAILING_S = 1.
TIMELINE_KEYS = ('cue_seconds', 'rotation_seconds', 'stop_seconds', 'cue_gap_s')


def timeline_config(spec):
    """Apply the formal cue ruling without rewriting the original case.

    Source-bound fixed50 inputs inherit the acquisition's two-second cue.
    The 2026-11-01 ruling uses one second and no gap for formal evaluation;
    training replay keeps its explicitly supplied acquisition timeline.
    The scene records this effective timeline in evidence/result metadata.
    """
    missing = [key for key in TIMELINE_KEYS if key not in spec.config]
    if missing:
        raise ValueError('S1 spec.config must give the timeline explicitly: missing %s' % missing)
    config = {key: float(spec.config[key]) for key in TIMELINE_KEYS}
    source = spec.config.get('test_source', {})
    if source.get('instance_set') == 'test50-r5' and source.get('split') == 'test':
        config.update(cue_seconds=1., cue_gap_s=0.)
    return config


def smoothstep(u):
    u = max(0., min(1., u))
    return u*u*(3-2*u)


def table_angle(t, cue, rotation, stop, degrees, gap=0.):
    elapsed, angle = t-cue-gap, 0.
    for delta in degrees:
        u = max(0., min(1., elapsed/rotation))
        angle += float(np.radians(delta))*u*u*(3-2*u)
        elapsed -= rotation+stop
    return angle


def rotation_end(cue, rotation, stop, segments, gap=0.):
    return cue+gap+segments*(rotation+stop)+TRAILING_S


def schedule(step, config, tables, gap=0.):
    """Return (phase, cue_visible, {table: angle}) for the world tick count `step` (120 Hz)."""
    t = step/120
    cue, rotation, stop = config['cue_seconds'], config['rotation_seconds'], config['stop_seconds']
    angles = {name: table_angle(t, cue, rotation, stop, spec['rotation_degrees'], gap)
              for name, spec in tables.items()}
    end = rotation_end(cue, rotation, stop, max(len(s['rotation_degrees']) for s in tables.values()), gap)
    phase = 'execution' if t >= end else ('cue' if t < cue else 'rotation')
    return phase, t < cue, angles


