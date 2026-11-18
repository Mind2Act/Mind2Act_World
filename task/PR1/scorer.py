"""PR1 final resistance at T0; FINAL-VERSION.md, no release/hold gate."""
import re
from math import isfinite
from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION
from mind2act_eval.tasks._precision.stability import KEYS

TOLERANCES = {'easy': 0.2, 'medium': 0.15, 'hard': 0.12}
ZERO_DISTANCES = {'easy': 8.0, 'medium': 4.0, 'hard': 2.0}
CAPS = {'easy': 30.0, 'medium': 22.0, 'hard': 15.0}

def difficulty(evidence):
    level = evidence.values.get('difficulty')
    if level is None:
        found = set(re.split(r'[_/\-]+', evidence.instance_id)) & set(CAPS)
        if len(found) != 1:
            raise ValueError('precision evidence requires an explicit difficulty or a bound legacy instance ID')
        level = found.pop()
    if level not in CAPS:
        raise ValueError('unknown precision difficulty')
    return level

HOLD_SECONDS = 0.5  # Recorder termination window only; scoring reads the final T0 value.


def score(evidence: NativeEvidence) -> Score:
    if evidence.task_id != 'PR1':
        raise ValueError('not PR1 evidence')
    v = evidence.values
    if set(v) - {'difficulty'} != KEYS or not v['values']:
        raise ValueError('missing final resistance evidence')
    if v['first_tick'] + len(v['values']) - 1 != evidence.tick:
        raise ValueError('final resistance sample is not T0')
    final, target = v['values'][-1], v['target_value']
    if not all(isinstance(x, (int, float)) and isfinite(x) for x in (final, target)):
        raise ValueError('nonfinite final resistance')
    error = abs(final - target)
    level = difficulty(evidence)
    tolerance, distance, cap = TOLERANCES[level], ZERO_DISTANCES[level], CAPS[level]
    success = error <= tolerance + 1e-12
    partial = cap * max(0., 1. - error / distance)
    return Score(100. if success else partial, dict(final_value=final, target_value=target,
        absolute_error=error, tolerance=tolerance, difficulty=level, partial_ceiling=cap, zero_distance=distance, read_tick=evidence.tick,
        success=success, scoring_version=SCORING_VERSION, credited_units=int(success), total_units=1))
