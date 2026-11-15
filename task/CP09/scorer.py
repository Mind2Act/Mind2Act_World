"""CP09 section 6: 100 on success; otherwise difficulty cap times repaired colors / colors.

The private scene produces chronological contact-raster measurements through
CP09.evidence.Progress. No policy state or reference-controller labels enter
this interface. Missing or inconsistent measurements raise ValueError so the
shared runner records invalid/null rather than fabricating a zero.
"""
from mind2act_eval.scoring import cp_score_from_completion
from collections.abc import Mapping

from mind2act_eval.contracts import NativeEvidence, Score
from . import CONTRACT

COLORS = {"easy": 1, "medium": 2, "hard": 3}
SCORING_VERSION = "mind2act-final-scoring-20261105"


def score_progress(evidence: NativeEvidence) -> Score:
    if evidence.task_id != "CP09":
        raise ValueError("CP09 evidence task mismatch")
    try:
        values = evidence.values; n = COLORS[values['difficulty']]
        progress = values['cp09_progress']
        if progress['contract'] != CONTRACT or progress['report_step_source'] != 'measured_physical_state':
            raise ValueError('CP09 measured raster contract missing')
        metrics = progress['final_surface']['per_color']
        import math
        if len(metrics) != n or {m['color'] for m in metrics} != set(range(1,n+1)):
            raise ValueError('CP09 final color coverage missing')
        units = []
        for m in metrics:
            precision,recall,extra = float(m['precision']),float(m['recall']),float(m['extra_fraction'])
            if not all(map(math.isfinite,(precision,recall,extra))) or not 0<=precision<=1 or not 0<=recall<=1 or extra<0:
                raise ValueError('CP09 invalid final surface metrics')
            units.append(precision >= .95 and recall >= .95 and extra <= .03)
        k = sum(units)
        # A target raster before the environment applies its annotation is not
        # a restored result. The measured report records presentation readiness.
        if len(progress['measured_report_steps']) != n:
            raise ValueError('CP09 color measurements mismatch')
        units = [a and b for a,b in zip(units,progress['measured_report_steps'])]
        if any(type(v) is not bool for v in progress['measured_report_steps']):
            raise ValueError('CP09 color measurements must be booleans')
        k = sum(units)
        return cp_score_from_completion(k==n,k,n,difficulty=values['difficulty'],components=dict(scoring_version=SCORING_VERSION,
            completed_colors=k,physical_evidence_contract=CONTRACT))
    except (KeyError, TypeError) as exc:
        raise ValueError('CP09 incomplete physical evidence') from exc


def score(evidence: NativeEvidence) -> Score:
    if 'audit' not in evidence.values:
        raise ValueError('CP09 complete append-only physical audit required')
    from .audit import read
    from .replay import Replay
    values=evidence.values
    rows=read(values['audit'],evidence.tick)
    from .instances import config,signature
    initial=rows[0]
    if evidence.instance_id.startswith('VAL-CP09-') or 'validation_source' in values:
        from mind2act_eval.cp09_validation_inputs import validate_evidence
    else:
        from mind2act_eval.cp09_training_inputs import validate_evidence
    validate_evidence(evidence,initial['source_case'])
    if initial['contract']!=CONTRACT:
        raise ValueError('CP09 initial audit/source binding differs')
    replay=Replay(values['difficulty'],values['seed'],initial['source_case'])
    for row in rows[1:]:replay.append(row)
    progress=replay.result()
    measured=NativeEvidence(evidence.task_id,evidence.instance_id,evidence.tick,dict(
        difficulty=values['difficulty'],cp09_progress=progress,normal_termination=progress['success']))
    return score_progress(measured)
