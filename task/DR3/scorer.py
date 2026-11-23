"""DR3 scoring from independently replayed native physical measurements.

The rule follows the DR3 scoring specification. Historical GPU
qualification flags are provenance only; missing physical evidence remains
invalid and is never converted into a policy failure.
"""
from pathlib import Path

from mind2act_eval.contracts import NativeEvidence, Score, to_json

from .audit import read_manifest, read_trace
from .events import rebuild_events
from .instances import SOURCE_ID
from mind2act_eval.scoring import SCORING_VERSION

LEGACY_RULE_VERSION = SCORING_VERSION
A_RULE_VERSION = 'mind2act-dr3-scoring-20261117-open-threshold-free-v1'


def score_A(evidence: NativeEvidence, *, rule_version=A_RULE_VERSION) -> Score:
    if rule_version not in (A_RULE_VERSION, LEGACY_RULE_VERSION):
        raise ValueError('unsupported DR3 scoring rule version')
    if evidence.task_id != "DR3" or type(evidence.tick) is not int or evidence.tick < 0:
        raise ValueError("DR3 scorer requires DR3 evidence")
    values = to_json(evidence.values)
    if (set(values) != {"difficulty", "source_id", "condition", "manifest", "raw_trace"}
            or values["source_id"] != SOURCE_ID
            or not isinstance(values["condition"], dict)
            or values["condition"].get("instance_id") != evidence.instance_id):
        raise ValueError("DR3 evidence schema/source condition mismatch")
    manifest_receipt = values["manifest"]
    trace_receipt = values["raw_trace"]
    manifest_path = Path(manifest_receipt["path"])
    trace_path = Path(trace_receipt["path"])
    if (manifest_path.name != "manifest.json" or trace_path.name != "raw-trace.jsonl"
            or manifest_path.parent != trace_path.parent):
        raise ValueError("DR3 manifest and raw trace must share the task audit directory")
    manifest = read_manifest(manifest_receipt)
    rows = read_trace(trace_receipt, evidence.tick)
    result = rebuild_events(rows, manifest, require_fully_open=rule_version == LEGACY_RULE_VERSION)
    if not result["evidence_complete"]:
        raise ValueError("DR3 physical evidence is incomplete: " + "; ".join(result["unknown_reasons"]))
    value = result["score"]
    if value not in (0.0, 5.0, 100.0):
        raise ValueError("DR3 replay produced an unsupported score")
    if result["success"] is not (value == 100.0):
        raise ValueError("DR3 success and terminal score disagree")
    if value == 5.:
        value = {'easy':7.5,'medium':5.5,'hard':3.75}[values['difficulty']]
    fields = ('violation_latched', 'passed_aperture', 'passed_aperture_count',
              'delivered', 'terminal_tray_support', 'center_inside_tray', 'cutoff_simulation_time_s')
    return Score(value, {
        "task_success": result["success"],
        "rule_version": rule_version,
        "scoring_version": rule_version,
        "event_detector_contract": result["event_detector_contract"],
        "measurement_contract": result["measurement_contract"],
        "intra_step_continuous_trajectory_claimed": False,
        "events": result["events"],
        "attempts": result["attempts"],
        "windows": result["windows"],
        "evidence": {key: result[key] for key in fields},
        "unknown_reasons": result["unknown_reasons"],
        "zero_proof": result["zero_proof"],
    })


def score_legacy(evidence: NativeEvidence) -> Score:
    """Recompute the historical rule, including near-full door travel."""
    return score_A(evidence, rule_version=LEGACY_RULE_VERSION)


import json
from importlib import resources
from .flight_kernel import from_rows as _flight_from_rows
from .flight_policy import apply_value as _flight_value, RULE_VERSIONS as A2_RULE_VERSIONS
from .withdrawal_audit import Audit as _WithdrawalAudit, apply_A3 as _withdrawal_value
A2_RULE_VERSION = A2_RULE_VERSIONS[.75]
RULE_VERSION = 'mind2act-dr3-scoring-a3-candidate-both-corridors-before-close-v1'


def _A2_measurement(evidence, peak_threshold=.75):
    baseline = score_A(evidence)
    values = to_json(evidence.values)
    manifest = read_manifest(values['manifest'])
    rows = read_trace(values['raw_trace'], evidence.tick)
    flight = _flight_from_rows(rows, manifest)
    delta = _flight_value(baseline.value, values['difficulty'], flight, peak_threshold)
    components = to_json(baseline.components)
    physical = dict(components['evidence'])
    physical['delivered'] = bool(physical['delivered'] and not delta['excluded_delivery'])
    components.update(task_success=delta['value'] == 100., rule_version=delta['rule_version'],
        scoring_version=delta['rule_version'], evidence=physical, baseline_A_value=baseline.value,
        baseline_A_rule_version=A_RULE_VERSION, a2_flight=dict(flight, **delta))
    return Score(delta['value'], components), values, manifest, rows


def score_A2(evidence):
    """The independent flight exclusion, retaining all other A decisions."""
    return _A2_measurement(evidence)[0]


def withdrawal_audit(manifest, condition, difficulty):
    # Exact source-case identity selects an extracted original timing schedule;
    # no policy/expert controller, guessed phase or observed-door velocity is used.
    registry = json.loads(resources.files(__package__).joinpath('config/withdrawal_timing.json').read_text())
    source = registry.get(condition['source_case_sha256'])
    if source is None:
        raise ValueError('DR3 original gate schedule is unavailable')
    phase = condition['phase_ticks']
    if type(phase) is not int or phase < 0:
        raise ValueError('DR3 gate phase must be explicit nonnegative ticks')
    return _WithdrawalAudit(manifest, source, difficulty, phase)


def score(evidence, *, rule_version=RULE_VERSION):
    if rule_version == LEGACY_RULE_VERSION:
        return score_legacy(evidence)
    if rule_version == A_RULE_VERSION:
        return score_A(evidence)
    if rule_version in A2_RULE_VERSIONS.values():
        peak = next(k for k, version in A2_RULE_VERSIONS.items() if version == rule_version)
        return _A2_measurement(evidence, peak)[0]
    if rule_version != RULE_VERSION:
        raise ValueError('unsupported DR3 scoring rule version')
    baseline, values, manifest, rows = _A2_measurement(evidence)
    audit = withdrawal_audit(manifest, values['condition'], values['difficulty'])
    for row in rows:
        audit.feed(row)
    diagnostics = audit.result()
    delta = _withdrawal_value(baseline.value, diagnostics)
    if delta['value'] is None:
        raise ValueError('DR3 actual both-arm corridor geometry is insufficient')
    components = to_json(baseline.components)
    physical = dict(components['evidence'])
    physical['passed_aperture_A2'] = physical['passed_aperture']
    failed = delta['withdrawal_failure'] or delta['closing_contact_failure']
    physical['passed_aperture'] = bool(physical['passed_aperture'] and not failed)
    physical['violation_latched'] = bool(physical['violation_latched'] or failed)
    components.update(task_success=delta['value'] == 100., rule_version=RULE_VERSION,
        scoring_version=RULE_VERSION, evidence=physical, baseline_A2_value=baseline.value,
        a3=dict(delta, audit=diagnostics, targets_reconstructed=True,
                corridor_margin_m=.01, default_impulse_ns=.001))
    return Score(delta['value'], components)
