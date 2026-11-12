"""DR3 scoring from independently replayed native physical measurements.

Rule authority: reviews/final-version-20261105/FINAL-VERSION.md. Historical GPU
qualification flags are provenance only; missing physical evidence remains
invalid and is never converted into a policy failure.
"""
from pathlib import Path

from mind2act_eval.contracts import NativeEvidence, Score, to_json

from .audit import read_manifest, read_trace
from .events import rebuild_events
from .instances import SOURCE_ID
from mind2act_eval.scoring import SCORING_VERSION

RULE_VERSION = SCORING_VERSION


def score(evidence: NativeEvidence) -> Score:
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
    result = rebuild_events(rows, manifest)
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
        "rule_version": RULE_VERSION,
        "scoring_version": SCORING_VERSION,
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
