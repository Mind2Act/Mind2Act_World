"""Typed private scene bindings with explicit diagnostic and held-out cohorts.

Training-layout tests require their immutable public identity and asset authority;
diagnostic training scenes never serve as an implicit test-slot fallback.
"""
import hashlib
import json
import re
from math import isfinite

from .scorer import COUNTS
from .rules import rule_for_kind

SCHEMA = "cp07.source-binding.v2"


class ProfileBlocked(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def goal_pose(slot, piece_id):
    """Source-local body frames may require a candidate-specific target pose."""
    return slot.get("candidate_goal_poses", {}).get(piece_id, slot)


def validate(spec, binding):
    validation=spec.config.get('layout_split')=='validation'
    if validation:
        from .validation import validate as validate_validation
        validate_validation(spec)
    else:
        from mind2act_eval.cp07_training_inputs import validate as finite_validate
        finite_validate(spec)
    expected = {key: getattr(spec, key) for key in
                ("task_id", "instance_id", "difficulty", "source_id", "seed", "asset_version")}
    if spec.task_id != "CP07" or binding["identity"] != expected:
        raise ValueError("CP07 source/instance/seed binding mismatch")
    if spec.clock.physics_hz != 120:
        raise ValueError("CP07 selected source requires 120 Hz physics")
    if binding["schema"] != SCHEMA:
        raise ValueError("CP07 binding schema mismatch")
    # Diagnostic selected-source loading needs no formal slots or Hard inventory.
    # The explicit mode prevents a private known subset claiming completeness.
    if validation and binding['mode']!='independent_validation':
        raise ProfileBlocked('CP07 independent validation mode differs')
    training_test=spec.config.get('cp07_mode')=='train_layout_test'
    test=spec.config.get('cp07_mode')=='source_bound_test'
    if training_test and binding['mode']!='train_layout_test':
        raise ProfileBlocked('CP07 training-layout mode differs')
    if test:
        from .original_inputs import validate_binding
        if binding['mode']!='source_bound_test':raise ProfileBlocked('CP07 original test mode differs')
        count,stock=validate_binding(spec,binding)
    elif not training_test and not validation and ('test_source' in spec.config or binding["mode"] != "diagnostic_training"
            or spec.config.get("cp07_mode") != "diagnostic_training"):
        raise ProfileBlocked("CP07 requires explicit diagnostic_training source binding")
    if not test and binding["solutions_scope"] != "known-source-solutions":
        raise ValueError("CP07 diagnostic solution scope must remain explicit")
    from .original_inputs import WORLD_SHA256
    if (binding["source_code_sha256"] != (WORLD_SHA256 if test else "fff77c74af87096814463f1be451cb21de50dded2df9fde67f3160e625425f16")
            or not re.fullmatch(r"[0-9a-f]{64}", binding["source_manifest_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", binding["scene_sha256"])):
        raise ValueError("CP07 source provenance missing")
    if not test:count, stock = COUNTS[spec.difficulty]
    pieces = binding["pieces"]
    if len(pieces) != stock or len({p["id"] for p in pieces}) != stock:
        raise ValueError("CP07 source stock count mismatch")
    by_id = {p["id"]: p for p in pieces}
    for piece in pieces:
        rule_for_kind(piece["kind"])
    paths = {p["path"] for p in pieces}
    if len(paths) != stock or any(not p.startswith("/World/Pieces/") for p in paths):
        raise ValueError("CP07 task piece paths are invalid")
    if not binding["solutions"]:
        raise ProfileBlocked("CP07 source physical solution manifest missing")
    for solution in binding["solutions"]:
        if len(solution["slots"]) != count:
            raise ValueError("CP07 ordered physical slots incomplete")
        for index, slot in enumerate(solution["slots"]):
            names = slot["compatible_piece_ids"]
            if not names or len(set(names)) != len(names) or any(n not in by_id for n in names):
                raise ValueError("CP07 incompatible solution stock")
            # Compatibility is geometry-based, not an expert's chosen ID.
            geometries = {tuple(by_id[n]["geometry_signature"]) for n in names}
            if len(geometries) != 1:
                raise ValueError("CP07 interchangeable pieces must have identical shape/size")
            if any(by_id[n]["kind"] != slot["kind"] for n in names):
                raise ValueError("CP07 compatible piece kind differs from slot")
            if slot["orientation_rule"] != rule_for_kind(slot["kind"]) or "angle_tolerance_rad" in slot:
                raise ValueError("CP07 requires the approved per-kind orientation rule")
            poses = slot.get("candidate_goal_poses")
            if poses is not None and set(poses) != set(names):
                raise ValueError("CP07 candidate-specific goal pose inventory mismatch")
            lower = slot.get("support_slots", [])
            if (any(type(i) is not int or not 0 <= i < index for i in lower)
                    or (not slot["support_paths"] and not lower)
                    or any(not p.startswith("/World/") or p.startswith(("/World/Robot", "/World/Pieces/"))
                           for p in slot["support_paths"])):
                raise ValueError("CP07 physical lower support paths missing")
            for pose in (slot, *(goal_pose(slot, name) for name in names)):
                if (len(pose["p_m"]) != 3 or len(pose["q_wxyz"]) != 4
                        or not all(isfinite(x) for x in (*pose["p_m"], *pose["q_wxyz"]))
                        or sum(x*x for x in pose["q_wxyz"]) < 1e-12):
                    raise ValueError("CP07 source slot transform invalid")
    for key in ("car_start_x_m", "roadtop_start_m", "finish_x_m", "lane_y_m", "road_width_m"):
        if not isfinite(binding[key]):
            raise ValueError("CP07 source geometry measurement invalid")
    if binding["road_width_m"] <= .038:
        raise ValueError("CP07 source lane cannot contain vehicle")
    bounds = binding["bridge_bounds_m"]
    if (len(bounds["lo"]) != 3 or len(bounds["hi"]) != 3
            or not all(isfinite(x) for x in (*bounds["lo"], *bounds["hi"]))
            or any(a >= b for a, b in zip(bounds["lo"], bounds["hi"]))
            or len(binding["button_xy_m"]) != 2 or not all(isfinite(x) for x in binding["button_xy_m"])
            or not isfinite(binding["roadtop_end_m"])):
        raise ValueError("CP07 physical scene bounds missing")
    return binding


def load(spec, assets):
    if spec.config.get('layout_split')=='validation':
        from .validation import load as load_validation
        return load_validation(spec,assets)
    from mind2act_eval.cp07_training_inputs import validate_assets
    validate_assets(spec, assets)
    training_test=spec.config.get('cp07_mode')=='train_layout_test'
    test=spec.config.get('cp07_mode')=='source_bound_test'
    if (not test and not training_test and ('test_source' in spec.config or spec.config.get("cp07_mode") != "diagnostic_training")):
        raise ProfileBlocked("CP07 requires explicit diagnostic_training source binding")
    from mind2act_eval.sim.source_assets import read_source_bytes, verify_source_file
    try:
        raw = read_source_bytes(assets, "cp07_binding", spec.config["cp07_binding_sha256"])
    except ValueError as exc:
        raise ValueError("CP07 binding SHA mismatch: " + str(exc)) from exc
    binding=json.loads(raw)
    if training_test:
        from .train_layout import authenticate
        binding=authenticate(spec,assets)
    if test:
        from mind2act_eval.original_test_inputs import audit_source
        binding['original_source']=audit_source(spec,assets)
    binding = validate(spec,binding)
    if test:
        from .original_geometry import validate_geometry
        from .original_inputs import authenticate
        validate_geometry(spec, assets, binding, authenticate(spec, binding['original_source']))
    try:
        verify_source_file(assets, "station", binding["scene_sha256"])
    except ValueError as exc:
        raise ValueError("CP07 prepared scene SHA mismatch: " + str(exc)) from exc
    return binding
