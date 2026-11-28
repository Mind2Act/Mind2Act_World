"""Resolve a private, exact training-scene diagnostic binding; never fill slots.

Current sources register training conditions. Formal clean/agentic/random
profiles are not delivered, including the reported missing Hard five slots.
"""
import hashlib
import json
from collections import Counter
from math import cos, isfinite, radians, sin

from .mechanism import angle_error_deg
from .scorer import PAIRS


class ProfileBlocked(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(spec, binding):
    if spec.task_id != "M2" or binding["identity"] != {key: getattr(spec, key) for key in
        ("task_id", "instance_id", "difficulty", "source_id", "seed", "asset_version")}:
        raise ValueError("M2 source/instance/seed binding mismatch")
    if spec.clock.physics_hz != 120:
        raise ValueError("M2 source mechanism requires 120 Hz physics")
    if binding["schema"] != "m2.source-binding.v1":
        raise ValueError("M2 binding schema mismatch")
    if 'test_source' in spec.config and spec.config.get('m2_mode')!='source_bound_test':
        raise ProfileBlocked('M2 original test identity requires source_bound_test mode')
    if spec.config.get('m2_mode')=='source_bound_test':
        from mind2act_eval.original_test_inputs import read_audit, source_identity
        if binding['mode']!='source_bound_test':
            raise ProfileBlocked('M2 original test mode differs')
        source=read_audit(spec,binding['original_source'])
        dimensions={'easy':(2,2),'medium':(2,4),'hard':(4,4)}
        layout=source['layout']
        if (source['task']!='M2' or source['native_task']!='MEMORY_CARDS'
                or source['difficulty']!=spec.difficulty or source['seed']!=spec.seed
                or source['task_definition_version']!='memory-bridge-final'
                or (layout['rows'],layout['cols'])!=dimensions[spec.difficulty]
                or binding['source_manifest_sha256']!=source_identity(spec)['source_configuration_sha256']
                or binding['card_identities']!=layout['card_types']
                or binding['source_layout']!=layout
                or binding['source_board_offset_xy']!=source['board_offset_xy']):
            raise ValueError('M2 original test deck/layout differs')
        target = [cos(radians(2)/2), 0., sin(radians(2)/2), 0.]
        count = layout['rows']*layout['cols']
        if (binding['button_rest_m'] != [.8055]*count
                or binding['faceup_quaternions_wxyz'] != [target]*count):
            raise ValueError('M2 original physical button rest or face-up target differs')
    elif spec.config.get('m2_mode') == 'source_bound_validation':
        if binding['mode'] != 'source_bound_validation':
            raise ValueError('M2 validation input mode differs')
    elif spec.config.get('m2_mode') == 'source_bound_mini':
        if spec.config.get('layout_split') != 'mini' or binding['mode'] not in ('source_bound_mini', 'source_bound_test'):
            raise ValueError('M2 mini input mode differs')
        dimensions = {'easy': (2, 2), 'medium': (2, 4), 'hard': (4, 4)}
        layout = binding['source_layout']
        if ((layout['rows'], layout['cols']) != dimensions[spec.difficulty]
                or binding['card_identities'] != layout['card_types']):
            raise ValueError('M2 mini deck/layout differs')
        target = [cos(radians(2)/2), 0., sin(radians(2)/2), 0.]
        count = layout['rows'] * layout['cols']
        if (binding['button_rest_m'] != [.8055] * count
                or binding['faceup_quaternions_wxyz'] != [target] * count):
            raise ValueError('M2 mini physical button rest or face-up target differs')
    elif binding["mode"] != "diagnostic_training" or spec.config.get("m2_mode") != "diagnostic_training":
        raise ProfileBlocked("M2 formal evaluation profiles/slots unavailable; no training substitution")
    n = 2 * PAIRS[spec.difficulty]
    identities = binding["card_identities"]
    if len(identities) != n or len(Counter(identities)) != n // 2 or set(Counter(identities).values()) != {2}:
        raise ValueError("M2 private deck must contain exactly two of every picture")
    if len(binding["faceup_quaternions_wxyz"]) != n:
        raise ValueError("M2 native target card poses missing")
    if (len(binding["button_rest_m"]) != n or any(type(x) not in (int, float) or not isfinite(x)
            for x in binding["button_rest_m"])):
        raise ValueError("M2 authored source button zero positions missing")
    for q in binding["faceup_quaternions_wxyz"]:
        angle_error_deg(q, q)
    return binding




def load(spec, assets):
    from mind2act_eval.sim.source_assets import asset_bytes, source_file
    if 'test_source' in spec.config and spec.config.get('m2_mode')!='source_bound_test':
        raise ProfileBlocked('M2 original test identity requires source_bound_test mode')
    if spec.config.get('m2_mode')=='source_bound_test':
        from mind2act_eval.original_test_inputs import audit_source
        raw=asset_bytes(assets, 'm2_binding')
        binding=json.loads(raw)
        binding['original_source']=audit_source(spec,assets)
        validate(spec,binding)
        from mind2act_eval.original_test_inputs import read_audit
        from .original_geometry import validate_geometry
        validate_geometry(spec, assets, binding, read_audit(spec, binding['original_source']))
        source_file(assets, 'station')
        return binding
    if spec.config.get('m2_mode') == 'source_bound_validation':
        if spec.config.get('layout_split') != 'validation':
            raise ValueError('M2 validation input requires the validation split')
        raw = asset_bytes(assets, 'm2_binding')
        binding = validate(spec, json.loads(raw))
        condition = asset_bytes(assets, 'm2_validation_condition')
        source = json.loads(condition)
        if (source['instance_id'] != spec.instance_id or source['difficulty'] != spec.difficulty
                or source['seed'] != spec.seed or source['split'] != 'validation'
                or source['profile'] != 'clean' or source['task'] != 'M2'
                or source['native_task'] != 'MEMORY_CARDS'
                or source['layout']['card_types'] != binding['card_identities']):
            raise ValueError('M2 finite VAL source condition differs')
        from .validation_geometry import validate_geometry
        validate_geometry(spec, assets, binding, source)
        source_file(assets, 'station')
        return binding
    if spec.config.get('m2_mode') == 'source_bound_mini':
        if spec.config.get('layout_split') != 'mini':
            raise ValueError('M2 mini input requires the mini split')
        from mind2act_eval.mini_layouts import validate as validate_mini
        validate_mini(spec)
        raw = asset_bytes(assets, 'm2_binding')
        if hashlib.sha256(raw).hexdigest() != spec.config.get('m2_binding_sha256'):
            raise ValueError('M2 mini binding sha256 differs')
        binding = validate(spec, dict(json.loads(raw), mode='source_bound_mini'))
        from .original_geometry import validate_geometry
        validate_geometry(
            spec,
            assets,
            binding,
            {'layout': binding['source_layout'], 'board_offset_xy': binding['source_board_offset_xy']},
        )
        source_file(assets, 'station')
        return binding
    if spec.config.get("m2_mode") != "diagnostic_training":
        raise ProfileBlocked("M2 formal evaluation profiles/slots unavailable; no training substitution")
    if "m2_source_spec" in spec.assets:
        raw = asset_bytes(assets, "m2_source_spec")
        digest = spec.config.get("m2_source_spec_sha256")
        binding = from_source_spec(spec, json.loads(raw), digest, spec.config["m2_scene_sha256"])
        source_file(assets, "station")
        return binding
    raw = asset_bytes(assets, "m2_binding")
    binding = validate(spec, json.loads(raw))
    source_file(assets, "station")
    return binding


def from_source_spec(spec, source, source_sha256, scene_sha256):
    """Convert the inspected source JSON format, without importing its runtime.

    Exact case, seed, condition digest and declared source version are required;
    original absolute paths and all source controller/configuration hooks are
    ignored. The existing CLI resolves/relocates the explicitly pinned station.
    This conversion does not claim scene/H5 equality or formal eval eligibility.
    """
    old = source["binding"]
    cfg = source["config"]
    if (source["task_id"] != "MEMORY_CARDS" or source["split"] != "train"
            or source["case_id"] != spec.instance_id or source["difficulty"] != spec.difficulty
            or cfg["seed"] != spec.seed or old["declared_version"] != spec.source_id
            or old["declared_version"] != "M2-frozen-source-sdk-v1"
            or old["task_definition_version"] != "memory-bridge-final"
            or old["config_sha256"] != spec.config["m2_condition_sha256"]):
        raise ValueError("M2 original source case/seed/condition/version mismatch")
    m = cfg["native_inputs"]["manifest"]["mechanism"]
    dimensions = {"easy": (2, 2), "medium": (2, 4), "hard": (4, 4)}
    if (m["rows"], m["cols"]) != dimensions[spec.difficulty]:
        raise ValueError("M2 original source board shape mismatch")
    n = m["rows"] * m["cols"]
    if set(m["button_rest"]) != {str(i) for i in range(n)}:
        raise ValueError("M2 original authored button positions missing")
    target = [cos(radians(2)/2), 0., sin(radians(2)/2), 0.]
    return validate(spec, dict(schema="m2.source-binding.v1", mode="diagnostic_training",
        identity={key: getattr(spec, key) for key in
            ("task_id", "instance_id", "difficulty", "source_id", "seed", "asset_version")},
        source_manifest_sha256=source_sha256, scene_sha256=scene_sha256,
        mechanism_sha256="2d5a516d615e32beafb1fa839b62dd2ff722fae0fe5b54cf2747ced782dc8e04",
        card_identities=m["card_types"], button_rest_m=[m["button_rest"][str(i)] for i in range(n)],
        faceup_quaternions_wxyz=[target[:] for _ in range(n)]))
