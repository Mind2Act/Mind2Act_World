"""CP03 configuration entry: turn a delivered native spec into a frozen InstanceSpec document.

The delivered per-case `native-spec` (B delivery `provenance/red-texture-native-spec.json`
for the YCB029 red-plate texture, `runtime/native-spec.json` is identical for the checked
case) carries the manifest M, config C and setup that the baseline scene consumed.
This module validates it, strips planner/private fields that no consumer needs, and emits
the document accepted by `InstanceSpec.from_dict` / `mind2act-eval validate`.

Assets are public ids resolved by the asset root: `station` is the scene layer
(`native_inputs.args.scene`, relative), `texture_map` the recovered YCB029 texture, and
`x5` must be supplied by the caller. `source_closure` hashes are kept so the resolver can
verify the layer and texture before a run; a missing texture is an asset gap, never replaced.
"""
from __future__ import annotations

import re

EXPECTED_UNITS = {"easy": 3, "medium": 5, "hard": 7}
SOURCE_ID = {"easy": "cp03-em-b-20261028", "medium": "cp03-em-b-20261028", "hard": "cp03-hard-b-20261028"}
# physco_precision_kinematics.HOME (degrees 0,65,100,-65,0,0) + open fingers 0.044 m, as used by the baseline scene
BASELINE_ROBOT_REST = tuple(__import__("math").radians(v) for v in (0, 65, 100, -65, 0, 0)) + (.044, .044)
CLOSURE_ROLES = {"station": ".usda", "texture_map": "texture_map.png"}


def instance_document(native_spec: dict, *, x5_asset: str, asset_version: str, config_version: str,
                      protocol_version: str = "v1", seed: int = 0, clock=None) -> dict:
    task = native_spec.get("task_id")
    level = native_spec.get("difficulty")
    if task != "CP03" or level not in EXPECTED_UNITS:
        raise ValueError("not a CP03 native spec")
    binding = native_spec["binding"]
    if binding["expected_units"] != EXPECTED_UNITS[level]:
        raise ValueError("CP03 plate denominator differs from the difficulty")
    inputs = native_spec["config"]["native_inputs"]
    manifest, config, setup, args = inputs["manifest"], inputs["config"], inputs["setup"], inputs["args"]
    if config["difficulty"] != level or config["task"] != "CP03" or config["count"] != EXPECTED_UNITS[level]:
        raise ValueError("CP03 native config disagrees with the spec")
    if level not in manifest["levels"]:
        raise ValueError("CP03 manifest lacks the requested level")
    closure = native_spec["config"]["source_closure"]
    assets = {"station": args["scene"], "x5": x5_asset}
    texture = [p for p in closure if p.endswith(CLOSURE_ROLES["texture_map"])]
    if len(texture) != 1:
        raise ValueError("CP03 requires exactly one recovered texture map in the source closure")
    assets["texture_map"] = texture[0]
    digests = {assets["station"]: closure.get(assets["station"]), assets["texture_map"]: closure[texture[0]]}
    if digests[assets["station"]] is None:
        raise ValueError("CP03 scene layer is missing from the source closure")
    doc = {
        "task_id": "CP03", "instance_id": native_spec["case_id"], "difficulty": level,
        "source_id": SOURCE_ID[level], "config_version": config_version, "asset_version": asset_version,
        "protocol_version": protocol_version, "seed": seed,
        "config": {"instruction": native_spec["instruction"],
                   "native_inputs": {"manifest": manifest, "config": config, "setup": setup},
                   "asset_sha256": digests, "case_binding": dict(binding),
                   "robot_rest": list(BASELINE_ROBOT_REST)},
        "assets": assets,
    }
    if clock is not None:
        doc["clock"] = dict(clock)
    _reject_private_paths(doc)
    return doc


def _reject_private_paths(value, where="document"):
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_private_paths(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_paths(item, f"{where}[{index}]")
    elif isinstance(value, str) and re.match(r"/(?:root|Users|home|shared)/", value):
        raise ValueError(f"private absolute path in CP03 instance at {where}")


def verify_assets(resolver, document: dict) -> None:
    """Check scene layer and texture bytes against the closure digests before building."""
    if document.get('instance_id','').startswith('VAL-CP03-') or document['config'].get('layout_split')=='validation':
        from mind2act_eval.tasks.CP03.validation_inputs import validate
        validate(document)
    for asset_id in ("station", "texture_map"):
        from mind2act_eval.sim.source_assets import source_file
        source_file(resolver, asset_id)
