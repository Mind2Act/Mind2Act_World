"""Data-only adapters for released layouts; never import a delivery's loader."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping

from mind2act_eval.contracts import freeze


def vector(value, size, *, quaternion=False):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"expected a {size}-vector")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("pose must contain finite numbers")
    result = tuple(float(x) for x in value)
    if quaternion and abs(sum(x*x for x in result) - 1) > 1e-3:
        raise ValueError("settled quaternion must be unit length; no silent normalization")
    return result


@dataclass(frozen=True)
class SettledObject:
    name: str
    asset_key: str
    position: tuple[float, ...]
    orientation: tuple[float, ...]  # wxyz
    scale: tuple[float, ...]
    physics: Mapping[str, Any]


@dataclass(frozen=True)
class Layout:
    task: str
    tier: str
    slot: str
    objects: tuple[SettledObject, ...]
    appearance: Mapping[str, Any]
    ground: str
    provenance: Mapping[str, Any]


def _objects(data, coordinates):
    # This is a source-specific declaration, not an interpretation of Table.
    if coordinates != {"settled_frame": "world", "units": "m", "quaternion": "wxyz"}:
        raise ValueError("requires reviewed world/metres/wxyz coordinate declaration")
    rigid = data.get("Rigid")
    if not isinstance(rigid, dict):
        raise ValueError("Rigid must be an explicit mapping, including when empty")
    objects = []
    allowed = {"category_idx", "clutter_idx", "default_pos", "default_ori", "settled_pos",
               "settled_ori", "scale", "physics", "relative_plane", "type", "yaml_path"}
    for category, rows in sorted(rigid.items()):
        if re.fullmatch(r"[a-zA-Z0-9_]+", category) is None or not isinstance(rows, list):
            raise ValueError("invalid clutter category")
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != allowed:
                raise ValueError("unsupported clutter fields or missing settled pose")
            if (type(row["category_idx"]) is not int or row["category_idx"] < 0
                    or type(row["clutter_idx"]) is not int or row["clutter_idx"] < 0
                    or row["type"] != "cluttered" or row["relative_plane"] != "Table"
                    or row["yaml_path"] != "Clutter/clutter.yml"):
                raise ValueError("unsupported clutter identity/frame; delivery paths are data only")
            physics = row["physics"]
            if (not isinstance(physics, dict) or physics.get("type") != "rigid"
                    or set(physics) - {"type", "mass", "friction", "static_friction", "dynamic_friction", "restitution"}):
                raise ValueError("unsupported distractor physics")
            for key in ("mass", "friction"):
                if key in physics and (type(physics[key]) not in (int, float)
                        or not math.isfinite(physics[key]) or physics[key] <= 0):
                    raise ValueError("invalid distractor physics value")
            has_static = "static_friction" in physics
            has_dynamic = "dynamic_friction" in physics
            if has_static != has_dynamic:
                raise ValueError("distractor static_friction and dynamic_friction must both be explicit")
            if "restitution" in physics and not (has_static and has_dynamic):
                raise ValueError("distractor restitution requires explicit static_friction and dynamic_friction")
            for key in ("static_friction", "dynamic_friction"):
                if key in physics and (type(physics[key]) not in (int, float)
                        or not math.isfinite(physics[key]) or physics[key] < 0):
                    raise ValueError("invalid distractor friction value")
            if "restitution" in physics and (type(physics["restitution"]) not in (int, float)
                    or not math.isfinite(physics["restitution"]) or not (0.0 <= physics["restitution"] <= 1.0)):
                raise ValueError("invalid distractor restitution value")
            scale = vector(row["scale"], 3)
            if any(x <= 0 for x in scale):
                raise ValueError("scale must be positive")
            vector(row["default_pos"], 3)
            vector(row["default_ori"], 4, quaternion=True)
            objects.append(SettledObject(f"{category}_{row['category_idx']:05d}_{i}",
                f"{category}/{row['category_idx']:05d}", vector(row["settled_pos"], 3),
                vector(row["settled_ori"], 4, quaternion=True), scale, freeze(physics)))
    return tuple(objects)


def _common(data, coordinates, *, new):
    if data.get("status") != "ok" or data.get("difficulty") not in ("easy", "medium", "hard"):
        raise ValueError("layout is not a finalized successful tier")
    task, tier, slot = data["task"], data["difficulty"], data["logical_slot"]
    rigid_digest = hashlib.sha256(json.dumps(data.get("Rigid"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if data.get("layout_sha256") != rigid_digest:
        raise ValueError("archived Rigid layout checksum mismatch")
    appearance = data["Appearance"]
    if (appearance.get("schema") != "mind2act.visual-randomization.v2"
            or appearance.get("mode") != "random" or appearance.get("task") != task
            or appearance.get("difficulty") != tier):
        raise ValueError("appearance/layout identity mismatch")
    for flag in ("table_enabled", "light_enabled"):
        if type(appearance.get(flag)) is not bool:
            raise ValueError("appearance enable flags must be explicit booleans")
    light = appearance["light"]
    if set(light) != {"type", "color_rgb", "intensity_scale"} or light["type"] not in (
            "Dome", "Distant", "Rect", "Disk", "Sphere", "Cylinder"):
        raise ValueError("unsupported delivered light")
    if any(x < 0 for x in vector(light["color_rgb"], 3)):
        raise ValueError("negative light colour")
    scale = light["intensity_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid intensity scale")
    objects = _objects(data, coordinates)
    settle = data.get("settle")
    if objects and not isinstance(settle, dict):
        raise ValueError("missing archived settling receipt")
    if objects:
        if (settle.get("any_below_table") is not False
                or settle.get("slot", slot) != slot or type(settle.get("steps")) is not int
                or settle["steps"] <= 0 or settle.get("stability", {}).get("unstable", [])):
            raise ValueError("archived settling receipt is incomplete or unstable")
        bodies = {b["name"]: b for b in settle["bodies"]}
        if len(bodies) != len(settle["bodies"]) or set(bodies) != {o.name for o in objects}:
            raise ValueError("settled body/instance coverage mismatch")
        for item in objects:
            body = bodies[item.name]
            if body.get("verdict") != "stable" or body.get("fell") is not False or body.get("below_table") is not False:
                raise ValueError("archived distractor did not settle successfully")
            position = vector(body["final_pos"], 3)
            orientation = vector(body["final_ori"], 4, quaternion=True)
            if (math.dist(position, item.position) > 1e-9
                    or math.dist(orientation, item.orientation) > 1e-9):
                raise ValueError("layout pose differs from archived final pose")
            if "mass" in body and (type(body["mass"]) not in (int, float)
                    or not math.isfinite(body["mass"]) or body["mass"] <= 0):
                raise ValueError("invalid archived distractor mass")
        # The archived run records effective mass, including the delivery's cap.
        # Preserve the raw layout physics in the source pin; never invent a default.
        objects = tuple(replace(item, physics=freeze({**item.physics, "mass": bodies[item.name]["mass"]}))
                        if "mass" in bodies[item.name] else item for item in objects)
    ground = data.get("ground") if new else appearance.get("ground")
    ground_id = ground.get("id") if isinstance(ground, dict) else ground
    if not isinstance(ground_id, str) or not ground_id:
        raise ValueError("missing explicit ground material identity")
    return Layout(task, tier, slot, objects, freeze(appearance), ground_id, freeze({
        key: data.get(key) for key in ("schema", "env_seed", "native_seed", "release_seed",
            "card_sha256", "public_config_sha256", "layout_sha256", "settle", "rejection_counts")
    }))


def legacy(data, coordinates):
    if data.get("schema") != "mind2act.random-layout.v1":
        raise ValueError("not a legacy release layout")
    return _common(data, coordinates, new=False)


def new15(data, coordinates):
    if data.get("schema") != "mind2act.new15.layout.v1":
        raise ValueError("not a new15 delivery layout")
    return _common(data, coordinates, new=True)


def read_layout(data, coordinates):
    schema = data.get("schema")
    if schema == "mind2act.random-layout.v1":
        return legacy(data, coordinates)
    if schema == "mind2act.new15.layout.v1":
        return new15(data, coordinates)
    raise ValueError("unknown layout schema; no automatic fallback")
