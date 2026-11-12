"""CPU admission and measurements for explicit, unscored native scene probes.

Delivery JSON is data. No delivery path, module, physics override, or command is
ever executed. Asset IDs and stage targets come from a separately pinned catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from mind2act_eval.contracts import to_json

PROJECTION = "random-v2-finalized-native-baseline-v2"
SOURCE_CLEAN_PREVIEW = "source-clean-preview-v1"
SOURCE_CLEAN_TASKS = frozenset(("CP01", "CP02", "CP03", "CP04", "CP05", "DR1", "DR2", "E1", "F1", "M1", "P1", "PR1", "PR2", "PR3", "S1"))
from mind2act_eval.tasks.catalog import TASK_MODULES
CLEAN_PREVIEW_TASKS = frozenset(TASK_MODULES) - {'demo'}
CLUTTER_SOURCE_SHA256 = "70a086cc9cbc0ae0392a7502ac0619003bdc4d6096b5ac4e0dc336accd25ac51"
APPEARANCE_SOURCE_SHA256 = "6d81f46d30cca3b046f111e531856fb32e28ac11743e2beec7eb87ca8d746ca0"
FLOOR_SOURCE_SHA256 = "c7d85e2fbd44f3040a5fcd14fb3dca2e1a5867a538d860fb8dd2226eacaa9594"
RECEIPT_SCHEMA = "mind2act_eval.scene_probe_receipt.v1"
TASKS = {"F2", "PR6", "S2", "E2", "M2", "PR4"}
ROOT = "/World/SceneProbe"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spec_digest(spec):
    return hashlib.sha256(json.dumps(to_json(spec), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA256")
    return value


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite JSON number")
    return value


def _vector(value, size, *, quaternion=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("invalid pose/vector dimension")
    values = tuple(_finite(v) for v in value)
    if quaternion and abs(sum(v*v for v in values) ** .5 - 1) > .001:
        raise ValueError("orientation must be a unit wxyz quaternion")
    return values


def read_pinned(path, expected):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("probe JSON must be a bounded regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise ValueError("probe input SHA256 mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    def finite_float(value):
        return _finite(float(value))
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("invalid clutter identifier")
    return value


def _target(value):
    if (not isinstance(value, str) or not value.startswith("/World/") or "//" in value
            or any(p in (".", "..") for p in value.split("/"))
            or any("robot" in p.lower() for p in value.split("/"))
            or not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in value.split("/")[1:])):
        raise ValueError("invalid/protected probe stage target")
    return value


@dataclass(frozen=True)
class Clutter:
    name: str
    asset_path: str
    asset_sha256: str
    position: tuple
    orientation: tuple
    scale: tuple
    mass_kg: float


@dataclass(frozen=True)
class ProbeInputs:
    config: dict
    layout: dict | None
    catalog: dict | None
    clutter: tuple[Clutter, ...]
    config_sha256: str

    @property
    def is_source_clean(self):
        return self.config["projection"] == SOURCE_CLEAN_PREVIEW

    @classmethod
    def read(cls, path, expected, spec):
        config = read_pinned(path, expected)
        if isinstance(config, dict) and config.get("projection") == SOURCE_CLEAN_PREVIEW:
            fields = {"schema", "projection", "task_id", "difficulty", "source_spec_sha256"}
            if (set(config) != fields or config["schema"] != "mind2act_eval.scene_probe.v1"
                    or spec.task_id not in CLEAN_PREVIEW_TASKS or spec.variation.mode != "clean"
                    or config["task_id"] != spec.task_id or config["difficulty"] != spec.difficulty
                    or spec.clock.physics_hz != 120 or config["source_spec_sha256"] != spec_digest(spec)):
                raise ValueError("clean preview must bind only the unchanged original clean source spec at 120Hz")
            return cls(config, None, None, (), expected)
        fields = {"schema", "projection", "task_id", "difficulty", "logical_slot",
                  "source_spec_sha256", "layout", "layout_sha256", "catalog", "catalog_sha256",
                  "generation_seed", "seed_provenance"}
        if not isinstance(config, dict) or set(config) != fields:
            raise ValueError("invalid scene probe configuration fields")
        if (config["schema"] != "mind2act_eval.scene_probe.v1" or config["projection"] != PROJECTION
                or spec.task_id not in TASKS or config["task_id"] != spec.task_id
                or config["difficulty"] != spec.difficulty or spec.clock.physics_hz != 120
                or config["source_spec_sha256"] != spec_digest(spec)):
            raise ValueError("scene probe must bind the unchanged native source spec at 120Hz")
        slot = config["logical_slot"]
        if not re.fullmatch(re.escape(spec.task_id + "_" + spec.difficulty) + r"_clean_\d{3}", slot):
            raise ValueError("invalid explicit diagnostic layout slot")
        if config["generation_seed"] is not None and type(config["generation_seed"]) not in (str, int):
            raise ValueError("invalid generation seed")
        if not isinstance(config["seed_provenance"], str) or not config["seed_provenance"]:
            raise ValueError("record seed provenance, including an explicit unknown")
        layout = read_pinned(config["layout"], config["layout_sha256"])
        catalog = read_pinned(config["catalog"], config["catalog_sha256"])
        validate_catalog(catalog, spec)
        allowed = {"schema", "task", "difficulty", "logical_slot", "status", "Appearance", "Rigid",
                   "ground", "settle", "card_sha256", "layout_sha256", "public_config_sha256", "rejection_counts"}
        if (not isinstance(layout, dict) or set(layout) - allowed or layout.get("schema") != "mind2act.new15.layout.v1"
                or layout.get("task") != spec.task_id or layout.get("difficulty") != spec.difficulty
                or layout.get("logical_slot") != slot or layout.get("status") != "ok"):
            raise ValueError("layout identity/schema/whitelist mismatch")
        clutter = validate_clutter(layout.get("Rigid"), catalog)
        appearance = layout.get("Appearance")
        appearance_fields = {"schema", "task", "difficulty", "mode", "table_enabled", "light_enabled",
                             "material", "light", "catalog_sha256", "profile_sha256"}
        if (not isinstance(appearance, dict) or set(appearance) != appearance_fields
                or appearance["task"] != spec.task_id or appearance["difficulty"] != spec.difficulty
                or appearance["schema"] != "mind2act.visual-randomization.v2" or appearance["mode"] != "random"
                or appearance["table_enabled"] is not True or appearance["light_enabled"] is not True):
            raise ValueError("unsupported appearance projection")
        for row in (appearance["material"], layout.get("ground")):
            if not isinstance(row, dict) or row.get("id") not in catalog["materials"]:
                raise ValueError("material is absent from the trusted catalog")
            pin = catalog["materials"][row["id"]]
            if row.get("sha256") != pin["sha256"] or row.get("mdl_name") != pin["mdl_name"]:
                raise ValueError("material identity mismatch")
        light = appearance["light"]
        if (not isinstance(light, dict) or set(light) != {"type", "color_rgb", "intensity_scale"}
                or light["type"] not in {"Rect", "Disk", "Sphere", "Cylinder", "Dome", "Distant"}
                or _finite(light["intensity_scale"]) <= 0):
            raise ValueError("unsupported light settings")
        _vector(light["color_rgb"], 3)
        return cls(config, layout, catalog, tuple(clutter), expected)


def validate_catalog(catalog, spec):
    fields = {"schema", "source_spec_sha256", "asset_root", "clutter", "materials", "table_targets", "floor_targets",
              "key_light", "dome_light", "floor_contact_targets", "support_z_m", "source_provenance"}
    if (not isinstance(catalog, dict) or set(catalog) != fields
            or catalog["schema"] != "mind2act_eval.scene_probe_catalog.v2"
            or catalog["source_spec_sha256"] != spec_digest(spec)
            or not isinstance(catalog["source_provenance"], dict) or not catalog["source_provenance"]):
        raise ValueError("invalid trusted probe catalog")
    if not isinstance(catalog["asset_root"], str) or not Path(catalog["asset_root"]).is_absolute():
        raise ValueError("trusted random assets need an explicit absolute root independent of source assets")
    for key in ("table_targets", "floor_targets", "floor_contact_targets"):
        if not isinstance(catalog[key], list) or not catalog[key]:
            raise ValueError("explicit native surface targets required")
        for value in catalog[key]:
            _target(value)
    _target(catalog["key_light"])
    _target(catalog["dome_light"])
    if catalog["key_light"] == catalog["dome_light"]:
        raise ValueError("HDR dome and interchangeable key light must be explicit separate targets")
    _finite(catalog["support_z_m"])
    for mapping in (catalog["clutter"], catalog["materials"]):
        if not isinstance(mapping, dict):
            raise ValueError("catalog assets must be an explicit mapping")
        for key, pin in mapping.items():
            fields = {"path", "sha256", "mdl_name"} if mapping is catalog["materials"] else {"path", "sha256"}
            if (not isinstance(pin, dict) or set(pin) != fields
                    or not isinstance(pin["path"], str) or not pin["path"]
                    or Path(pin["path"]).is_absolute() or ".." in Path(pin["path"]).parts
                    or "\\" in pin["path"] or ":" in pin["path"]):
                raise ValueError("catalog asset paths must remain within the trusted asset root")
            _sha(pin["sha256"])
            if mapping is catalog["materials"]:
                _identifier(pin.get("mdl_name"))


def validate_clutter(rigid, catalog):
    if not isinstance(rigid, dict) or not rigid:
        raise ValueError("probe requires an explicit nonempty clutter layout")
    result = []
    allowed = {"category_idx", "clutter_idx", "default_pos", "default_ori", "settled_pos", "settled_ori",
               "scale", "physics", "relative_plane", "type", "yaml_path"}
    for category, rows in rigid.items():
        _identifier(category)
        if not isinstance(rows, list):
            raise ValueError("clutter category must contain an instance list")
        for i, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != allowed or type(row["category_idx"]) is not int
                    or row["category_idx"] < 0 or row["type"] != "cluttered" or row["relative_plane"] != "Table"
                    or row["yaml_path"] != "Clutter/clutter.yml"
                    or not isinstance(row["physics"], dict) or set(row["physics"]) - {"type", "mass", "friction"}
                    or row["physics"].get("type") != "rigid"):
                raise ValueError("unsupported clutter record; no delivery physics/drive/path overrides")
            for key in ("mass", "friction"):
                if key in row["physics"]:
                    _finite(row["physics"][key])
            _vector(row["default_pos"], 3)
            _vector(row["default_ori"], 4, quaternion=True)
            scale = _vector(row["scale"], 3)
            if any(v <= 0 for v in scale):
                raise ValueError("scale must be positive")
            key = f"{category}/{row['category_idx']:05d}"
            if key not in catalog["clutter"]:
                raise ValueError("clutter is absent from the trusted asset catalog: " + key)
            pin = catalog["clutter"][key]
            result.append(Clutter(f"{category}_{row['category_idx']:05d}_{i}", pin["path"], pin["sha256"],
                                  _vector(row["settled_pos"], 3), _vector(row["settled_ori"], 4, quaternion=True), scale,
                                  clutter_mass(row["physics"])))
    if not 5 <= len(result) <= 100:
        raise ValueError("clutter count outside supported diagnostic bounds")
    return result


def clutter_mass(physics):
    """Exact fixed baseline spawn rule; never adjusted after a probe outcome."""
    value = _finite(physics.get("mass", .5))
    return min(value, .5) if value > 0 else .05


def angle_degrees(a, b):
    dot = abs(sum(x*y for x, y in zip(a, b)))
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return math.degrees(2 * math.acos(min(1., max(0., dot/denominator))))


def stability(samples, *, support_z):
    """Draft v2 gates; contact-gap/penetration/support-only facts stay diagnostic."""
    if len(samples) != 601:
        raise ValueError("need entry plus all 600 actual physics samples")
    names = set(samples[0])
    if not names or any(set(row) != names for row in samples):
        raise ValueError("body inventory changed during settling")
    rows = []
    for name in sorted(names):
        trajectory = [sample[name] for sample in samples]
        positions = [_vector(list(row["position"]), 3) for row in trajectory]
        orientations = [_vector(list(row["orientation"]), 4, quaternion=True) for row in trajectory]
        shift = lambda a, b: max(abs(x-y) for x, y in zip(a, b))
        window = 401  # final 200 post-step observations: samples 401..600
        total_shift = max(shift(positions[0], p) for p in positions)
        total_angle = max(angle_degrees(orientations[0], q) for q in orientations)
        window_shift = max(shift(positions[window], p) for p in positions[window:])
        window_angle = max(angle_degrees(orientations[window], q) for q in orientations[window:])
        bottom = min(_finite(row["bottom_z"]) for row in trajectory)
        floor = any(row.get("floor_contact") is True for row in trajectory)
        failures = [key for key, failed in (("below_support_or_dropped", bottom < support_z - .05),
                    ("floor_contact", floor), ("window_drift", window_shift > .04),
                    ("window_rotation", window_angle > 10), ("total_shift", total_shift > .04),
                    ("total_rotation", total_angle > 30)) if failed]
        rows.append(dict(name=name, max_total_axis_shift_m=total_shift, max_total_angle_deg=total_angle,
                         max_window_axis_shift_m=window_shift, max_window_angle_deg=window_angle,
                         minimum_bottom_z=bottom, floor_contact=floor, failures=failures,
                         passed=not failures))
    return {"bodies": rows, "passed": all(r["passed"] for r in rows),
            "diagnostic_only": ["contact_gap", "penetration", "missing_support_contact", "reload_pixels"]}


@dataclass(frozen=True)
class SceneProbeReceipt:
    schema: str
    task_id: str
    instance_id: str
    logical_slot: str | None
    projection: str
    infra_valid: bool
    completed_steps: int
    entry_clock: dict
    exit_clock: dict
    source_spec_sha256: str
    config_sha256: str
    layout_sha256: str | None
    before_head: str | None
    after_head: str | None
    stability: dict
    provenance: dict
    error: str | None = None

    def __post_init__(self):
        clean = self.projection == SOURCE_CLEAN_PREVIEW
        if (self.schema != RECEIPT_SCHEMA or self.projection not in (PROJECTION, SOURCE_CLEAN_PREVIEW)
                or self.task_id not in (CLEAN_PREVIEW_TASKS if clean else TASKS) or type(self.infra_valid) is not bool
                or type(self.completed_steps) is not int or not 0 <= self.completed_steps <= 600):
            raise ValueError("invalid typed scene probe receipt")
        for value in (self.source_spec_sha256, self.config_sha256):
            _sha(value)
        if clean:
            if self.logical_slot is not None or self.layout_sha256 is not None:
                raise ValueError("clean preview cannot claim a random layout")
        else:
            _sha(self.layout_sha256)
        if self.infra_valid:
            if self.completed_steps != 600 or self.error is not None:
                raise ValueError("valid scene probe requires all 600 steps without infrastructure error")
            validate_clock_delta(self.entry_clock, self.exit_clock, 600)
            if (self.before_head, self.after_head) != ("before-head.png", "after-head.png"):
                raise ValueError("valid scene probe requires independent before/after head images")


def validate_clock_delta(before, after, ticks):
    for row in (before, after):
        if type(row.get("tick")) is not int or row["tick"] < 0:
            raise ValueError("actual World physics step index unavailable")
        _finite(row.get("time_s"))
    if after["tick"] - before["tick"] != ticks or abs(after["time_s"] - before["time_s"] - ticks/120) > 1e-5:
        raise ValueError("actual World clock does not match 120Hz diagnostic steps")
