"""Resolve frozen background inputs on CPU; apply once through the task builder.

The CLI adds a manifest named variation_manifest to the deployment resolver;
the frozen clean spec's asset mapping and hash remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mind2act_eval.contracts import InstanceSpec, VariationSpec, freeze, to_json
from .layouts import Layout, read_layout, vector


def digest(value):
    return hashlib.sha256(json.dumps(to_json(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def base_spec_digest(spec):
    # Preserve pre-variation spec identity; no config/asset keys are removed.
    value = to_json(spec)
    value.pop("variation", None)
    return digest(value)


def clean_spec(spec):
    return replace(spec, variation=VariationSpec())


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path, expected, *, strict=False):
    data = Path(path).read_bytes()
    if strict and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("variation input digest mismatch: " + str(path))
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate variation JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("variation manifest paths must be relative")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("variation path escapes its manifest root")
    return result


@dataclass(frozen=True)
class AssetPin:
    path: Path
    sha256: str
    metadata: Mapping[str, Any]
    dependencies: tuple[tuple[Path, str], ...] = ()
    search_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VariationPlan:
    spec: VariationSpec
    instance_sha256: str
    layout: Layout | None = None
    assets: Mapping[str, AssetPin] | None = None
    binding: Mapping[str, Any] | None = None
    mdl_search_paths: tuple[Path, ...] = ()

    @property
    def sha256(self):
        # Local install paths never participate in portable plan identity.
        return digest({"variation": self.spec, "base_spec_sha256": self.instance_sha256,
                       "layout": self.layout, "binding": self.binding,
                       "assets": {k: {"sha256": v.sha256, "metadata": v.metadata}
                                  for k, v in (self.assets or {}).items()}})


def resolve(spec: InstanceSpec, assets, *, strict=False) -> VariationPlan:
    base = base_spec_digest(spec)
    if spec.variation.mode == "clean":
        return VariationPlan(spec.variation, base)
    if strict and base != spec.variation.base_spec_sha256:
        raise ValueError("variation does not match the frozen clean InstanceSpec")
    manifest_path = Path(assets.resolve("variation_manifest")).resolve(strict=True)
    manifest = _read(manifest_path, spec.variation.asset_manifest_sha256, strict=strict)
    if manifest.get("schema") != "mind2act_eval.variation_manifest.v1":
        raise ValueError("unsupported variation manifest")
    binding = manifest["layouts"][spec.variation.layout_id]
    if ((strict and binding["base_spec_sha256"] != base) or binding["task_id"] != spec.task_id
            or binding["instance_id"] != spec.instance_id or binding["difficulty"] != spec.difficulty
            or binding["seed"] != spec.seed or type(binding["seed"]) is not int):
        raise ValueError("layout manifest task/slot/seed/base binding mismatch")
    root = manifest_path.parent
    search_paths = tuple(_path(root, p) for p in manifest.get("mdl_search_paths", []))
    if any(not p.is_dir() for p in search_paths):
        raise ValueError("variation MDL search path must be a directory")
    module_files = []
    for row in manifest.get("mdl_files", []):
        path = _path(root, row["path"])
        if strict and file_digest(path) != row["sha256"]:
            raise ValueError("variation MDL module changed: " + row["path"])
        module_files.append((path, row["sha256"]))
    if search_paths and not module_files:
        raise ValueError("variation MDL search paths require pinned module files")
    layout_data = _read(_path(root, binding["path"]), spec.variation.layout_sha256, strict=strict)
    layout = read_layout(layout_data, binding["coordinates"])
    if (layout.task != binding["layout_task"] or layout.slot != binding["layout_slot"]
            or layout.tier != spec.difficulty):
        raise ValueError("layout payload does not match the reviewed slot mapping")
    # Do not infer generation/settle/baseline versions from filenames.
    for key in ("generator_version", "baseline_sha256", "settled_receipt_sha256", "exclusion_receipt_sha256"):
        if not isinstance(binding.get(key), str) or not binding[key]:
            raise ValueError("missing reviewed layout provenance: " + key)
    if strict and binding["baseline_sha256"] != digest(spec.config):
        raise ValueError("layout baseline configuration differs from frozen task")
    from math import isfinite
    for key in ("initialization_pose_tolerance_m", "initialization_pose_tolerance_rad"):
        value = binding.get(key)
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("requires frozen source initialization tolerance: " + key)
    if strict and digest(layout.provenance["settle"]) != binding["settled_receipt_sha256"]:
        raise ValueError("archived settle receipt changed")
    excluded = binding.get("forbidden_aabbs")
    if not isinstance(excluded, list):
        raise ValueError("requires explicit source exclusion zones (empty list allowed)")
    for zone in excluded:
        lo, hi = vector(zone[0], 3), vector(zone[1], 3)
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError("invalid source exclusion zone")
    if strict and digest(excluded) != binding["exclusion_receipt_sha256"]:
        raise ValueError("source exclusion receipt changed")
    keys = {o.asset_key for o in layout.objects}
    if layout.appearance["table_enabled"]:
        keys.add(layout.appearance["material"]["id"])
    keys.add(layout.ground)
    pins = {}
    for key in sorted(keys):
        pin = manifest["assets"][key]
        path = _path(root, pin["path"])
        if strict and file_digest(path) != pin["sha256"]:
            raise ValueError("variation asset digest mismatch: " + key)
        dependencies = []
        for dep in pin["dependencies"]:
            dep_path = _path(root, dep["path"])
            if strict and file_digest(dep_path) != dep["sha256"]:
                raise ValueError("variation asset dependency changed: " + key)
            dependencies.append((dep_path, dep["sha256"]))
        metadata = {k: v for k, v in pin.items() if k not in ("path", "sha256")}
        pins[key] = AssetPin(path, pin["sha256"], freeze(metadata), tuple([*dependencies, *module_files]), search_paths)
    # Pin payload material identity as well as the actual file.
    material_rows = [layout.appearance["material"]] if layout.appearance["table_enabled"] else []
    if isinstance(layout_data.get("ground"), dict):
        material_rows.append(layout_data["ground"])
    for row in material_rows:
        pin = pins[row["id"]]
        if (strict and pin.sha256 != row["sha256"]) or pin.metadata["mdl_name"] != row["mdl_name"]:
            raise ValueError("delivered material differs from reviewed catalog")
    from types import MappingProxyType
    return VariationPlan(spec.variation, base, layout, MappingProxyType(pins), freeze(binding), search_paths)


def configure_material_search(spec, assets, baseline_ids=()):
    """Configure pinned variation modules before Kit, retaining baseline roots."""
    plan = resolve(spec, assets)
    if not plan.mdl_search_paths:
        return None
    from types import SimpleNamespace
    from mind2act_eval.sim.material_closure import configure_mdl_search
    paths = {key: assets.resolve(key) for key in baseline_ids}
    paths.update({f"variation_mdl_{i}": path for i, path in enumerate(plan.mdl_search_paths)})
    return dict(configure_mdl_search(SimpleNamespace(resolve=paths.__getitem__), list(paths)),
                prioritize_declared_paths=True)


@dataclass(frozen=True)
class VariationTargets:
    """Builder capabilities: exact background paths, never delivery-supplied paths."""
    clutter_root: str
    table_surfaces: tuple[str, ...]
    floor_surfaces: tuple[str, ...]
    key_light: str
    dome_light: str
    protected_roots: tuple[str, ...] = ()
    runtime_layer: Path | None = None

    def __post_init__(self):
        from pxr import Sdf
        paths = (self.clutter_root, *self.table_surfaces, *self.floor_surfaces,
                 self.key_light, self.dome_light)
        if len(set(paths)) != len(paths):
            raise ValueError("variation capabilities must be disjoint")
        for path in (*paths, *self.protected_roots):
            p = Sdf.Path(path)
            if not p.IsAbsolutePath() or not p.IsPrimPath() or path in ("/", "/World"):
                raise ValueError("variation targets must be concrete absolute prim paths")
        for a in paths:
            for b in paths:
                if a != b and Sdf.Path(a).HasPrefix(Sdf.Path(b)):
                    raise ValueError("overlapping variation capabilities")
            for b in self.protected_roots:
                if Sdf.Path(a).HasPrefix(Sdf.Path(b)) or Sdf.Path(b).HasPrefix(Sdf.Path(a)):
                    raise ValueError("variation capability overlaps a protected task/robot/camera")


def station_targets(*, table_surfaces=("/World/Worktable",), runtime_layer=None):
    """Explicit shared station background names, declared by each opting-in builder.

    A station with different names fails if that capability is needed. It must
    receive a reviewed builder declaration; no path guessing or fallback.
    """
    return VariationTargets("/World/EvalVariation", table_surfaces, ("/World/Floor",),
                            "/World/Softbox", "/World/Studio",
                            ("/World/RobotLeft", "/World/RobotRight", "/World/HeadCamera", "/World/Task"),
                            runtime_layer)


def prepare_station(ctx, *, table_surfaces=("/World/Worktable",)):
    """Opt-in builder seam. Legacy clean contexts perform no additional work."""
    if getattr(ctx, "variation_session", None) is not None:
        ctx.prepare_variation(station_targets(table_surfaces=table_surfaces,
                                             runtime_layer=ctx.output_dir / "variation.usda"))


def apply(plan, stage, targets):
    from .stage import apply as compose
    return compose(plan, stage, targets)


class VariationSession:
    def __init__(self, plan):
        self.plan, self.receipt, self.stage, self.targets = plan, None, None, None

    def apply(self, stage, targets):
        if self.receipt is not None:
            raise ValueError("variation builder seam invoked twice")
        self.receipt = apply(self.plan, stage, targets)
        self.stage, self.targets = stage, targets

    def verify_visuals(self, *, stage=None):
        if self.receipt is None:
            raise ValueError("random task builder has no variation initialization seam")
        from .stage import verify_appearance
        active_stage = self.stage if stage is None else stage
        self.receipt.pop("post_initialization_visual", None)
        self.receipt["post_initialization_visual"] = verify_appearance(active_stage, self.targets, self.receipt)
        return self.receipt

    def verify(self, *, native=False, stage=None):
        self.verify_visuals(stage=stage)
        from .stage import verify_settled
        active_stage = self.stage if stage is None else stage
        self.receipt["post_initialization"] = verify_settled(self.plan, active_stage, self.targets, native=native)
        if native:
            from .geometry import verify_geometry
            self.receipt["geometry_clearance"] = verify_geometry(active_stage, self.targets)
        return self.receipt
