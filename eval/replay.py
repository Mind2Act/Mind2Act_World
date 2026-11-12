"""Trusted training H5 replay. Never load phases, rewards or expert labels.

Metadata is mandatory: use H5 attributes or an explicit reviewed sidecar.
No source-rate guessing, implicit resampling or final-target padding.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from mind2act_eval.contracts import Action, ClockSpec, InstanceSpec, freeze, to_json


@dataclass(frozen=True)
class ReplayMetadata:
    episode_id: str
    source_id: str
    physics_hz: int
    control_hz: int
    rgb_hz: int
    arm_unit: str
    gripper_unit: str
    gripper_open: float
    gripper_closed: float
    left_arm_key: str = "action/left_arm_joint_states"
    right_arm_key: str = "action/right_arm_joint_states"
    left_gripper_key: str = "action/left_ee_joint_states"
    right_gripper_key: str = "action/right_ee_joint_states"
    instance_spec: object = None
    gripper_action_scale: float = 1.0
    gripper_action_offset: float = 0.0

    def __post_init__(self):
        from math import isfinite
        if not self.episode_id or not self.source_id:
            raise ValueError("replay requires episode_id and source_id")
        ClockSpec(self.physics_hz, self.control_hz, self.rgb_hz)
        if self.instance_spec is None:
            raise ValueError("replay requires full instance_spec binding")
        bound = self.instance_spec if isinstance(self.instance_spec, InstanceSpec) else InstanceSpec.from_dict(self.instance_spec)
        if bound.source_id != self.source_id or bound.clock != self.clock:
            raise ValueError("recorded instance clock/source differs from metadata")
        object.__setattr__(self, "instance_spec", bound)
        if self.arm_unit not in ("rad", "deg") or self.gripper_unit not in ("normalized", "m"):
            raise ValueError("unsupported recorded action units")
        if not all(isfinite(v) for v in (self.gripper_open, self.gripper_closed)) or self.gripper_open == self.gripper_closed:
            raise ValueError("recorded gripper polarity requires two distinct finite endpoints")
        if not all(isfinite(v) for v in (self.gripper_action_scale, self.gripper_action_offset)):
            raise ValueError("gripper action mapping requires finite scale and offset")

    @property
    def clock(self):
        return ClockSpec(self.physics_hz, self.control_hz, self.rgb_hz)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class H5Replay:
    """Streaming adapter: exactly one recorded action per recorded control period."""
    requires_rgb = False
    def __init__(self, path: Path, metadata: ReplayMetadata | None = None, *, expected_sha256=None):
        import h5py
        self.path = Path(path)
        self.h5_sha256 = file_sha256(self.path)
        if expected_sha256 is not None and self.h5_sha256 != expected_sha256:
            raise ValueError("training H5 digest mismatch")
        self._file = h5py.File(self.path, "r")
        try:
            recorded = self._file.attrs.get("mind2act_replay")
            if recorded is not None:
                if isinstance(recorded, bytes):
                    recorded = recorded.decode()
                native = ReplayMetadata(**json.loads(recorded))
                if metadata is not None and native != metadata:
                    raise ValueError("H5/sidecar replay metadata mismatch")
                metadata = native
            if metadata is None:
                raise ValueError("missing recorded replay metadata; provide reviewed explicit metadata")
            self.metadata = metadata
            self._arms = [self._file[key] for key in (metadata.left_arm_key, metadata.right_arm_key)]
            self._grippers = [self._file[key] for key in (metadata.left_gripper_key, metadata.right_gripper_key)]
            self.rows = len(self._arms[0])
            if not self.rows or any(len(ds) != self.rows for ds in self._arms + self._grippers):
                raise ValueError("empty or unequal H5 action row counts")
            if any(ds.shape != (self.rows, 6) for ds in self._arms):
                raise ValueError("H5 arms must be Nx6")
            if any(ds.shape not in ((self.rows,), (self.rows, 1)) for ds in self._grippers):
                raise ValueError("H5 grippers must be N or Nx1; paired fingers require an explicit conversion")
            # HDF5 external links/storage must not bypass the reviewed trajectory file.
            for key in (metadata.left_arm_key, metadata.right_arm_key, metadata.left_gripper_key, metadata.right_gripper_key):
                parts = key.strip("/").split("/")
                group = self._file
                for part in parts:
                    if not isinstance(group.get(part, getlink=True), h5py.HardLink):
                        raise ValueError("replay action dataset uses an indirect HDF5 link")
                    group = group[part]
                if group.external or group.is_virtual:
                    raise ValueError("external/virtual H5 action storage is unsupported")
            self._row = 0
            self._eof = False
        except BaseException:
            self._file.close()
            raise

    def reset(self, seed: int):
        self._row = 0
        self._eof = False

    def assert_spec(self, spec):
        if spec.clock != self.metadata.clock or spec.source_id != self.metadata.source_id:
            raise ValueError("replay clock/source differs from frozen instance")
        if spec.variation.mode == "random":
            from mind2act_eval.variations import base_spec_digest, clean_spec
            if self.metadata.instance_spec.variation.mode != "clean":
                raise ValueError("random replay requires the exact clean expert instance")
            clean_spec(spec).assert_binding(self.metadata.instance_spec)
            if base_spec_digest(self.metadata.instance_spec) != spec.variation.base_spec_sha256:
                raise ValueError("random replay base digest differs from expert instance")
        else:
            spec.assert_binding(self.metadata.instance_spec)

    def act(self, observation):
        if self._row >= self.rows:
            self._eof = True
            raise StopIteration
        if observation.tick != self._row * self.metadata.clock.ticks_per_action:
            raise ValueError("replay observation tick differs from recorded control cadence")
        action = self._action_at(self._row)
        self._row += 1
        return action

    def peek_next_action(self):
        """Read the following recorded row without consuming it or declaring EOF."""
        return self._action_at(self._row) if self._row < self.rows else None

    def _action_at(self, row):
        import numpy as np
        arms = [np.asarray(ds[row], dtype=float) for ds in self._arms]
        if self.metadata.arm_unit == "deg":
            arms = [np.deg2rad(values) for values in arms]
        grippers = []
        for ds in self._grippers:
            raw = float(np.asarray(ds[row]).reshape(-1)[0])
            source_opening = ((raw - self.metadata.gripper_closed) /
                              (self.metadata.gripper_open - self.metadata.gripper_closed))
            grippers.append(self.metadata.gripper_action_scale * source_opening +
                            self.metadata.gripper_action_offset)
        return Action(tuple(arms[0]), tuple(arms[1]), *grippers)

    def receipt(self):
        return {"metadata": to_json(self.metadata), "h5_sha256": self.h5_sha256,
                "rows": self.rows,
                "conversion": "recorded units/polarity normalization plus explicit affine action-target mapping; recorded clock preserved",
                "consumer_kind": "expert_replay", "model_tested": False}

    def coverage_receipt(self):
        """Return replay progress only; keep the bound spec and H5 values private."""
        return {
            "rows_total": self.rows,
            "adapter_returned_rows": self._row,
            "adapter_eof": self._eof,
        }

    def close(self):
        self._file.close()
