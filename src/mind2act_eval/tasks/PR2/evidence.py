"""Measured voltage from the live knob yaw.

V = zero_value + (yaw_deg - zero_deg) / deg_per_unit, yaw = atan2(R[1,0], R[0,0]) of the knob's world
transform (baseline cases_unified/PR2/adapter.py:knob_value). The reading is the simulated knob, never
the commanded pose.
"""
import math
from mind2act_eval.tasks._precision.recorder import Recorder  # noqa: F401  (re-exported)


class VoltageModel:
    def __init__(self, zero_deg: float, zero_value: float, deg_per_unit: float):
        if deg_per_unit == 0:
            raise ValueError('deg_per_unit must be nonzero')
        self.zero_deg, self.zero_value, self.deg_per_unit = zero_deg, zero_value, deg_per_unit

    @classmethod
    def from_config(cls, mapping: dict) -> "VoltageModel":
        return cls(mapping['zero_deg'], mapping['zero_value'], mapping['deg_per_unit'])

    def volts(self, yaw_deg: float) -> float:
        value = self.zero_value + (yaw_deg - self.zero_deg) / self.deg_per_unit
        if not math.isfinite(value):
            raise ValueError('nonfinite voltage')
        return float(value)

    def yaw_deg(self, volts: float) -> float:
        return self.zero_deg + (volts - self.zero_value) * self.deg_per_unit


def yaw_deg(rotation_rows) -> float:
    """World yaw from the first column of a 3x3 rotation (row-major rows)."""
    return math.degrees(math.atan2(rotation_rows[1][0], rotation_rows[0][0]))
