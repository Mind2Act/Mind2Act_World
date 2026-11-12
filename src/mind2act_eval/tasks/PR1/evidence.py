"""Measured resistance of the carriage and the per-tick sample record.

The reading comes from the simulated slider body, never from the commanded pose:
R = full_scale_ohm * (x - shift_x - zero_world_x) / travel   (baseline cases_unified/PR1/evidence.py).
"""
from math import isfinite
from mind2act_eval.tasks._precision.recorder import Recorder  # noqa: F401  (re-exported)


class ResistanceModel:
    def __init__(self, zero_world_x_m: float, travel_m: float, full_scale_ohm: float, shift_x_m: float = 0.):
        self.zero_world_x_m, self.travel_m, self.full_scale_ohm, self.shift_x_m = zero_world_x_m, travel_m, full_scale_ohm, shift_x_m

    @classmethod
    def from_config(cls, levels: dict, shift_x_m: float = 0.) -> "ResistanceModel":
        m = levels["resistance_model"]
        return cls(m["zero_world_x_m"], m["travel_m"], m["full_scale_ohm"], shift_x_m)

    def ohm(self, x_m: float) -> float:
        value = self.full_scale_ohm * (x_m - self.shift_x_m - self.zero_world_x_m) / self.travel_m
        if not isfinite(value):
            raise ValueError("nonfinite resistance")
        return float(value)

    def position_m(self, ohm: float) -> float:
        return self.zero_world_x_m + self.travel_m * ohm / self.full_scale_ohm + self.shift_x_m
