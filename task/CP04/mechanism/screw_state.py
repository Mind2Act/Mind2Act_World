"""Physical separation hysteresis for a removed screw cap.

An opening cap must clear its neck before a later lowering/clockwise motion
can count as reinstallation. No action phase or timeline drives this state.
"""
import math

def cleared_neck(axial_gap_m,lateral_offset_m):
    if not math.isfinite(axial_gap_m) or not math.isfinite(lateral_offset_m):
        raise ValueError('Non-finite screw separation')
    return axial_gap_m>=.012 or lateral_offset_m>=.045
