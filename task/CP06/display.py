"""Purely visual LCD driver; no collision or rigid body and no numeric readout."""
import math
from pxr import Gf, UsdGeom

def update_cursor(stage, balance_path, beam_relative_angle_deg):
    cursor=UsdGeom.Xform(stage.GetPrimAtPath(balance_path+'/Base/LCD/Cursor'))
    x=-.01225*beam_relative_angle_deg
    cursor.GetOrderedXformOps()[0].Set(Gf.Vec3d(x/1.65,0,0))
    return x


def green_angle_interval(stage, balance_path):
    prim=stage.GetPrimAtPath(balance_path+'/Base/LCD/GreenBand')
    attr=prim.GetAttribute('cp06:angleIntervalDeg')
    value=attr.Get() if attr else None
    if value is not None:return float(value[0]),float(value[1])
    half=float(prim.GetAttribute('cp06:halfWidthDeg').Get())
    return -half,half

def is_in_green(stage, balance_path, corrected_angle_deg, reading_valid):
    lo,hi=green_angle_interval(stage,balance_path)
    return bool(reading_valid and lo <= float(corrected_angle_deg) <= hi)
