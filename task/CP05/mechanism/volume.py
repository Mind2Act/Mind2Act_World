import numpy as np

def volume_reached_max(position, initial, axis, travel, contact):
 """Require finite measured travel along the same axis as the passive joint."""
 import math
 if axis not in ('X','Y') or not math.isfinite(travel) or travel<=.004:
  raise ValueError('Invalid volume travel contract')
 if not contact or len(position)!=3 or len(initial)!=3:
  return False
 if not all(math.isfinite(float(v)) for v in (*position,*initial)):
  return False
 index={'X':0,'Y':1}[axis]
 return float(position[index]-initial[index])>=travel-.004
