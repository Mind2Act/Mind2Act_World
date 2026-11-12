"""Passive telescopic drawer: robot contact supplies opening/closing force."""
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,PhysxSchema
import numpy as np
from .geometry import DRAWER_TRAVEL,DRAWER_CENTER_X
ROOT='/World/Cooking/CondimentCabinet'

def apply_rail_detent(body,middle=None):
    """Passive short end detents prevent a freely sliding tray drifting back.

    Spring acts only within the final 12 mm of either end; it cannot open a
    closed drawer or close an open one without robot contact traversing travel.
    """
    position,_=body.get_world_pose();y=float(position[1]);target=None
    for stop in [.065,.065-DRAWER_TRAVEL]:
        if abs(y-stop)<.012:target=stop;break
    if target is not None:
        velocity=body.get_linear_velocity()[1];force=float(np.clip(120*(target-y)-2*velocity,-1.5,1.5))
        body._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.array([[0,force,0]],np.float32),is_global=True)
    if middle is not None:
        middle_y=float(middle.get_world_pose()[0][1]);error=middle_y-(.0325+.5*y)
        speed=middle.get_linear_velocity()[1]-.5*body.get_linear_velocity()[1]
        force=float(np.clip(-400*error-5*speed,-4,4))
        middle._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.array([[0,force,0]],np.float32),is_global=True)
        body._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.array([[0,-.5*force,0]],np.float32),is_global=True)
