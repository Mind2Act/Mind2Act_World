"""Passive appliance controls and measured-state thermal coupling.

The robot must depress a spring-return power plunger and rotate the physical
knob. No action name or trajectory timestamp switches appliance state.
"""
import math
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,PhysxSchema
from scipy.spatial.transform import Rotation as R
from .thermal import ThermalState,ThermalDisplay

ROOT='/World/Cooking'

class Appliance:


    def pre_step(self):
        _,q=self.knob.get_world_pose();angle=R.from_quat([*q[1:],q[0]]).as_euler('xyz')[2]
        # Passive bistable detent: stable at +/-45 deg, unstable at zero.
        speed=self.knob.get_angular_velocity()[2]
        torque=.025*math.sin(4*angle)-.003*speed
        self.knob._rigid_prim_view.apply_forces_and_torques_at_pos(torques=np.array([[0,0,torque]],np.float32),is_global=True)

    def post_step(self,dt,*,update_display=True):
        position,_=self.button.get_world_pose();travel=float(self.power_origin[2]-position[2])
        if travel>.0018 and self.armed:
            self.powered=not self.powered;self.press_count+=1;self.armed=False
        if travel<.0005:self.armed=True
        _,q=self.knob.get_world_pose();angle=float(R.from_quat([*q[1:],q[0]]).as_euler('xyz',degrees=True)[2])
        # Hysteresis prevents a transient midpoint from being a false mode toggle.
        if angle<-30:self.mode='KEEP WARM'
        elif angle>30:self.mode='HEAT'
        state=self.thermal.step(dt,powered=self.powered,knob_mode=self.mode)
        if update_display:self.display.update(state['temperature'],state['mode'],bool(state['failure']))
        row=dict(**state,button_travel_m=travel,power_press_count=self.press_count,knob_angle_deg=angle,powered=self.powered)
        self.trace.append(row);return row
