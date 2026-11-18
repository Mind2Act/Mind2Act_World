"""Observable task temperature model, driven by measured appliance controls.

This is an explicit benchmark state model, not heat-transfer simulation. The
renderer only reads this state; it cannot advance temperature or mark success.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from .thermal_profile497 import window_seconds, EPS_C

WINDOW_SECONDS={'easy':12.,'medium':8.,'hard':5.}
@dataclass
class ThermalState:
    difficulty:str
    temperature:float=60.
    time:float=0.
    mode:str='OFF'
    window_entered:float|None=None
    switched_at:float|None=None
    hold_temperature:float|None=None
    failure:str|None=None
    power_off_after_hold:bool=False

    thermal_profile_version:str|None=None

    @property
    def limit(self):return 80.+2.5*window_seconds(self.difficulty,self.thermal_profile_version)
    def step(self,dt,*,powered,knob_mode):
        if not math.isfinite(dt) or dt<0:raise ValueError('Invalid elapsed simulation time')
        if knob_mode not in ('HEAT','KEEP WARM'):raise ValueError('Unknown measured knob mode')
        mode=knob_mode if powered else 'OFF'
        if mode=='KEEP WARM' and self.mode=='HEAT' and self.switched_at is None:
            self.switched_at=self.time
            if self.temperature<80.:self.failure=self.failure or 'switched_before_preheat'
            elif self.temperature>self.limit+(EPS_C if self.thermal_profile_version else 0.):self.failure=self.failure or 'over_temperature'
            else:self.hold_temperature=self.temperature
        if mode=='OFF' and self.hold_temperature is not None:self.power_off_after_hold=True
        old=self.temperature
        if mode=='HEAT':self.temperature+=2.5*dt
        elif mode=='KEEP WARM' and self.hold_temperature is not None:self.temperature=self.hold_temperature
        elif mode=='OFF' and self.thermal_profile_version is None:self.temperature=max(60.,self.temperature-.5*dt)
        # Version497 v3: measured OFF leaves actual scalar temperature unchanged.
        if self.window_entered is None and old<80.<=self.temperature:
            self.window_entered=self.time+(80.-old)/2.5
        if self.temperature>self.limit+1e-8:self.failure=self.failure or 'over_temperature'
        self.time+=dt;self.mode=mode
        return self.snapshot()
    def snapshot(self):
        return dict(**asdict(self),limit=self.limit,window_seconds=window_seconds(self.difficulty,self.thermal_profile_version),
                    valid_switch=self.hold_temperature is not None and self.failure is None,
                    model='benchmark scalar temperature; not calibrated thermal physics')

class ThermalDisplay:
    """Visible progress fill plus pointer; avoids relying on a tiny cursor alone."""
    def update(self,temperature,mode='OFF',failed=False):
        # FXAA uses current-frame pixels; no temporal digit/mode history.
        # Retain native120Hz,30Hz RGB and exactly eight render refreshes.
        import carb
        settings=carb.settings.get_settings()
        if settings.get('/rtx/post/aa/op') != 2:settings.set('/rtx/post/aa/op',2)
        from pxr import UsdShade,UsdGeom
        v=max(60.,min(140.,temperature));x=self.bar_left+(v-60)/80*(self.bar_right-self.bar_left)
        width=max(.0001,x-self.bar_left)
        self.position.Set(self.Gf.Vec3d(self.bar_left+width/2,self.bar_middle,.0057))
        self.scale.Set(self.Gf.Vec3f(width,self.bar_top-self.bar_bottom-.004,.00015))
        self.fill.MakeVisible() if v>60.01 else self.fill.MakeInvisible()
        self.cook.cursor.Set(self.Gf.Vec3d(x,0,0))
        text=f'{v:5.1f} C'
        if text!=self.digit_text:
            for index,(choices,char) in enumerate(zip(self.bank,text)):
                for glyph,attr in choices.items():attr.Set(self.Gf.Vec3d(-.079+index*.0105,(.5-62/500)*.092,.0058 if glyph==char else -.1))
            self.digit_text=text
        shown='OVER LIMIT' if failed else mode
        for name,label in self.mode_labels.items():
            label.GetPrim().GetAttribute('xformOp:translate').Set(self.Gf.Vec3d(.075,.030,.006 if name==shown else -.1))
