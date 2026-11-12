"""Physical CP06 controls. Caller supplies measured pan occupancy, never a phase label."""
import math
from pxr import Gf,UsdPhysics,UsdShade
class BalanceControls:
 def __init__(self,stage,button,base,beam,lever=None,prefix='/World/Task'):
  lever = lever if stage.GetPrimAtPath(prefix+'/ControlJoints/Brake') else None
  self.stage,self.button,self.base,self.beam,self.lever=stage,button,base,beam,lever
  self.prefix=prefix;self.pivot=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(prefix+'/Balance/Joints/Pivot'))
  self.button_origin=self._relative(button)[0];self.zero_deg=0.;self.calibrated=False;self.locked=False;self.armed=True;self.stable_seconds=0.;self.previous_angle=None;self.events=[]
  self.lever_drive=UsdPhysics.DriveAPI(stage.GetPrimAtPath(prefix+'/ControlJoints/Brake'),'angular') if lever else None
 def _relative(self,body):
  p,q=body.get_world_pose();bp,bq=self.base.get_world_pose();br=Gf.Rotation(Gf.Quatd(float(bq[0]),Gf.Vec3d(*map(float,bq[1:]))));pos=br.GetInverse().TransformDir(Gf.Vec3d(*map(float,p))-Gf.Vec3d(*map(float,bp)));rot=br.GetInverse()*Gf.Rotation(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))));return pos,rot
 def step(self,beam_deg,empty_pans,rider_g=0.,dt=1/120):
  # Empty means no measured load/support on either pan; missing measurement is invalid.
  empty=empty_pans is True;slow=self.previous_angle is not None and abs(beam_deg-self.previous_angle)/dt<.2;self.previous_angle=beam_deg
  self.stable_seconds=self.stable_seconds+dt if empty and slow and not self.locked and abs(rider_g)<.25 else 0.
  if self.lever:
   _,rot=self._relative(self.lever);v=rot.TransformDir(Gf.Vec3d(0,0,1));angle=math.degrees(math.atan2(-v[1],v[2]))
   if not self.locked and angle<-10:
    if empty and abs(beam_deg)<1:
     self.pivot.GetLowerLimitAttr().Set(0.);self.pivot.GetUpperLimitAttr().Set(0.);self.locked=True;self.lever_drive.GetTargetPositionAttr().Set(-30.);self.events.append({'type':'brake_locked','empty_pans':True,'beam_deg':beam_deg})
    else:self.lever_drive.GetTargetPositionAttr().Set(30.)
   if self.locked and angle>10:
    self.pivot.GetLowerLimitAttr().Set(-8.);self.pivot.GetUpperLimitAttr().Set(8.);self.locked=False;self.lever_drive.GetTargetPositionAttr().Set(30.);self.events.append({'type':'brake_released','beam_deg':beam_deg})
  else:angle=None
  pos,_=self._relative(self.button);travel=self.button_origin[2]-pos[2]
  if travel<.0015:self.armed=True
  if self.armed and travel>=.004:
   self.armed=False;valid=empty
   if valid:self.zero_deg=float(beam_deg);self.calibrated=True
   self.events.append({'type':'calibration_press','accepted':valid,'beam_deg':beam_deg,'zero_deg':self.zero_deg,'empty_pans':empty,'rider_g':rider_g,'released':not self.locked})
  ready=self.calibrated and not self.locked
  lamp=self.stage.GetPrimAtPath(self.prefix+'/Balance/Base/ReadyLight')
  if lamp:
   mat=UsdShade.MaterialBindingAPI(lamp).ComputeBoundMaterial()[0];shader=UsdShade.Shader(self.stage.GetPrimAtPath(mat.GetPath().AppendChild('Surface')));shader.GetInput('emissiveColor').Set(Gf.Vec3f(.02,.8,.05) if ready else Gf.Vec3f(0));shader.GetInput('diffuseColor').Set(Gf.Vec3f(.02,.65,.05) if ready else Gf.Vec3f(.012,.025,.015))
  return {'ready_light_on':ready,'button_travel_m':float(travel),'lever_angle_deg':angle,'brake_locked':self.locked,'zero_offset_deg':self.zero_deg,'display_and_score_angle_deg':float(beam_deg)-self.zero_deg,'calibrated':self.calibrated,'reading_valid':self.calibrated and not self.locked,'score_valid':self.calibrated and not self.locked,'calibration_ready':empty}
