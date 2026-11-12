"""Original source task-condition authoring before reset; no physical tuning."""
import numpy as np
from scipy.spatial.transform import Rotation


def apply(stage,spec):
 from pxr import UsdGeom,Gf
 c=spec.config['task_condition'];start=np.array(c['start_offset_xy_m']);goal=np.array(c['goal_offset_xy_m']);yaw=c['tray_initial_yaw_deg'];rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
 changed=[]
 def translate(path,delta):
  prim=stage.GetPrimAtPath(path);assert prim.IsValid(),path
  attr=prim.GetAttribute('xformOp:translate');assert attr.IsValid(),path
  original=np.array(attr.Get());p=original+np.r_[delta,0.];attr.Set(Gf.Vec3d(*p));changed.append(dict(path=path,initial_position=original.tolist(),position=p.tolist()))
 translate('/World/PR7/StartPlatform',start);translate('/World/PR7/GoalPlatform',goal)
 tray=stage.GetPrimAtPath('/World/PR7/Tray');t=np.array(tray.GetAttribute('xformOp:translate').Get())
 translate('/World/PR7/Tray',start)
 orient=tray.GetAttribute('xformOp:orient');old=orient.Get();q=Gf.Quatf(float(np.cos(np.radians(yaw)/2)),Gf.Vec3f(0.,0.,float(np.sin(np.radians(yaw)/2))))
 orient.Set(q*old)
 for i in range(spec.config['source_case']['ball_count']):
  prim=stage.GetPrimAtPath('/World/PR7/Ball'+str(i));attr=prim.GetAttribute('xformOp:translate');p=np.array(attr.Get());p=t+rz@(p-t)+np.r_[start,0.];attr.Set(Gf.Vec3d(*p));changed.append(dict(path=str(prim.GetPath()),position=p.tolist()))
 names=['Column'] if spec.difficulty=='medium' else ['ColumnLeft','ColumnRight'] if spec.difficulty=='hard' else []
 for name,delta in zip(names,c['obstacle_offsets_xy_m']):
  translate('/World/PR7/'+name,delta)
  if spec.difficulty=='medium':
   for part in ['ColumnMount','ColumnVisual']:
    if stage.GetPrimAtPath('/World/PR7/'+part).IsValid():translate('/World/PR7/'+part,delta)
 # Printed START/GOAL labels follow their task platforms.
 for prim in stage.GetPrimAtPath('/World/PR7').GetChildren():
  if 'Start' in prim.GetName() and prim.GetName()!='StartPlatform':
   if prim.GetAttribute('xformOp:translate').IsValid():translate(str(prim.GetPath()),start)
  if 'Goal' in prim.GetName() and prim.GetName()!='GoalPlatform':
   if prim.GetAttribute('xformOp:translate').IsValid():translate(str(prim.GetPath()),goal)
 return dict(task_condition=c,physical_changes=changed,initial_tray_yaw_deg=yaw,applied_before_physics_reset=True)
