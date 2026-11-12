"""Task bookkeeping follows actual robot opening commands and bilateral marker contact."""
import numpy as np
from pxr import UsdPhysics


def release(env,side):
 tool='eraser' if side=='Left' else 'pen'
 if tool=='eraser' and env.grasp_latches['eraser'] and env.contact_counts['eraser']>0 and env.original_ink_after_erasing is None:
  env.original_ink_after_erasing=True
 env.board_support.release(side)
 if side=='Left':env.caps.release()
 if env.grasp_latches[tool]:
  root=env.tool_root(tool);env.stage_usd.RemovePrim(root+'/VerifiedGraspLatch');UsdPhysics.FilteredPairsAPI(env.stage_usd.GetPrimAtPath(root)).GetFilteredPairsRel().ClearTargets(True);env.grasp_latches[tool]=False

def before_control(env):
 for side in ('Left','Right'):
  now=np.asarray(env.commands[side][6:]);old=env.previous_finger_commands[side]
  # Match the source's existing >20 mm opening convention; ignore retained-grasp jitter.
  if np.mean(now)>.020 and np.all(now>old):release(env,side)

def select_contacted_marker(env):
 if env.grasp_latches['pen']:return
 tcp,_=env.kin['Right'].fk(env.robots['Right'].get_joint_positions()[:6]);candidates=[]
 for color in env.pen_verified:
  body=env.all_tools['pen'+str(color)];p,q=body.get_world_pose()
  from scipy.spatial.transform import Rotation
  r=Rotation.from_quat([*q[1:],q[0]]).as_matrix();distance=np.linalg.norm(tcp-(p+r@np.array([0,0,env.grip_height['pen']])))
  if distance>.035:continue
  root=env.tool_root_for('pen',color)+'/'
  contacts=[any(any(p.startswith(root) for p in pair) and any(p.startswith('/World/RobotRight/Asset/'+link+'/') for p in pair) and active and (impulse>1e-8 or sep<.00005) for pair,(active,impulse,sep) in env.contact_pairs.items()) for link in ('link7','link8')]
  if all(contacts):candidates.append((distance,color))
 if candidates:
  _,color=min(candidates)
  if color!=env.active_pen:env.select_pen(color);env.interaction_events.append(dict(tick=env.tick,event='active_marker_from_actual_bilateral_contact',color=color))
