"""Private A3 predicate; targets reconstructed from original source function/spec."""
import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from shapely.geometry import MultiPoint, box
from mind2act_eval.tasks.DR3.geometry import slab_projection
from mind2act_eval.tasks.DR3.geometry import transform_vertices
from .mechanism import gates as build_schedules,single_target
IMPULSE_THRESHOLDS=(0.,1e-5,1e-4,1e-3,1e-2)
DEFAULT_IMPULSE_NS=1e-3
CORRIDOR_MARGIN_M=.01
CORRIDOR_MARGINS_M=(0.,.005,.01,.02,.03)
NUMERICAL_EPS_M=1e-9

def source_target(source,difficulty,gate_id,physics_tick,phase_ticks):
 t=max(0,physics_tick-1+phase_ticks)/120
 schedule=build_schedules(source,difficulty)[gate_id]
 return single_target(t,source)if schedule is None else schedule.target(t)

def gate_contact_paths(manifest,gate):
 paths=set(manifest['gate_colliders']+manifest['frame_colliders'])
 if gate['id']=='single':return paths
 prefix='/World/Task/Gate/'+gate['id']+'/'
 return {p for p in paths if p.startswith(prefix)}

def corridor_overrun(manifest,poses,gate,margin_m=CORRIDOR_MARGIN_M):
 near=float(gate['origin'][0])-gate['half_thickness_m'];origin=np.asarray(gate['origin']);low=np.asarray(gate['aperture_low_yz'])+origin[1:]-margin_m;high=np.asarray(gate['aperture_high_yz'])+origin[1:]+margin_m;hits=[]
 for name,b in manifest['bodies'].items():
  if b['role']!='robot'or not name.startswith(('/World/RobotLeft/','/World/RobotRight/')):continue
  if b.get('vertices_are_actual_collision_points')is not True:raise ValueError('left collider not certified actual vertices')
  parts=b.get('collision_parts');enclosure_only=parts is None
  if enclosure_only:parts=[dict(vertices=b['vertices'],approximation='unresolved_enclosure',source_path=';'.join(b['colliders']))]
  for part in parts:
   if not enclosure_only and part.get('approximation')!='convexHull':raise ValueError('left collision approximation not supported by private A3 convex predicate: '+str(part.get('approximation')))
   xyz=transform_vertices(part['vertices'],poses[name]);lo=xyz.min(0);hi=xyz.max(0)
   if hi[0]<=near+NUMERICAL_EPS_M or np.any(hi[1:]<low)or np.any(lo[1:]>high):continue
   hull=ConvexHull(xyz)
   # Actual convex collision part clipped to the infinite opening corridor;
   # maximize X over that intersection, rather than penalize out-of-corridor vertices.
   answer=linprog([-1.,0.,0.],A_ub=hull.equations[:,:3],b_ub=-hull.equations[:,3],bounds=[(None,None),(float(low[0]),float(high[0])),(float(low[1]),float(high[1]))],method='highs',options={'threads':1})
   if answer.status==2:continue
   if not answer.success:raise ValueError('corridor intersection LP unresolved: '+answer.message)
   if answer.x[0]>near+NUMERICAL_EPS_M:hits.append(dict(body=name,collider=part['source_path'],witness_world_m=answer.x.tolist(),overrun_mm=float((answer.x[0]-near)*1000),unresolved_enclosure=enclosure_only))
 return hits

def static_clearances(manifest,poses):
 result=[]
 names=['/World/RobotRight/Asset/base_link','/World/RobotLeft/Asset/base_link','/World/RobotLeft/Asset/link6','/World/RobotLeft/Asset/link8','/World/RobotRight/Asset/link6','/World/RobotRight/Asset/link8']
 for name in names:
  if name not in manifest['bodies']:continue
  body=manifest['bodies'][name];parts=body.get('collision_parts');enclosure=parts is None
  if enclosure:parts=[dict(vertices=body['vertices'])]
  for gate in manifest['gates']:
   near=gate['origin'][0]-gate['half_thickness_m'];low=np.asarray(gate['aperture_low_yz'])+np.asarray(gate['origin'][1:]);high=np.asarray(gate['aperture_high_yz'])+np.asarray(gate['origin'][1:]);all_projection=[];far_projection=[];maximum_x=-float('inf')
   for part in parts:
    xyz=transform_vertices(part['vertices'],poses[name]);maximum_x=max(maximum_x,float(xyz[:,0].max()));all_projection.append(MultiPoint(xyz[:,1:]).convex_hull)
    if xyz[:,0].max()>near+NUMERICAL_EPS_M:
     section=slab_projection(xyz,near,float(xyz[:,0].max()))
     if len(section):far_projection.append(MultiPoint(section[:,1:]).convex_hull)
   distance={}
   for margin in CORRIDOR_MARGINS_M:
    rect=box(low[0]-margin,low[1]-margin,high[0]+margin,high[1]+margin)
    distance[str(float(margin))]=dict(all_part_projection_clearance_mm=1000*min(p.distance(rect)for p in all_projection),far_side_part_projection_clearance_mm=1000*min(p.distance(rect)for p in far_projection)if far_projection else None)
   result.append(dict(body=name,gate_id=gate['id'],max_world_x_m=maximum_x,near_slab_x_m=near,entire_body_robot_side=maximum_x<=near+NUMERICAL_EPS_M,conservative_enclosure=enclosure,by_margin=distance))
 return result

class Audit:
 def __init__(self,manifest,source,difficulty,phase_ticks,*,margin_m=CORRIDOR_MARGIN_M):
  self.m=manifest;self.source=source;self.difficulty=difficulty;self.phase=phase_ticks;self.margin=margin_m;self.events=[];self.contact_events=[];self.all_contact_peaks={};self.passed={};self.closing={};self.done={};self.initial=None;self.right_initial={};self.right_max_motion=0.;self.right_item_contact_peak=0.;self.right_gate_contact_peak=0.;self.n=0
  self.paths={g['id']:gate_contact_paths(manifest,g)for g in manifest['gates']}
  self.item=manifest['item_root'];self.vertices=manifest['bodies'][self.item]['vertices'];self.schedules=build_schedules(source,difficulty);self.previous_targets={};self.allgatepaths=set().union(*self.paths.values())
 def feed(self,row):
  i=row['physics_tick'];assert i==self.n;self.n+=1;f=row['detector'];assert f['contacts_complete']is True
  if self.initial is None:
   self.initial=dict(corridor_hits_by_margin={str(float(margin)):{g['id']:corridor_overrun(self.m,f['poses'],g,margin)for g in self.m['gates']}for margin in CORRIDOR_MARGINS_M},static_clearances=static_clearances(self.m,f['poses']))
  # Right-arm involvement is diagnostic only; never affects A3.
  for name,pose in f['poses'].items():
   if name.startswith('/World/RobotRight/')and name.rsplit('/',1)[-1]in('link6','link7','link8'):
    p=np.asarray(pose['position']);self.right_initial.setdefault(name,p.copy());self.right_max_motion=max(self.right_max_motion,float(np.linalg.norm(p-self.right_initial[name])))
  current_contacts=[]
  for c in f['contacts']:
   assert type(c['active'])is bool and math.isfinite(c['impulse_ns'])
   if not c['active']:continue
   pair=c['pair'];right=any(p.startswith('/World/RobotRight/')for p in pair)
   if right:
    if any(p==self.item or p.startswith(self.item+'/')for p in pair):self.right_item_contact_peak=max(self.right_item_contact_peak,c['impulse_ns'])
    if any(p in self.allgatepaths for p in pair):self.right_gate_contact_peak=max(self.right_gate_contact_peak,c['impulse_ns'])
   if any(p.startswith(('/World/RobotLeft/','/World/RobotRight/'))for p in pair):
    for g in self.m['gates']:
     if any(p in self.paths[g['id']]for p in pair):
      self.all_contact_peaks[g['id']]=max(self.all_contact_peaks.get(g['id'],0.),c['impulse_ns']);current_contacts.append((g['id'],c))
  if len(self.passed)<len(self.m['gates']):
   xyz=transform_vertices(self.vertices,f['poses'][self.item]);minx=float(xyz[:,0].min())
   for g in self.m['gates']:
    if g['id']not in self.passed and minx>=g['origin'][0]+g['half_thickness_m']:self.passed[g['id']]=i
  for g in self.m['gates']:
   gid=g['id'];t=max(0,i-1+self.phase)/120;schedule=self.schedules[gid];target=single_target(t,self.source)if schedule is None else schedule.target(t);prev=self.previous_targets.get(gid,target);self.previous_targets[gid]=target
   closing=target<prev-1e-12
   if closing and not self.closing.get(gid,False):self.done[gid]=False
   completed=False;blocked=False;reason=None
   if closing and not self.done.get(gid,False):
    if float(f['gates'][gid])<=g['closed_tolerance_m']:completed=True;reason='actual_reached_closed_tolerance'
    elif target<=1e-12:completed=True;blocked=True;reason='scheduled_close_end_but_blocked'
    if gid in self.passed and i>=self.passed[gid]and not any(e['gate_id']==gid for e in self.events):
     for contact_gid,c in current_contacts:
      if contact_gid==gid and c['impulse_ns']>0:self.contact_events.append(dict(tick=i,gate_id=gid,impulse_ns=c['impulse_ns'],pair=c['pair'],actual_position_m=f['gates'][gid],reconstructed_target_m=target))
   if completed:
    self.done[gid]=True
    if gid in self.passed and i>self.passed[gid]and not any(e['gate_id']==gid for e in self.events):
     geometry_by_margin={str(float(margin)):corridor_overrun(self.m,f['poses'],g,margin)for margin in CORRIDOR_MARGINS_M};hits=geometry_by_margin[str(float(self.margin))]
     self.events.append(dict(tick=i,gate_id=gid,item_pass_tick=self.passed[gid],reason=reason,blocked=blocked,actual_position_m=f['gates'][gid],reconstructed_target_m=target,near_slab_x_m=g['origin'][0]-g['half_thickness_m'],corridor_margin_m=self.margin,not_withdrawn=any(not h['unresolved_enclosure']for h in hits),unknown_withdrawal=any(h['unresolved_enclosure']for h in hits),hits=hits,geometry_by_margin=geometry_by_margin))
   self.closing[gid]=closing
 def result(self):
  return dict(passed_gate_ticks=self.passed,first_close_completion_events=self.events,closing_contact_events=self.contact_events,all_time_both_gate_contact_peak_by_gate=self.all_contact_peaks,initial_geometry=self.initial,right_max_wrist_motion_m=self.right_max_motion,right_item_contact_peak_ns=self.right_item_contact_peak,right_gate_contact_peak_ns=self.right_gate_contact_peak,targets_reconstructed=True,raw_goal_positions_recorded=False,sampling_rows=self.n)

def apply_A3(A2_value,audit,threshold_ns=DEFAULT_IMPULSE_NS,margin_m=CORRIDOR_MARGIN_M):
 if threshold_ns not in IMPULSE_THRESHOLDS or margin_m not in CORRIDOR_MARGINS_M:raise ValueError('unsupported A3 sensitivity')
 geometry=[];unknown=False
 for e in audit['first_close_completion_events']:
  hits=e['geometry_by_margin'][str(float(margin_m))]
  if any(not h['unresolved_enclosure']for h in hits):geometry.append(dict(e,selected_margin_m=margin_m,selected_hits=hits))
  unknown|=any(h['unresolved_enclosure']for h in hits)
 contacts=[e for e in audit['closing_contact_events']if e['impulse_ns']>0 and(threshold_ns==0 or e['impulse_ns']>=threshold_ns)]
 failed=bool(geometry or contacts)
 return dict(value=0. if failed else None if unknown and A2_value>0 else A2_value,unverified_withdrawal_geometry=unknown and not failed,withdrawal_failure=bool(geometry),closing_contact_failure=bool(contacts),threshold_ns=threshold_ns,margin_m=margin_m,geometry_events=geometry,first_closing_contact=contacts[0]if contacts else None,closing_contact_peak_ns=max((e['impulse_ns']for e in audit['closing_contact_events']),default=0.),all_time_contact_reference_would_zero=any(v>0 and(threshold_ns==0 or v>=threshold_ns)for v in audit['all_time_both_gate_contact_peak_by_gate'].values()))
