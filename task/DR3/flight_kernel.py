"""Private candidate: S9 exact motion equations; final transport segment qualification added."""
import math
DT=1/120

def norm(a):return math.sqrt(sum(x*x for x in a))
def linefit(t,y):
 mt=sum(t)/len(t);my=sum(y)/len(y);den=sum((x-mt)**2 for x in t);b=sum((x-mt)*(z-my) for x,z in zip(t,y))/den if den else 0;a=my-b*mt
 return a,b,math.sqrt(sum((z-a-b*x)**2 for x,z in zip(t,y))/len(t))

def detect_record(record):
 out={"events":[]}
 lastrobot=lastother=lasttray=-1;prevtray=False;last_transport_touchdown=None
 for i,z in enumerate(record):
  if z['tray'] and not prevtray:
   lastbase=max(lastrobot,lastother,lasttray);start=lastbase+1
   if lastbase>=0 and lastbase!=lasttray:last_transport_touchdown=i
   # Tray bounces after delivery are not a new launch into the tray.
   if start<i and lastbase!=lasttray and lastbase>=0:
    segment=record[start:i+1];duration=(i-start)*DT;delta=[z['p'][j]-record[start]['p'][j] for j in range(3)];distance=norm(delta[:2]);peak=max(w['speed'] for w in segment);runs=[];begin=None
    for k in range(start,i):
     if record[k]['free'] and begin is None:begin=k
     if begin is not None and (not record[k]['free'] or k==i-1):
      end=k if record[k]['free'] else k-1
      if end-begin+1>=3:
       tt=[(a-begin)*DT for a in range(begin,end+1)];positions=[record[a]['p'] for a in range(begin,end+1)];fx=linefit(tt,[a[0] for a in positions]);fy=linefit(tt,[a[1] for a in positions]);fz=linefit(tt,[a[2]+.5*9.81*t*t for a,t in zip(positions,tt)])
       runs.append(dict(start_tick=begin,end_tick=end,samples=end-begin+1,fixed_g_z_rms_m=fz[2],xy_rms_m=norm([fx[2],fy[2]]),fitted_vxy_m_s=norm([fx[1],fy[1]])))
      begin=None
    compatible=[a for a in runs if a['fixed_g_z_rms_m']<=.0025 and a['xy_rms_m']<=.0025]
    geom=duration>=.05-1e-9 and distance>=.05 and distance/duration>=.35 and bool(compatible)
    if peak>=.5 and distance>=.02 and duration>=2*DT:
     before=record[max(0,start-6):start];recent_robot=any(a['robot'] for a in before);prior_gate=any(a['gate'] and a['gate_impulse']>1e-5 for a in before);midgate=[a['tick'] for a in segment if a['gate'] and a['gate_impulse']>1e-5]
     cause='robot_release' if recent_robot and not prior_gate else 'gate_contact_launch_candidate' if prior_gate and not recent_robot else 'mixed_robot_gate' if recent_robot and prior_gate else 'other_or_unresolved'
     ev=dict(start_tick=start,last_prelaunch_contact_tick=lastbase,touchdown_tick=i,duration_s=duration,horizontal_distance_m=distance,displacement_3d_m=norm(delta),peak_speed_m_s=peak,peak_tick=max(segment,key=lambda w:w['speed'])['tick'],start_z_m=record[start]['p'][2],end_z_m=z['p'][2],free_runs=runs,ballistic_compatible=bool(compatible),kinematic_criteria_without_peak=geom,flag_050=bool(geom and peak>=.5),flag_075=bool(geom and peak>=.75),flag_100=bool(geom and peak>=1.),launch_context=cause,gate_contact_ticks_during_flight=midgate,last_robot_contact_tick=lastrobot,terminal_tray_contact_up=z['up'])
     out['events'].append(ev)
  if z['robot']:lastrobot=i
  if z['other']:lastother=i
  if z['tray']:lasttray=i
  prevtray=z['tray']
 return dict(out,last_transport_touchdown_tick=last_transport_touchdown)

def from_rows(rows,manifest):
 item=manifest["item_root"];gates=set(manifest["gate_colliders"]+manifest["frame_colliders"]);trays=set(manifest["tray_colliders"]);record=[]
 for row in rows:
  tick=row["physics_tick"];det=row["detector"];pose=det["poses"][item];rc=gc=tc=oc=False;gi=ri=0.;up=False
  if tick!=len(record) or det.get("contacts_complete") is not True:raise ValueError("incomplete contact clock")
  for c in det["contacts"]:
   if not c["active"] or not any(x==item or x.startswith(item+"/") for x in c["pair"]):continue
   other=next((x for x in c["pair"] if not(x==item or x.startswith(item+"/"))),"")
   if other.startswith("/World/Robot"):rc=True;ri+=c["impulse_ns"]
   elif other in gates:gc=True;gi+=c["impulse_ns"]
   elif other in trays:tc=True;up|=any(n>0 for n in c["item_upward_normals"])
   else:oc=True
  record.append(dict(tick=tick,p=pose["position"],speed=pose["linear_speed_m_s"],robot=rc,gate=gc,tray=tc,other=oc,free=not(rc or gc or tc or oc),gate_impulse=gi,robot_impulse=ri,up=up))
 return detect_record(record)
