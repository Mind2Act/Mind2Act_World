"""CP09 single-board acquisition mechanics, shared X5 and sole shared clock."""
import json,math
from pathlib import Path
import numpy as np
from mind2act_eval.contracts import Health,NativeEvidence
from mind2act_eval.tasks._observation import observation
from . import CONTRACT
from .instances import validate,config,signature
from .audit import Audit
from .replay import Replay
from .geometry import descriptor,verify


class RestorationScene:
    def __init__(self,spec,ctx,sensors):
        validate(spec);self.spec,self.ctx,self.sensors=spec,ctx,sensors
        self.replay=Replay(spec.difficulty,spec.seed,sensors.case);self.tick=0
        if signature(self.replay.case)!=spec.config['source_case_sha256']:raise ValueError('CP09 selected source case differs')
        self.original_source={}
        if 'test_source' in spec.config:
            from .instances import original_case
            from mind2act_eval.original_test_inputs import audit_source
            if signature(sensors.case)!=signature(original_case(spec,ctx.assets)):
                raise ValueError('CP09 physical scene differs from original test geometry')
            self.original_source=audit_source(spec,ctx.assets)
        self.audit=Audit(Path(ctx.output_dir)/'CP09'/'raw-trace.jsonl')
        self.audit.append(dict(tick=0,contract=CONTRACT,source_case=sensors.case));self.complete=False

    def before_step(self,tick):
        if tick!=self.tick+1:raise ValueError('CP09 before-step clock differs')
        self.sensors.before_step(tick)

    def after_step(self,tick):
        if tick!=self.tick+1:raise ValueError('CP09 missing physics sample')
        row=self.sensors.after_step(tick);self.replay.append(row);self.audit.append(row);self.tick=tick
        # Raster metrics update only changed patches; no per-tick history replay.
        self.complete=bool(self.replay.result()['success'])

    def observe(self,tick):return observation(self.spec,self.ctx.robot,tick,self.spec.config['instruction'])
    def evidence(self,tick):
        if tick!=self.tick:raise ValueError('CP09 unsampled physical evidence')
        return NativeEvidence('CP09',self.spec.instance_id,tick,dict(difficulty=self.spec.difficulty,
            seed=self.spec.seed,audit=self.audit.receipt(),**self.original_source))
    def is_success(self):return self.complete
    is_complete=is_success
    def health(self):return Health(True)
    def result_metadata(self):
        return dict(geometry_decision=self.spec.config['geometry_decision'],dock_rule_status=self.spec.config['dock_rule_status'],
                    source_case_sha256=self.spec.config['source_case_sha256'],native_validation='not_run')
    def close(self):self.audit.close();self.sensors.close()


class NativeSensors:
    def __init__(self,env,stage,subscription,tools_module):
        self.env,self.stage,self.subscription=env,stage,subscription
        self.case=env.case;self.tools_module=tools_module;self.last_time=float(env.world.current_time)
        env.previous_finger_commands={s:q[6:].copy() for s,q in env.commands.items()}
        env.original_ink_after_erasing=None;env.interaction_events=[]

    def before_step(self,tick):
        from .mechanism.interaction import before_control
        env=self.env
        env.commands={s:np.asarray(r.get_joint_positions()).copy() for s,r in env.robots.items()}
        before_control(env);env._boards()

    def after_step(self,tick):
        from .mechanism.interaction import select_contacted_marker
        from .mechanism.retrieval_geometry import BODY_Z
        from .mechanism.props import write_ink_mesh
        env=self.env;now=float(env.world.current_time)
        if abs(now-self.last_time-1/120)>1e-7:raise ValueError('CP09 native clock differs')
        self.last_time=now;env.tick=tick
        verify(self.stage)
        self.robot.verify_task_collision_overlay(self.stage)
        for s,r in env.robots.items():env.gripper_opening[s]=float(np.mean(r.get_joint_positions()[6:]))
        pick_start=len(env.pickup_events);cap_start=len(env.caps.events)
        select_contacted_marker(env);env._measure_pickup();env.board_support.sample();env.caps.sample();env._boards()
        contacts=[]
        # Every actual tool may draw; active-hand selection is diagnostic.
        from scipy.spatial.transform import Rotation
        tools=[('eraser',env.active_pen,env.tools['eraser'])]+[
            ('pen',color,env.all_tools['pen'+str(color)]) for color in env.pen_verified]
        for tool,color,body in tools:
            p,q=body.get_world_pose();p=np.asarray(p)
            r=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
            root=env.tool_root_for(tool,color)+'/Contact'
            active=any(root in pair and env.board_path in pair and on and (impulse>1e-8 or sep<=0)
                       for pair,(on,impulse,sep) in env.contact_pairs.items())
            board_r=env.board_support.rotation();local=board_r.T@(p-env.board_support.position())
            if env.annotation_applied and active:
                env.board_support.record_ink(tool);rr=board_r.T@r;angle=math.atan2(rr[1,0],rr[0,0])
                changed=env.surface.contact(tool,local[:2],True,tick,angle,color=color)
                env.contact_counts[tool]+=1
                if tool=='pen':env.pen_contact_counts[color]+=1
                contacts.append(dict(tool=tool,color=color,xy_m=local[:2].tolist(),angle_rad=angle,
                    changed_pixels=changed,physx_contact=True,pickup_verified=bool(env.pickup_verified[tool]),
                    cap_open=bool(env.caps.is_open(color))))
        env.previous_finger_commands={s:q[6:].copy() for s,q in env.commands.items()}
        if (tick-1)%4==0:
            revision=(env.annotation_applied,env.surface.changed)
            if getattr(env,'last_ink_mesh_revision',None)!=revision:
                write_ink_mesh(env.w['ink_meshes']['Work'],env.surface.ink,env.board_surface_z+.00008)
                env.last_ink_mesh_revision=revision
        b=env.board_support;eraser_p=env._tool_pose('eraser')[0]
        eraser_support=any(any(q.startswith(env.tool_root('eraser')+'/') for q in pair)
            and any('/World/CP09/ToolTray' in q for q in pair) and active and (impulse>1e-8 or sep<.0002)
            for pair,(active,impulse,sep) in env.contact_pairs.items())
        return dict(tick=tick,simulation_time_s=now,annotation_applied=env.annotation_applied,
            board=dict(position_m=b.position().tolist(),rotation=b.rotation().tolist(),machine_offset_m=float(b.machine_offset),
                storage_clip_removed=not self.stage.GetPrimAtPath('/World/CP09/Board/StorageClip').IsValid(),
                foot_table_contact=bool(b._table_foot_contact()),latches=dict(b.latches),release_pending=dict(b.release_pending)),
            eraser=dict(position_m=eraser_p.tolist(),initial_position_m=env.initial_tool_positions['eraser'].tolist(),
                latched=bool(env.grasp_latches['eraser']),tray_contact=bool(eraser_support)),
            pickup_events=env.pickup_events[pick_start:],cap_events=[e for e in env.caps.events[cap_start:] if e['event']=='cap_pulled_open'],
            ink_contacts=contacts)

    def close(self):
        if self.subscription is not None:self.subscription.unsubscribe();self.subscription=None


def build(spec,ctx):
    row=validate(spec)
    from .binding import validate_station
    from importlib.resources import files
    import hashlib
    station_config_sha=hashlib.sha256(files(__package__).joinpath('config/station.json').read_bytes()).hexdigest()
    validate_station(row,ctx.assets,station_config_sha,spec=spec)
    # Shared owner must authorize and audit this exact collision-only exception.
    overlay=getattr(ctx.robot,'apply_task_collision_overlay',None)
    if overlay is None:raise RuntimeError('CP09 requires shared task collision overlay API; generic geometry is prohibited')
    from importlib import import_module
    from isaacsim.core.prims import SingleRigidPrim
    from omni.physx import get_physx_simulation_interface
    from .mechanism.props import build_props
    from .mechanism.environment import Environment
    from .poses import flange_readers
    from scipy.spatial.transform import Rotation
    stage,world=ctx.sim.stage,ctx.world
    ctx.robot.prepare_stage(stage)
    cfg=config('station');ctx.robot.mount_bases(stage,cfg['robots'],legacy_anchors=True)
    from pxr import Usd,UsdGeom,UsdPhysics
    # Same acquisition support collisions, without altering station geometry.
    for root in ('/World/Worktable','/World/TableFrame','/World/PedestalLeft','/World/PedestalRight','/World/CameraStand','/World/Floor'):
        prim=stage.GetPrimAtPath(root)
        if not prim:continue
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube):
                UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
                if p.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('none')
    overlay(stage,descriptor(spec.difficulty))
    verify(stage)
    output=Path(ctx.output_dir)/'CP09'/'private';output.mkdir(parents=True,exist_ok=True)
    if 'test_source' in spec.config:
        from .instances import original_case
        source_case=original_case(spec,ctx.assets)
    else:
        source_case=json.loads(ctx.assets.resolve('source_case').read_text())
    if signature(source_case)!=spec.config['source_case_sha256']:raise ValueError('CP09 exact acquisition case asset differs')
    task=build_props(stage,output,spec.difficulty,spec.seed,cfg,source_case)
    tools=import_module('.mechanism.'+spec.difficulty+'_tools',__package__)
    support=import_module('.mechanism.'+spec.difficulty+'_board_support',__package__)
    env=Environment(task,{}, {},world,tools)
    tools.build_tools(env);env.board_support=support.BoardSupport(env);env.caps=tools.PullCaps(env)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    env.board_support.register(world);env.caps.register(world);env.board_path='/World/CP09/Board/BoardContact'
    env.all_tools={'eraser':world.scene.add(SingleRigidPrim(tools.root_for('eraser'),name='CP09_Eraser'))}
    for color in env.pen_verified:
        env.all_tools['pen'+str(color)]=world.scene.add(SingleRigidPrim(tools.root_for('pen',color),name='CP09_Marker'+str(color)))
    env.tools={'eraser':env.all_tools['eraser'],'pen':env.all_tools['pen1']}
    views=flange_readers(stage,world)
    class MeasuredFrame:
        def __init__(self,view):self.view=view
        def fk(self,_q):
            positions,quats=self.view.get_world_poses();p=np.asarray(positions[0]);q=np.asarray(quats[0])
            r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();return p+r@np.array([.145,0,0]),r
    env.kin={s:MeasuredFrame(v) for s,v in views.items()}
    ctx.robot.initialize(world);env.robots=ctx.robot.articulations(world)
    from mind2act_eval.contracts import Action
    from mind2act_eval.robot import normalized_observation
    rest=ctx.robot.rest_pose()
    if not np.allclose(rest,spec.config['robot_rest'],rtol=0,atol=1e-12):raise ValueError('CP09 source rest differs')
    hold=Action(rest[:6],rest[:6],normalized_observation(rest[6]),normalized_observation(rest[6]))
    for _ in range(120):ctx.robot.apply(hold);ctx.sim.step(render=False)
    env.initial_pen_positions={c:np.asarray(env.all_tools['pen'+str(c)].get_world_pose()[0]).copy() for c in env.pen_verified}
    env.initial_tool_positions={'eraser':env._tool_pose('eraser')[0].copy(),'pen':env.initial_pen_positions[1].copy()}
    if env._tool_attachment_paths():raise ValueError('CP09 tools must start unattached')
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(env._contact_report)
    sensors=NativeSensors(env,stage,subscription,tools)
    sensors.robot=ctx.robot
    try:return RestorationScene(spec,ctx,sensors)
    except BaseException:sensors.close();raise
