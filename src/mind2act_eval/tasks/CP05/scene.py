"""CP05 native scene: authored stage, measured key edges and actual rack support.

Selected Easy/Medium/Hard runtime sources are identical. No expert imports.
"""
from types import SimpleNamespace
from collections import deque
import numpy as np
from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import NativeEvidence, Observation, Health
from .mechanism.keyboard_clock import KeyboardClock
from .mechanism.key_press_policy import KeyPressPolicy, PRESS_M, RELEASE_M
from .mechanism.stability import StablePlacement
from .mechanism.replay_events import ReplayEvents


def _mechanism(w, setup):
    level=setup['level'];sides=list(w.tools);root='/World/CP12'
    paths={s:root+'/Tools/'+s+'/Mallet' for s in sides}
    w.allowed_contacts=list(paths.values())+[root+'/VolumeCap','/World/DustCloth']
    clock=KeyboardClock(setup['sequence'],press_depth_m=PRESS_M,release_depth_m=RELEASE_M);w.clock=clock
    key_press_policy=getattr(w,'key_press_policy',None) or KeyPressPolicy(w.keys);w.key_press_policy=key_press_policy
    initial={s:w.tools[s].get_world_pose()[0].copy() for s in sides}
    gates={s:StablePlacement(120,.003,require_manipulation=False) for s in sides}
    pickup={};returned={};cloth_history=deque(maxlen=61)
    state=dict(cloth_bimanual=not setup['cloth'],cloth_front=not setup['cloth'],volume_contact=False,
               volume_max=not setup['volume_required'],cloth_evidence_available=not setup['cloth'])
    materials={k:w.shared.material('FormalCP05_'+k,c,rough=.5) for k,c in
               [('white',(.89,.87,.8)),('cyan',(.02,.8,.85)),('green',(.05,.75,.15)),('yellow',(.9,.65,.02))]}
    # Each mechanism belongs to one scene/stage; a fresh instance starts unbound.
    material_bindings={}
    def bind_material(path,color):
        binding,previous=material_bindings.get(path,(None,None))
        if previous==color:
            return
        if binding is None:
            binding=w.UsdShade.MaterialBindingAPI.Apply(w.stage.GetPrimAtPath(path))
        binding.Bind(materials[color])
        material_bindings[path]=(binding,color)
    # Visual feedback follows measured held-tip contact on a depressed key.
    # Scored new-press events remain exclusively owned by KeyboardClock.
    visual_pressed=set()
    def finger_contact(path,side):
        return any(w.touch(path,other='/World/Robot'+side,link=link) for link in ('link7','link8'))
    def sample(step):
        vpath=root+'/VolumeCap'
        from .mechanism.volume import volume_reached_max
        contact_now=any(finger_contact(vpath,s) for s in sides)
        state['volume_contact'] |= contact_now
        at_max=volume_reached_max(w.volume.get_world_pose()[0],w.control_positions['VolumeCap'],
            setup.get('volume_axis','Y'),setup.get('volume_travel_m',.07),True)
        released_now=not w.touch(vpath,robot_only=True)
        state['volume_max']=(not setup['volume_required'] or bool(at_max))
        ready=(not setup['cloth'] or state['cloth_bimanual'] and state['cloth_front']) and (not setup['volume_required'] or state['volume_max'])
        held={s:any(finger_contact(paths[s],arm) for arm in w.robots) for s in sides}
        for s in sides:
            pos=w.tools[s].get_world_pose()[0]
            if ready and held[s] and np.linalg.norm(pos-initial[s])>.01:pickup.setdefault(s,step/120)
        synced=len(pickup)==len(sides) and (len(sides)==1 or max(pickup.values())-min(pickup.values())<=.5)
        authored_depths={i:.822-float(b.get_world_pose()[0][2]) for i,b in w.keys.items()}
        tips={i for i in w.keys if any(held[s] and w.touch(paths[s]+'/SoftTip',other=f'{root}/Keyboard/Keys/White{i:02d}') for s in sides)}
        fingers={i for i in w.keys if w.touch(f'{root}/Keyboard/Keys/White{i:02d}',robot_only=True)}
        tool_key_contact=any(w.touch(path,other=f'{root}/Keyboard/Keys/White{i:02d}') for path in paths.values() for i in w.keys)
        key_press_policy.observe_idle(authored_depths,clear=not fingers and not tool_key_contact)
        # During explicit fresh idle calibration, key event inputs are suppressed.
        # Physical depths and readiness are independently recorded by qualification.
        depths=key_press_policy.relative_depths(authored_depths) if key_press_policy.ready else {i:0. for i in w.keys}
        clock.update(step,button_depth=w.control_positions['StartFace'][2]-float(w.button.get_world_pose()[0][2]),
                     button_tip=False,
                     depths=depths,tips=tips if ready else set(),finger_keys=fingers,
                     missing_tools=tuple(s for s in sides if not held[s]))
        # A shaft can depress a key before the held nib reaches it. The scoring
        # edge stays invalid, but later real nib contact must still be visible.
        # Rearm visual feedback only after measured release, never each tick.
        visual_pressed.intersection_update(i for i,d in depths.items() if d>=RELEASE_M)
        feedback_keys={i for i in depths if depths[i]>=PRESS_M}
        if clock.state=='replay':
            for i in feedback_keys-visual_pressed:
                clock.flash_until[i]=max(clock.flash_until.get(i,0),step+30)
                visual_pressed.add(i)
        for s in sides:
            pos,quaternion=w.tools[s].get_world_pose();released=not w.touch(paths[s],robot_only=True)
            supported=w.supported(paths[s],other=root+'/Tools/'+s)
            gates[s].update(pos,bilateral=False,elevated=False,released=released,supported=supported,
                           correct=w.rack_contains(s,pos,quaternion))
            if gates[s].complete:returned.setdefault(s,step/120)
            else:returned.pop(s,None)
        key=clock.cue_key(step)
        for i in w.keys:
            color='cyan' if i==key else ('green' if clock.flash_until.get(i,0)>step else 'white')
            bind_material(f'{root}/Keyboard/Keys/White{i:02d}/Cap',color)
        bind_material(root+'/StatusLamp','green' if clock.state=='replay' else 'yellow')
    return sample, state, gates, paths


def _material(stage, name, color, metal=0, rough=.45):
    from pxr import UsdShade, Sdf, Gf
    material = UsdShade.Material.Define(stage, '/World/Materials/' + name)
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Shader'))
    shader.CreateIdAttr('UsdPreviewSurface')
    for key, typ, value in [('diffuseColor', Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
                            ('metallic', Sdf.ValueTypeNames.Float, metal),
                            ('roughness', Sdf.ValueTypeNames.Float, rough)]:
        shader.CreateInput(key, typ).Set(value)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


class PianoScene:
    def __init__(self, spec, ctx, w):
        self.spec, self.ctx, self.w = spec, ctx, w
        self._sample, self.state, self.gates, self.paths = _mechanism(w, spec.config['setup'])
        self.events = ReplayEvents()
        self.volume_tick = None
        update = w.clock.update
        def record(step, **kwargs):
            before = set(w.clock.down)
            update(step, **kwargs)
            fingers = set(kwargs.get('finger_keys', ()))
            self.events.observe(step, phase=w.clock.state, before_down=before,
                after_down=set(w.clock.down), depths=kwargs['depths'], tips=kwargs['tips'],
                finger_keys=fingers, tool_category=lambda key, valid:
                    'other' if key in fingers else self._tool_id(key, valid))
        w.clock.update = record
        self.tick = 0

    def _tool_id(self, key, valid):
        w = self.w
        for side, path in self.paths.items():
            held = any(w.touch(path, other='/World/Robot'+arm, link=link)
                       for arm in w.robots for link in ('link7', 'link8'))
            if held and w.touch(path + ('/SoftTip' if valid else ''),
                                other=f'/World/CP12/Keyboard/Keys/White{key:02d}'):
                return 'single' if self.spec.difficulty == 'easy' else side.lower()
        return 'other'

    def before_step(self, tick):
        pass

    def after_step(self, tick):
        self.tick = tick
        self._sample(tick)
        if self.spec.difficulty == 'hard' and self.volume_tick is None and self.state['volume_max']:
            self.volume_tick = tick

    def observe(self, tick):
        return Observation(tick, tick/120, **self.ctx.robot.observe(),
            instruction=public_instruction(self.spec), protocol_version=self.spec.protocol_version)

    def evidence(self, tick):
        w = self.w
        play = w.clock.play_tick
        done = w.clock.state == 'replay'
        done_tick = w.clock.cue_start + len(w.clock.sequence)*w.clock.period_ticks if done else None
        e = dict(triggered_play=play is not None, play_tick=play,
                 demonstration_completed=done, demonstration_completed_tick=done_tick,
                 volume='MAX' if self.state['volume_max'] else 'MIN',
                 volume_max_tick=self.volume_tick,
                 replay_events=self.events.events, failures=w.clock.failures,
                 all_keys_released=not w.clock.down,
                 arms_withdrawn=all(not w.touch('/World/CP12', other='/World/Robot'+s) for s in w.robots),
                 tools=[dict(tool_id='single' if self.spec.difficulty == 'easy' else s.lower(),
                             returned=self.gates[s].complete,
                             released=not w.touch(p, robot_only=True)) for s,p in self.paths.items()])
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
            dict(difficulty=self.spec.difficulty, target_keys=self.spec.config['setup']['sequence'],
                 normal_termination=True, cp05_evidence=e))

    def events_complete(self):
        e=self.evidence(self.tick).values['cp05_evidence']
        from .scorer import TOOLS
        from collections import Counter
        tools=e['tools']
        return bool(e['demonstration_completed']
            and len(e['replay_events'])>=len(self.spec.config['setup']['sequence'])
            and Counter(t['tool_id'] for t in tools)==Counter(TOOLS[self.spec.difficulty])
            and all(t['returned'] for t in tools))

    is_complete = events_complete

    def health(self):
        return Health(True)

    def result_metadata(self):
        return dict(key_calibration=self.w.key_press_policy.snapshot(),
                    initialization_physics_ticks=getattr(self.w,'initialization_physics_ticks',0), cue_period_ticks=self.w.clock.period_ticks)

    def close(self):
        pass


def build(spec, ctx):
    if spec.clock.physics_hz != 120 or spec.clock.control_hz not in (30, 120) or spec.clock.rgb_hz != 30:
        raise ValueError('CP05 requires 120 Hz physics, explicit 30/120 Hz control and 30 Hz RGB')
    setup = spec.config['setup']
    if setup['level'] != spec.difficulty or len(setup['sequence']) != (12 if spec.difficulty == 'hard' else 8):
        raise ValueError('CP05 tier/sequence mismatch')
    if setup['cloth'] or setup['lid'] or not setup['asset_package_scene']:
        raise ValueError('CP05 selected baseline requires authored no-cloth/no-lid scene')
    from pxr import Usd, UsdGeom, UsdShade, Gf, PhysxSchema, UsdPhysics
    from isaacsim.core.prims import SingleRigidPrim
    stage = ctx.sim.stage
    ctx.robot.prepare_stage(stage)
    ctx.robot.attach_cameras(stage)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/CP12') and (prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI)):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    robots = ctx.robot.articulations(world)
    def rigid(path, name):
        if not stage.GetPrimAtPath(path):
            raise ValueError('CP05 missing native prim: ' + path)
        return world.scene.add(SingleRigidPrim(prim_path=path, name=name))
    sides = ('Left',) if spec.difficulty == 'easy' else ('Left','Right')
    tools = {s:rigid('/World/CP12/Tools/'+s+'/Mallet', 'mallet'+s) for s in sides}
    button = rigid('/World/CP12/StartFace', 'start_button')
    keys = {i:rigid(f'/World/CP12/Keyboard/Keys/White{i:02d}', f'key{i}') for i in range(29)}
    volume = rigid('/World/CP12/VolumeCap', 'volume_slider')
    contacts = ctx.sim.contacts
    from .mechanism.rack import RackGeometry
    rack=RackGeometry(stage,sides)
    world.reset()
    ctx.robot.reset_to_rest()
    ctx.robot.verify(robots, stage)
    control_positions = {'StartFace':button.get_world_pose()[0].copy(), 'VolumeCap':volume.get_world_pose()[0].copy()}
    # Establish sensor rest depths during initialization, before model T0.
    # Live notice actions can never block calibration or key feedback.
    key_press_policy = KeyPressPolicy(keys)
    initialization_physics_ticks = 0
    while not key_press_policy.ready:
        world.step(render=False)
        initialization_physics_ticks += 1
        key_press_policy.observe_idle({i:.822-float(b.get_world_pose()[0][2]) for i,b in keys.items()},
            clear=not any(contacts.touch(f'/World/CP12/Keyboard/Keys/White{i:02d}',robot_only=True) for i in keys))
    contacts.clear()
    shared = SimpleNamespace(material=lambda name,color,metal=0,rough=.45: _material(stage,name,color,metal,rough))
    w = SimpleNamespace(stage=stage, robots=robots, tools=tools, button=button, keys=keys,
                        volume=volume, control_positions=control_positions, touch=contacts.touch, supported=contacts.supported,
                        key_press_policy=key_press_policy,initialization_physics_ticks=initialization_physics_ticks,
                        shared=shared, UsdShade=UsdShade, clothmesh=None,rack_contains=rack.contains)
    return PianoScene(spec, ctx, w)
