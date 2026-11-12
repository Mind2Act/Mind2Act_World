"""Native CP04 passive scene, ported from cp04-b-v42-delivery.

Formal evaluation uses one 120 Hz solve per logical event. The explicit
historical source profile retains eight 960 Hz solves. Robot
initialization and actions belong exclusively to the shared robot service.
"""
import numpy as np
from mind2act_eval.tasks._instructions import public_instruction
from mind2act_eval.contracts import NativeEvidence, Observation, Health
from .binding import bind_recipe
from .clock import CookingClock, LOGICAL_HZ, resolve


def contact_free(touching, *body_paths):
    """Release is absence of physical contact, without a TCP-distance gate."""
    return not any(touching(path) for path in body_paths)


def _enable_task_contact_reports(stage):
    """Enable reports on cooking bodies; robot reports belong to robot/."""
    from pxr import UsdPhysics, PhysxSchema
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Cooking/') and (
                prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


def _initialize(spec, ctx, binding):
    from .mechanism.contact_contract import apply
    stage = ctx.sim.stage
    C = dict(spec.config)
    clock = resolve(spec)
    ctx.robot.prepare_stage(stage)
    ctx.robot.attach_cameras(stage)
    apply(stage)
    _enable_task_contact_reports(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    from pxr import UsdGeom, UsdPhysics, PhysicsSchemaTools
    from isaacsim.core.prims import SingleRigidPrim, RigidPrim
    from .mechanism.multi_jars import JarSet
    from .mechanism.appliance import Appliance
    from .mechanism.thermal import ThermalState
    from .mechanism.observer import Observer, IDS
    robots = ctx.robot.articulations(world)
    def rigid(path, name):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError('native candidate rigid missing: ' + path)
        return world.scene.add(SingleRigidPrim(prim_path=path, name=name))
    jars = JarSet.__new__(JarSet)
    jars.items = {}; jars.food_paths = []; jars.stage = stage; jars.enable_recap = False
    jars.implicit_friction = True; jars.physics_dt = 1/clock.physics_hz
    root = '/World/Cooking/'
    for i in range(4):
        body = root + ('Jar0' if i == 3 else 'Condiment'+str(i))
        cap = root + ('TrialLid' if i == 3 else 'CondimentCap'+str(i))
        guide = root + ('ScrewGuide' if i == 3 else 'CondimentScrew'+str(i))
        jars.add_primary(i, rigid(body,'jar'+str(i)), rigid(cap,'cap'+str(i)),
            UsdPhysics.Joint(stage.GetPrimAtPath(guide)), body, cap)
    jars.build_batch(world)
    food_paths = [str(p.GetPath()) for p in stage.Traverse()
        if '/LooseFood' in str(p.GetPath()) and p.GetName().startswith('Piece') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(food_paths) != 12:
        raise RuntimeError('expected 4 x 3 native food pieces: '+str(food_paths))
    food = world.scene.add(RigidPrim(prim_paths_expr=food_paths, name='food_observer'))
    drawer = rigid(root+'MovingDrawer','drawer')
    rails = rigid(root+'MovingMiddleRails','middle_rails')
    lid = rigid(root+'PotLid','pot_lid')
    from mind2act_eval.robot.read_only import robot_body_readers
    flange_readers = robot_body_readers(stage, {'flange'+s: '/World/Robot'+s+'/Asset/link6' for s in robots}, world=world)
    flanges = {s: flange_readers['flange'+s] for s in robots}
    appliance = Appliance.__new__(Appliance)
    appliance.stage = stage; appliance.powered = False; appliance.armed = True
    appliance.press_count = 0; appliance.trace = []; appliance.mode = 'HEAT'
    appliance.thermal = ThermalState(binding.thermal_difficulty,
                                    thermal_profile_version=binding.thermal_profile_version)
    appliance.button = rigid(root+'PowerPlunger','power_plunger')
    appliance.knob = rigid(root+'Knob','thermal_knob')
    appliance.power_origin = np.asarray(stage.GetPrimAtPath(root+'PowerSpring').GetAttribute('physics:localPos0').Get(), dtype=float)
    from .mechanism.display import attach
    appliance.display = attach(stage, binding.thermal_difficulty, binding.thermal_profile_version)
    hidden_faces = [p for p in stage.Traverse() if p.GetName().endswith('Hidden') and 'Recipe' in str(p.GetPath())]
    if len(hidden_faces) != 1:
        raise RuntimeError('training recipe visible/hidden pair missing')
    hidden = hidden_faces[0]
    visible = stage.GetPrimAtPath(str(hidden.GetPath())[:-6])
    UsdGeom.Imageable(hidden).MakeInvisible(); UsdGeom.Imageable(visible).MakeVisible()
    # CP04 baseline counts active contact pairs without the shared CP01
    # impulse filter. Preserve that predicate for release measurements.
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    active_contacts = set()
    support_edges = set()
    from .mechanism.containment import Containers, support_reaches, support_directions
    containers = Containers(stage)
    def receive(headers, data):
        for header in headers:
            pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(int(x)))
                                for x in (header.collider0, header.collider1)))
            directed = tuple(str(PhysicsSchemaTools.intToSdfPath(int(x)))
                             for x in (header.collider0, header.collider1))
            if header.type == ContactEventType.CONTACT_LOST:
                support_edges.discard(directed); support_edges.discard(tuple(reversed(directed)))
                active_contacts.discard(pair)
            else:
                active_contacts.add(pair)
                points = data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]
                if points:
                    support_edges.discard(directed); support_edges.discard(tuple(reversed(directed)))
                for point in points:
                    support_edges.update(support_directions(*directed,point.normal,point.impulse,float(point.separation)))
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(receive)
    world.reset()
    ctx.robot.initialize(world)
    ctx.robot.reset_to_rest()
    # Preserve the baseline 0.1 s pre-roll before logical tick zero.
    for _ in range(clock.initialization_ticks):
        world.step(render=False)
    ctx.robot.verify(robots, stage)
    initial = {IDS[i]: jars.pose(item['jar'])[0].tolist() for i,item in jars.items.items()}
    observer = Observer(C, initial)
    counter = {'step': 0}
    record = {'initial_drawer_pose': drawer.get_world_pose()[0].tolist()}
    from .mechanism.containment import DrawerSlots
    drawer_slots=DrawerSlots(stage,initial,record['initial_drawer_pose'])
    jar_centers={IDS[i]:np.asarray(containers.cache.ComputeUntransformedBound(
        stage.GetPrimAtPath(item['body_path'])).ComputeAlignedRange().GetMidpoint()) for i,item in jars.items.items()}
    def touching(path):
        return any(any(x.startswith('/World/Robot') for x in pair)
                   and any(x.startswith(path) for x in pair) for pair in active_contacts)
    def sample():
        dp = drawer.get_world_pose()[0]
        measured = {'drawer':dp.tolist(),'jars':{},'food':{name:[] for name in IDS},
                    'food_in_pot':{name:[] for name in IDS}}
        fp,_ = food.get_world_poses()
        for path,pos in zip(food.prim_paths, fp):
            i = int(path.split('/LooseFood')[1].split('/')[0])
            measured['food'][IDS[i]].append(pos.tolist())
            measured['food_in_pot'][IDS[i]].append(bool(containers.pot_contains(pos)
                and support_reaches(path,root+'PotCollision',support_edges)))
        for i,item in jars.items.items():
            bp,br = jars.pose(item['jar']); cp,cr = jars.pose(item['lid'])
            center=bp+br@jar_centers[IDS[i]]
            measured['jars'][IDS[i]] = dict(body=bp.tolist(),cap=cp.tolist(),
                body_speed=float(np.linalg.norm(item['jar'].get_linear_velocity())),
                body_angular_speed=float(np.linalg.norm(item['jar'].get_angular_velocity())),
                cap_speed=float(np.linalg.norm(item['lid'].get_linear_velocity())),
                cap_angular_speed=float(np.linalg.norm(item['lid'].get_angular_velocity())),
                tilt_deg=float(np.degrees(np.arccos(np.clip(br[2,2],-1,1)))),
                detached=bool(item['released']),
                body_released=contact_free(touching,item['body_path']), cap_released=contact_free(touching,item['lid_path']),
                in_drawer=bool(drawer_slots.contains(IDS[i],center,dp)
                    and (support_reaches(item['body_path'],root+'MovingDrawer',support_edges)
                         or support_reaches(item['body_path'],root+'MovingMiddleRails',support_edges))),
                in_tray=bool(support_reaches(item['lid_path'],root+'CapTray',support_edges)
                    and all(lo <= c <= hi for c,lo,hi in zip(cp[:2],containers.tray_lower,containers.tray_upper))))
        lp,lq = lid.get_world_pose()
        measured.update(pot_lid_supported=bool(support_reaches(root+'PotLid',root+'PotCollision',support_edges)
                and containers.pot_contains(lp,rim=True)),
            pot_lid_released=contact_free(touching,root+'PotLid'),
            pot_lid_stable=bool(np.linalg.norm(lid.get_linear_velocity())<.015 and np.linalg.norm(lid.get_angular_velocity())<.15),
            controls_released=contact_free(touching,root+'PowerPlunger',root+'Knob'))
        observer.sample(counter['step'], measured)
    observer.thermal_sample(0, appliance.thermal.snapshot()); sample()
    return locals()


class CookingScene:
    def __init__(self, spec, ctx, native):
        self.spec, self.ctx, self.native = spec, ctx, native
        self.clock = resolve(spec) if spec is not None else CookingClock()

    def before_step(self, tick):
        from .mechanism.contact_contract import sync_release_contacts
        from .mechanism.drawer_physics import apply_rail_detent
        n = self.native
        n['jars'].pre_step()
        sync_release_contacts(n['stage'], n['jars'])
        n['appliance'].pre_step()
        drawer = n['drawer']
        apply_rail_detent(drawer, n['rails'])
        y, vy = float(drawer.get_world_pose()[0][1]), float(drawer.get_linear_velocity()[1])
        if y <= -.36 + .060 and (vy > 1e-4 or (abs(vy) <= 1e-4 and y <= -.355)):
            force = -min(4., 32. * max(vy, 0.) + 1.5)
            drawer._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.array([[0., force, 0.]], np.float32), is_global=True)

    def after_step(self, tick):
        if tick % self.clock.substeps:
            return
        n = self.native
        logical = tick // self.clock.substeps
        n['counter']['step'] = logical
        row = n['appliance'].post_step(1/LOGICAL_HZ, update_display=logical % 4 == 0)
        n['observer'].thermal_sample(logical, row)
        if logical == 420:
            from .mechanism.recipe_screen import show_ready
            show_ready(n['visible'], n['hidden'])
            n['observer'].hidden = logical
        if logical % 4 == 0:
            n['sample']()

    def observe(self, tick):
        return Observation(tick, tick / self.clock.physics_hz, **self.ctx.robot.observe(),
                           instruction=public_instruction(self.spec), protocol_version=self.spec.protocol_version)

    def evidence(self, tick):
        if self.native['observer'].latest_sample_tick != self.native['counter']['step']:
            self.native['sample']()
        return NativeEvidence(self.spec.task_id, self.spec.instance_id, tick,
            dict(difficulty=self.spec.difficulty, recipe=self.spec.config['recipe'],
                 normal_termination=True, cp04_evidence=self.native['observer'].evidence()))

    def events_complete(self):
        e=self.evidence(self.native['counter']['step']*self.clock.substeps).values['cp04_evidence']
        actions=e['thermal']['events']
        off=bool(actions and actions[-1]['mode']=='OFF' and any(r['mode']!='OFF' for r in actions))
        terminal=e['terminal_measurements']
        jars=terminal['jars'] if terminal else {}
        return bool(off and all(n in jars and jars[n]['in_drawer'] and jars[n]['in_tray']
                               and jars[n]['detached'] for n in self.spec.config['recipe']))

    is_complete = events_complete

    def health(self):
        return Health(True)

    def result_metadata(self):
        return dict(**self.clock.receipt(),
                    physics_dt_s=1/self.clock.physics_hz, logical_dt_s=1/LOGICAL_HZ,
                    control_hz=self.spec.clock.control_hz if self.spec is not None else 30,
                    ticks_per_control=self.spec.clock.ticks_per_action if self.spec is not None else 4,
                    geometry_profile='shortturn413_30')

    def close(self):
        self.native['subscription'] = None


def build(spec, ctx):
    clock = resolve(spec)
    binding = bind_recipe(spec)
    if tuple(spec.config['recipe']) != binding.recipe:
        raise ValueError('CP04 recipe names differ from tier-bound ingredient indices')
    # Shared command-reference policy, not a task-local driver/target clip.
    control = ctx.robot.configure_arm_reference_slew(5.0)
    if (control['physics_hz'] != clock.physics_hz or control['arm_reference_slew_rad_s'] != 5.0
            or control['per_physics_tick_delta_limit_rad'] != 5.0 / clock.physics_hz
            or control['native_parameter'] is not False):
        raise ValueError(f'CP04 requires shared 5 rad/s command-reference slew at {clock.physics_hz} Hz')
    return CookingScene(spec, ctx, _initialize(spec, ctx, binding))
