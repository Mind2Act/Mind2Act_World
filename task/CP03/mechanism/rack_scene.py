# ruff: noqa: F401
"""CP03 rack/selector scene construction.

Source: cp03-em-b-20261028 and cp03-hard-b-20261028 cases_unified/CP03/scene.py
(byte-identical in both), frozen from project-v8/scripts/physco_dish_collection_replay.py,
SHA256 192b8e65... The scene USD (red YCB029 texture variant) is the prepared station
asset opened by ctx.sim; this file only modifies it as the baseline did.

Structural changes only: no SimulationApp / open_stage / World construction (ctx.sim),
X5 stage work and rest/park writes go through ctx.robot, the redundant post-reset gain
block is dropped (its kp 4400 / kd 40 / vmax 5 / armature .01 / effort 100 equal the
shared official X5 profile and are verified by ctx.robot.verify).
"""
from pathlib import Path
import numpy as np
from mind2act_eval.sim.stage_authoring import copy_local_prim, remove_composed_prim


def create(M, C, setup, ctx):
    from types import SimpleNamespace
    a=SimpleNamespace(output=str(ctx.output_dir),level=C["difficulty"])
    O=Path(a.output);O.mkdir(parents=True,exist_ok=True)
    params=setup['parameters'];S=dict(M['levels'][a.level])
    S['empty_slot_indices']=params['empty_slot_indices']
    a.rail_origin=params['rail_origin_m'];a.transport_speed=S['speed']
    a.static_rack=a.pick_only=a.settle_only=a.controls_only=False
    a.controls=True;a.grip_force=40.;a.grip_stiffness=2300.
    motor_start=None;selector_speeds=M.get('selector_speeds_m_s')
    pivot_selector=M.get('version') in ['washer_v19_recessed_full_drum_preview','washer_v20_panel_spacing_preview'] and a.level!='easy'
    if M.get('selector_type')=='side_pivot_lever_preview' and a.level!='easy' and not pivot_selector:
        raise ValueError('Unvalidated selector type')
    DT=1/120;BASE='/World/DishLoading'
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf, PhysicsSchemaTools
    from isaacsim.core.prims import SingleRigidPrim
    from scipy.spatial.transform import Rotation

    stage = ctx.sim.stage
    # No visible progress counter, including when loading an older scene.
    if stage.GetPrimAtPath(BASE+'/Panel/Status'):
        remove_composed_prim(stage, BASE+'/Panel/Status')
    source_xf = UsdGeom.Xformable(stage.GetPrimAtPath(BASE+'/Source'))
    source_shift = Gf.Vec3d(params['source_dx_m'], params['source_dy_m'], 0)
    # Current recorded CP03 scenes already contain the Source transform op.
    # Reuse it instead of adding a duplicate USD xformOp on every reset.
    source_ops = source_xf.GetOrderedXformOps()
    if source_ops:
        source_ops[0].Set(source_shift)
    else:
        source_xf.AddTranslateOp().Set(source_shift)
    material = UsdShade.Material.Define(stage, '/World/DishPhysicsMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(.85)
    pm.CreateDynamicFrictionAttr(.70)
    pm.CreateRestitutionAttr(0.)

    def collider(prim, dynamic=False):
        authored=prim.GetAttribute('physics:collisionEnabled')
        if authored.HasAuthoredValueOpinion() and authored.Get() is False:
            return
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull' if dynamic else 'none')
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(.0004)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')

    carriage = stage.GetPrimAtPath(BASE+'/Carriage')
    carriage_op = UsdGeom.Xformable(carriage).GetOrderedXformOps()[0]
    carriage_base = np.asarray(carriage_op.Get(), dtype=float)
    half_travel = S['travel_m'][1]

    def rail_x(t):
        t = 0. if motor_start is None else max(0., t-motor_start)
        if a.static_rack or a.pick_only or a.settle_only:
            return a.rail_origin or 0.
        origin = S.get('initial_rail_origin_m', -half_travel) if a.rail_origin is None else a.rail_origin
        if a.level != 'hard':
            # Easy/Medium model a one-way washer infeed. The episode ends
            # before the rack reaches the covered exit, so speed never varies.
            return origin + a.transport_speed * t
        # Round each reversal over a fixed 0.6 s, independent of robot phases.
        # All loading windows must lie in constant-speed portions of a leg.
        span = 2. * half_travel
        initial_phase = origin+half_travel if params['direction'] == 1 else 2*span-(origin+half_travel)
        phase = (initial_phase + a.transport_speed * t) % (2. * span)
        turn_distance = a.transport_speed*.3
        endpoint_distance = min(phase, abs(phase-span), 2*span-phase)
        if endpoint_distance < turn_distance:
            sign = 1. if abs(phase-span) < turn_distance else -1.
            return sign*(half_travel-turn_distance/2-endpoint_distance**2/(2*turn_distance))
        return (-half_travel + phase if phase <= span
                else half_travel - (phase - span))

    initial_rail = rail_x(0)
    carriage_op.Set(Gf.Vec3d(initial_rail, *carriage_base[1:]))
    # Recorded dishes are world-space siblings of the moving fixture, not its
    # children.  Apply a new case's reset offset once to those bodies as well.
    # Otherwise Easy's changed rail origin moves the rack out from under them.
    live = stage.GetPrimAtPath('/World/LiveDishes')
    if live:
        for prim in live.GetChildren():
            name = prim.GetName()
            if name.startswith('Preloaded'):
                delta = Gf.Vec3d(initial_rail-carriage_base[0], 0, 0)
            elif name.startswith('Source'):
                delta = source_shift
            else:
                continue
            xf = UsdGeom.Xformable(prim)
            pose = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pose.SetTranslateOnly(pose.ExtractTranslation()+delta)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(pose)
    sources, preloaded, initial_positions = [], [], {}
    # Released replay scenes carry the already materialized dishes under
    # /World/LiveDishes/Source* and Preloaded*. Older templates instead carry
    # /World/DishLoading/Source/Plate* and Carriage/Plate*. Accept both while
    # preserving the authored pose and asset identity.
    UsdGeom.Xform.Define(stage, '/World/LiveDishes')
    for kind, parent, count in [('Source', BASE+'/Source', S['count']),
                                ('Preloaded', BASE+'/Carriage', S['total_slots'])]:
        for i in range(count):
            canonical = parent+f'/Plate{i}'
            path = '/World/LiveDishes/'+kind+str(i)
            old = canonical
            old_prim = stage.GetPrimAtPath(old)
            if not old_prim.IsValid():
                existing = stage.GetPrimAtPath(path)
                if existing.IsValid():
                    old = path
                    old_prim = existing
            if kind == 'Preloaded' and i in S['empty_slot_indices']:
                # Remove either the old authored prototype or the materialized
                # slot, but never leave an empty slot body in the world.
                if old_prim.IsValid():
                    remove_composed_prim(stage, old)
                continue
            if not old_prim.IsValid():
                # A fixed test occupancy can differ from the recorded source
                # occupancy. Reuse the nearest authored plate prototype and place
                # the clone at the missing slot so reset remains deterministic.
                candidates = []
                for j in range(i):
                    for candidate in (BASE + '/Carriage/Plate' + str(j),
                                      '/World/LiveDishes/Preloaded' + str(j)):
                        if stage.GetPrimAtPath(candidate).IsValid():
                            candidates.append((j, candidate))
                if not candidates:
                    raise RuntimeError('No authored plate prototype for slot: '+canonical)
                source_i, source_path = min(candidates, key=lambda pair: abs(pair[0]-i))
                # Copy into the world-space dish container.  Copying a recorded
                # world transform under Carriage would apply the rack offset
                # a second time and put newly occupied slots outside the rack.
                pose = UsdGeom.Xformable(stage.GetPrimAtPath(source_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                pose.SetTranslateOnly(pose.ExtractTranslation() +
                                      Gf.Vec3d((i-source_i)*S['slot_pitch_m'], 0, 0))
                copy_local_prim(stage, source_path, path)
                old = path
                old_prim = stage.GetPrimAtPath(old)
                xf = UsdGeom.Xformable(old_prim)
                xf.ClearXformOpOrder()
                xf.AddTransformOp().Set(pose)
            UsdGeom.Imageable(old_prim).MakeVisible()
            pose = UsdGeom.Xformable(old_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if old != path:
                copy_local_prim(stage, old, path)
                remove_composed_prim(stage, old)
            node = stage.GetPrimAtPath(path)
            if not node.IsValid():
                raise RuntimeError('Failed to materialize plate at reset: '+path)
            xf = UsdGeom.Xformable(node)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(pose)
            rb = UsdPhysics.RigidBodyAPI.Apply(node)
            rb.CreateRigidBodyEnabledAttr(True)
            rb.CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(node).CreateMassAttr(.12)
            ph = PhysxSchema.PhysxRigidBodyAPI.Apply(node)
            ph.CreateSolverPositionIterationCountAttr(64)
            ph.CreateSolverVelocityIterationCountAttr(8)
            ph.CreateEnableCCDAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(node).CreateThresholdAttr(0.)
            for child in Usd.PrimRange(node):
                if child.IsA(UsdGeom.Mesh): collider(child, True)
            (sources if kind == 'Source' else preloaded).append(path)
            initial_positions[path] = np.asarray(pose.ExtractTranslation()).tolist()

    for prefix in ['/World/Worktable', '/World/TableFrame', '/World/PedestalLeft',
                   '/World/PedestalRight', '/World/CameraStand', BASE]:
        prim = stage.GetPrimAtPath(prefix)
        if not prim: continue
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Gprim) and not any(x in str(child.GetPath()) for x in ['Label', 'Tick', 'Status', 'Target']):
                collider(child, str(child.GetPath()).startswith(BASE+'/Carriage'))
    rb = UsdPhysics.RigidBodyAPI.Apply(carriage)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)

    control_paths, control_origins = {}, {}
    if a.controls or a.controls_only:
        for name, stem in [('start', 'StartCap'), ('send', 'SendCap'), ('mode', 'ModeDial'), ('speed', 'SpeedLever')]:
            path = BASE+'/Panel/'+stem
            cap = stage.GetPrimAtPath(path)
            if not cap: continue
            control_paths[name] = path
            center = UsdGeom.Xformable(cap).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            control_origins[name] = np.array(center)
            UsdPhysics.RigidBodyAPI.Apply(cap).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(cap).CreateMassAttr(.015)
            PhysxSchema.PhysxRigidBodyAPI.Apply(cap).CreateDisableGravityAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(cap).CreateThresholdAttr(0.)
            ring = stage.GetPrimAtPath(BASE+'/Panel/'+('ModeRing' if name == 'mode' else stem.replace('Cap', 'Ring')))
            if ring: UsdPhysics.CollisionAPI(ring).CreateCollisionEnabledAttr(False)
            joint = (UsdPhysics.RevoluteJoint if name == 'mode' else UsdPhysics.PrismaticJoint).Define(stage, path+'/ControlJoint')
            joint.CreateBody0Rel().SetTargets([])
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*center))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateAxisAttr('Y' if name == 'speed' else 'Z')
            joint.CreateLowerLimitAttr(0. if name == 'speed' else -5. if name == 'mode' else -.004)
            joint.CreateUpperLimitAttr(.070 if name == 'speed' else 90. if name == 'mode' else 0.)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular' if name == 'mode' else 'linear')
            drive.CreateStiffnessAttr(0. if name in ['mode', 'speed'] else 180.)
            drive.CreateDampingAttr(1. if name == 'speed' else .05 if name == 'mode' else 4.)
            drive.CreateMaxForceAttr(.5 if name == 'mode' else 3.)
            drive.CreateTargetPositionAttr(0.)
            if name == 'mode':
                pointer_path = BASE+'/Panel/ModePointer'
                old_pointer = stage.GetPrimAtPath(pointer_path)
                pointer_pose = UsdGeom.Xformable(old_pointer).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                cap_pose = UsdGeom.Xformable(cap).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                copy_local_prim(stage, pointer_path, path+'/Pointer')
                pointer = UsdGeom.Xformable(stage.GetPrimAtPath(path+'/Pointer'))
                pointer.ClearXformOpOrder()
                pointer.AddTransformOp().Set(pointer_pose*cap_pose.GetInverse())
                remove_composed_prim(stage, pointer_path)

    if pivot_selector and (a.controls or a.controls_only):
        path = BASE+'/Selector/Lever'
        cap = stage.GetPrimAtPath(path)
        control_paths['gear'] = path
        center = UsdGeom.Xformable(cap).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        control_origins['gear'] = np.array(center)
        UsdPhysics.RigidBodyAPI.Apply(cap).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(cap).CreateMassAttr(.08)
        ph = PhysxSchema.PhysxRigidBodyAPI.Apply(cap)
        ph.CreateDisableGravityAttr(True)
        ph.CreateMaxAngularVelocityAttr(30.)
        ph.CreateSolverPositionIterationCountAttr(64)
        ph.CreateSolverVelocityIterationCountAttr(8)
        PhysxSchema.PhysxContactReportAPI.Apply(cap).CreateThresholdAttr(0.)
        for child in Usd.PrimRange(cap):
            if child.IsA(UsdGeom.Gprim): collider(child, True)
        # PhysX's primitive-cylinder collision proxy extends into the tight socket.
        # Use an explicit convex cylinder with the visual mesh's exact dimensions.
        drum = stage.GetPrimAtPath(path+'/FullDrum')
        UsdPhysics.CollisionAPI(drum).CreateCollisionEnabledAttr(False)
        count = 64
        points = [(x, .032*np.cos(2*np.pi*i/count), .032*np.sin(2*np.pi*i/count))
                  for x in [-.024, .024] for i in range(count)]
        faces = [list(reversed(range(count))), list(range(count, 2*count))]
        faces += [[i, (i+1)%count, count+(i+1)%count, count+i] for i in range(count)]
        proxy = UsdGeom.Mesh.Define(stage, path+'/DrumCollider')
        proxy.CreatePointsAttr(points)
        proxy.CreateFaceVertexCountsAttr([len(face) for face in faces])
        proxy.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        proxy.CreateSubdivisionSchemeAttr('none')
        proxy.CreateVisibilityAttr('invisible')
        collider(proxy.GetPrim(), True)
        # The axle is an internal bearing represented by the revolute constraint.
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(BASE+'/Selector/Axle')).CreateCollisionEnabledAttr(False)
        joint = UsdPhysics.RevoluteJoint.Define(stage, path+'/ControlJoint')
        joint.CreateBody0Rel().SetTargets([])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*center))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(-19.)
        joint.CreateUpperLimitAttr(19.)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
        drive.CreateStiffnessAttr(.4)
        drive.CreateDampingAttr(.02)
        drive.CreateMaxForceAttr(.15)
        drive.CreateTargetPositionAttr(18.)

    ctx.robot.prepare_stage(stage)
    ctx.robot.mount_bases(stage, M['station']['config']['robots'], legacy_anchors=True)
    ctx.robot.attach_cameras(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    robots = ctx.robot.articulations(world)
    bodies = {path: world.scene.add(SingleRigidPrim(path, name=path.split('/')[-1])) for path in sources+preloaded}
    from mind2act_eval.robot.read_only import robot_body_readers
    reader_paths = {'Base'+s: '/World/Robot'+s+'/Asset/base_link' for s in robots}
    reader_paths.update({'Clearance'+link: '/World/RobotLeft/Asset/'+link for link in ['link6', 'link7', 'link8']})
    readers = robot_body_readers(stage, reader_paths, world=world)
    bases = {s: readers['Base'+s] for s in robots}
    wrists = {link: readers['Clearance'+link] for link in ['link6', 'link7', 'link8']}
    controls = {name: world.scene.add(SingleRigidPrim(path, name='Control'+name)) for name, path in control_paths.items()}
    dish_colliders = [c for path in bodies for c in Usd.PrimRange(stage.GetPrimAtPath(path)) if c.HasAPI(UsdPhysics.CollisionAPI)]
    for c in dish_colliders: UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr(False)
    world.reset()
    effective_drive = ctx.robot.verify(robots, stage)
    rest = ctx.robot.rest_pose()
    commands = {s: rest.copy() for s in robots}
    previous_commands = {s: rest.copy() for s in robots}
    ctx.robot.reset_to_rest()
    for c in dish_colliders: UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr(True)
    def start_motor(t):
        nonlocal motor_start
        motor_start=t
    return locals()
