# ruff: noqa: F401
"""CP02 native cake scene initialization.

Source: cp02-hard-support-v3 cases_unified/CP02/scene.py (SHA256 a364dbc2... lineage:
cases/CP/CP02/runtime/scene.py). The Easy/Medium source cp02-em-qualified-batch
differs only by `physicalize` not returning the Hard support-topology record, which
this code does for level == "hard" alone, so one code path serves all difficulties.

Structural changes only: no SimulationApp/open_stage/World construction here
(ctx.sim owns them), X5 stage work goes through ctx.robot, args/manifest/config
arrive as frozen mappings.
"""
from pathlib import Path
import numpy as np
import os


def initialize(M, C, ctx):
    from types import SimpleNamespace
    a=SimpleNamespace(output=str(ctx.output_dir))
    O=Path(a.output);O.mkdir(parents=True,exist_ok=True)
    DT=1/120;CAKE_TOP=.0805
    spec=M['spec']
    if 'ingredient_shift_xy_m' not in spec:
        raise RuntimeError('CP02 requires fixed-case ingredient_shift_xy_m')
    center_xy=spec.get('center_xy_m')
    if not isinstance(center_xy, (list, tuple)) or len(center_xy) != 2:
        raise RuntimeError('CP02 frozen source missing center_xy_m')
    CENTER=(float(center_xy[0]), float(center_xy[1]))
    S='/World/CP02V1'
    from mind2act_eval.tasks.CP02.mechanism.native_assets import restore_authored, physicalize, ROOT

    a.cover_only=False;a.cover_side_view=False;a.grip_force=3.
    cover_raise=.035
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade, PhysicsSchemaTools
    from scipy.spatial.transform import Rotation
    from isaacsim.core.prims import SingleRigidPrim
    stage = omni.usd.get_context().get_stage()
    V = restore_authored(stage, M)
    cover_stem_top=V['cover_handle_z']+cover_raise-.006
    cover_stem_height=cover_stem_top-.120
    cover_stem_center=(cover_stem_top+.120)/2
    if spec['difficulty']=='hard':
        knob=UsdGeom.Xformable(stage.GetPrimAtPath('/World/CP02V1/Cover/Handle'));knob.GetOrderedXformOps()[0].Set(Gf.Vec3d(0,0,V['cover_handle_z']+cover_raise))
        # Plan-bound handle height; stem stays embedded 2 mm into the roof.
        stem=UsdGeom.Cylinder(stage.GetPrimAtPath('/World/CP02V1/Cover/HandleStem'));stem.GetHeightAttr().Set(cover_stem_height);UsdGeom.Xformable(stem).GetOrderedXformOps()[0].Set(Gf.Vec3d(0,0,cover_stem_center))
    if a.cover_only:
        for item in spec['items']:
            pos=np.array(item['target_cake_local_m'])+np.array([*CENTER,spec['assembly_z_m']])
            node=UsdGeom.Xformable(stage.GetPrimAtPath('/World/CP02V1/Source/'+item['identity']));node.GetOrderedXformOps()[0].Set(Gf.Vec3d(*pos))
    # Raised plunger caps provide finger clearance above their protective bezel.
    for name in V['buttons']:
        prim=stage.GetPrimAtPath('/World/CP02V1/'+name+'Button');op=UsdGeom.Xformable(prim).GetOrderedXformOps()[0];p0=op.Get();op.Set(Gf.Vec3d(p0[0],p0[1],p0[2]+.015))
        stem=UsdGeom.Cylinder.Define(stage,'/World/CP02V1/'+name+'Button/Stem');stem.CreateRadiusAttr(.011);stem.CreateHeightAttr(.017);stem.AddTranslateOp().Set(Gf.Vec3d(0,0,-.010));UsdShade.MaterialBindingAPI.Apply(stem.GetPrim()).Bind(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(S+'/'+name+'Button/Cap')).ComputeBoundMaterial()[0])
        # Runtime height is recorded separately; original manifest hash stays fixed.
    # Shallow lower box with a skirt on the lid: when open, the rim clears the
    # existing diagonal pastry grasp; when closed, the two halves meet.
    for idx in [0,1,2]:
        prim=stage.GetPrimAtPath('/World/CP02V1/Box/Wall'+str(idx));ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
        pos=ops[0].Get();scale=ops[1].Get();ops[0].Set(Gf.Vec3d(pos[0],pos[1],.768+.0125));ops[1].Set(Gf.Vec3d(scale[0],scale[1],.025))
    for idx,pos,size in [(0,(-.18,-.0915,-.011),(.006,.183,.026)),(1,(.18,-.0915,-.011),(.006,.183,.026)),(2,(0,-.181,-.011),(.36,.006,.026))]:
        c=UsdGeom.Cube.Define(stage,'/World/CP02V1/Lid/Skirt'+str(idx));c.CreateSizeAttr(1);c.AddTranslateOp().Set(Gf.Vec3d(*pos));c.AddScaleOp().Set(Gf.Vec3d(*size));UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(S+'/Lid/Panel')).ComputeBoundMaterial()[0])
    # Shared initializer owns all robot root/body/joint/drive properties.
    shared_robot_intent=ctx.robot.prepare_stage(stage)
    ctx.robot.attach_cameras(stage)
    ctx.robot.mount_bases(stage, M['station']['robots'])  # station base poses are applied by the shared robot service
    material = UsdShade.Material.Define(stage, '/World/CP02V1ContactMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(.9)
    pm.CreateDynamicFrictionAttr(.8)
    pm.CreateRestitutionAttr(0.)

    def collision(prim, approximation='none'):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(.0007)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')

    for prefix in ('/World/Worktable', '/World/TableFrame', '/World/CameraStand',
                   '/World/PedestalLeft', '/World/PedestalRight', '/World/Floor',
                   S + '/Machine', S + '/IngredientTray', S + '/Parchment'):
        parent = stage.GetPrimAtPath(prefix)
        if not parent:
            continue
        for prim in Usd.PrimRange(parent):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
                collision(prim)

    def cylinder(path, radius, height, center):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(height)
        c.CreateAxisAttr('Z')
        c.AddTranslateOp().Set(Gf.Vec3d(*center))
        c.MakeInvisible()
        collision(c.GetPrim())
        return c

    def rigid(prim, mass):
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(8)
        api.CreateSleepThresholdAttr(0.)
        api.CreateEnableCCDAttr(True)
        if str(prim.GetPath()).startswith(S+'/Source/'):
            api.CreateMaxDepenetrationVelocityAttr(1.)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)

    cake_path = S + '/CakeAssembly'
    cake_xform = UsdGeom.Xform.Define(stage, cake_path)
    cake = cake_xform.GetPrim()
    cake.GetAttribute('xformOp:translate').Set(Gf.Vec3d(CENTER[0], CENTER[1], spec['assembly_z_m']))
    cake.GetAttribute('xformOp:rotateZ').Set(float(spec.get('initial_cake_phase_deg', 0.)))
    # CP02 owns the platter, cake body and demonstration carriers. These are
    # authored under CP02V1 so a reset cannot bind a neighbouring task geometry.
    cylinder(cake_path + '/CakeContact', spec['geometry']['cake_diameter_m']/2, .077, (0, 0, CAKE_TOP - .077 / 2))
    # The native collection replay has only the cake contact body here.  An
    # extra platter disk under the same rigid cake creates a static-friction
    # contact with the authored machine and stalls the spin joint before the
    # policy ever acts, so do not synthesize that second collision body.
    rigid(cake, 1.2)
    # Match the native collection replay: the cake motor and contact impulses
    # are solved together at high velocity-iteration count.  The lower default
    # silently lets the motor stall as soon as a policy begins manipulating a
    # source item.
    PhysxSchema.PhysxRigidBodyAPI.Apply(cake).CreateSolverVelocityIterationCountAttr(64)
    spin = UsdPhysics.RevoluteJoint.Define(stage, S + '/SpinJoint')
    spin.CreateBody0Rel().SetTargets([])
    spin.CreateBody1Rel().SetTargets([cake_path])
    spin.CreateAxisAttr('Z')
    spin.CreateLocalPos0Attr(Gf.Vec3f(*CENTER, spec['assembly_z_m']))
    spin.CreateLocalPos1Attr(Gf.Vec3f(0))
    spin.CreateLocalRot0Attr(Gf.Quatf(1))
    spin.CreateLocalRot1Attr(Gf.Quatf(1))
    spin.CreateExcludeFromArticulationAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(spin.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(48000.)
    drive.CreateMaxForceAttr(320.)
    speed_attr = drive.CreateTargetVelocityAttr(0.)
    # Visual demonstration macarons and hidden source copies are separate
    # prims so the cue can be removed without losing the physical objects.
    demo_root = UsdGeom.Xform.Define(stage, cake_path + '/Demonstration').GetPrim()
    paths = []
    for item in spec['items']:
        ident = item['identity']
        demo_path = cake_path + '/Demonstration/' + ident
        demo = stage.GetPrimAtPath(demo_path)
        if not demo or not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(demo)):
            raise RuntimeError('CP02 authored demonstration mesh missing: ' + ident)
        demo.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*item['target_cake_local_m']))
        UsdGeom.Imageable(demo).MakeVisible()
        path = S + '/Source/' + ident
        paths.append(path)
        node = stage.GetPrimAtPath(path)
        if not node or not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(node)):
            raise RuntimeError('CP02 authored source mesh missing: ' + ident)
        cylinder(path + '/ContactShape', .0205, .0225, (0, 0, .01175))
        # Bind the source copy to the fixed layout position from this case.
        src = item.get('source_bottom_world_m')
        if not isinstance(src, (list, tuple)) or len(src) != 3:
            raise RuntimeError('CP02 case missing source_bottom_world_m for ' + ident)
        node.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*src))
        rigid(node, .025)
        UsdGeom.Imageable(node).MakeInvisible()
    runtime_manifest=__import__('copy').deepcopy(M)
    runtime_manifest['v4']=V
    topology=physicalize(stage,runtime_manifest,collision,rigid,cylinder)
    if topology is not None:
        (Path(a.output)/'cover-support-topology.json').write_text(__import__('json').dumps(topology,indent=2)+'\n')
    # Cylinder contact reporting may emit repeated PhysX material-face warnings.
    # Keep normal errors while reducing only PhysX warning log spam.
    import carb.settings
    carb.settings.get_settings().set('/log/omni.physx.plugin/level', 'error')
    # Use the same solver contract as native collection.  The default TGS
    # scene changes the contact/motor balance and is not comparable to the
    # recorded CP02 physics.
    for scene_prim in stage.Traverse():
        if scene_prim.IsA(UsdPhysics.Scene):
            api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            api.CreateSolverTypeAttr('TGS')
            api.CreateEnableStabilizationAttr(False)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    # World/PhysicsContext defaults overwrite the pre-World authored solver.
    world.get_physics_context().set_solver_type('TGS')
    robots = ctx.robot.articulations(world)
    from mind2act_eval.robot.read_only import robot_body_readers
    base_readers = robot_body_readers(stage, {'Base'+s: '/World/Robot'+s+'/Asset/base_link' for s in robots}, world=world)
    bases = {s: base_readers['Base'+s] for s in robots}
    rotor = world.scene.add(SingleRigidPrim(prim_path=cake_path, name='Cake'))
    bodies = [world.scene.add(SingleRigidPrim(prim_path=path, name='Macaron' + str(i))) for i, path in enumerate(paths)]
    props={name:world.scene.add(SingleRigidPrim(prim_path=ROOT+'/'+name,name='Prop'+name)) for name in ['Lid']+[n+'Button' for n in V['buttons']]+(['Cover'] if spec['difficulty']=='hard' else [])}
    import carb
    carb.settings.get_settings().set('/log/level','error')
    from mind2act_eval.sim.asset_relocation import export_relative_layer
    export_relative_layer(stage.GetRootLayer(), O/'scene_runtime.usda')
    print('CP02_RESET_BEGIN', flush=True)
    from mind2act_eval.runtime_diagnostics import run_native_phase
    run_native_phase(world.reset, O, phase='CP02_RESET', timeout_s=300.)
    print('CP02_RESET_DONE', flush=True)
    cover_contact_view=None
    if spec['difficulty']=='hard':
        # Read-only tensor telemetry, independent of callback packet maxima.
        # Both reporter preparation and stabilization changes are disabled;
        # do not alter the scene, solver, contact offsets, mass or motion.
        import hashlib,json
        from isaacsim.core.api.sensors.rigid_contact_view import RigidContactView
        before=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest()
        cover_contact_view=RigidContactView(prim_paths_expr=ROOT+'/Cover',
            filter_paths_expr=[cake_path],name='CP02CoverContactTelemetry',
            prepare_contact_sensors=False,disable_stablization=False,max_contact_count=2048)
        cover_contact_view.initialize()
        after=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest()
        if before!=after:raise RuntimeError('Contact telemetry changed authored scene; refuse diagnostic')
        (O/'cover-contact-telemetry-init.json').write_text(json.dumps(dict(
            schema='cp02-cover-force-telemetry-v1',stage_before_sha256=before,stage_after_sha256=after,
            authored_scene_unchanged=True,prepare_contact_sensors=False,disable_stablization=False,
            body=ROOT+'/Cover',filter=cake_path,max_contact_count=2048,
            scope='Measurement only; original supported flag and240tick gate unchanged'),indent=2)+'\n')
    shared_robot_native=ctx.robot.verify(robots,stage)
    (O/'shared-x5-native.json').write_text(__import__('json').dumps(shared_robot_native,indent=2)+'\n')
    (O/'shared-x5-intent.json').write_text(__import__('json').dumps(shared_robot_intent,indent=2)+'\n')
    try:
        carb.logging.acquire_logging().set_level_threshold(carb.logging.LEVEL_ERROR)
    except AttributeError:pass
    rest = ctx.robot.rest_pose()
    command = rest.copy()
    velocity = np.zeros(8)
    right_command=rest.copy();right_velocity=np.zeros(8)
    active_arm='Left'
    controls={'started':spec['difficulty']=='easy' or a.cover_only,'stopped':a.cover_only,'start_event':None,'stop_event':None}
    for robot in robots.values():
        assert list(robot.dof_names) == ['joint' + str(i) for i in range(1, 9)]
    ctx.robot.reset_to_rest()  # sole joint writer: positions, zero velocities, hold target

    commands={s:rest.copy() for s in robots}
    return locals()
