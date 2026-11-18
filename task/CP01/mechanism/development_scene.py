# ruff: noqa: F401
"""CP01 task-owned scene initialization (Medium/Hard baseline).

Source: cp01-mh-single-place-keep-final-camera-v2, cases/CP/CP01/runtime/development_scene.py
(extracted from scripts/physco_cp01_meal_adaptive_replay.py, SHA256 4ba8e22a...).
No reference actions are loaded. Physics body is the selected baseline verbatim:
scene CCD + TGS, tray CCD/SpeculativeCCD, tray ContactOffset 0.002, repair_foods/
repair_fixture_primitives and initial_stack_clearance are all part of the baseline.

Changes from the source, all structural:
 * no SimulationApp / open_stage / World construction here; ctx.sim owns them.
 * X5 stage mounting, wrist cameras, rest-pose writes and native verification go
   through ctx.robot (reset_to_rest also zeroes the velocity target, which the baseline loop left implicit) (the single shared X5 service); no robot values in this file.
 * helper imports point at this package instead of physco_* scripts.
 * setup arrives as a frozen mapping instead of the collector args namespace.
"""

from pathlib import Path
import numpy as np


def initialize(setup, ctx):
    import omni.usd
    output = Path(ctx.output_dir)
    P = setup
    S = P["spec"]
    from pxr import (
        Usd,
        UsdGeom,
        UsdPhysics,
        PhysxSchema,
        Gf,
        UsdShade,
        PhysicsSchemaTools,
    )
    from isaacsim.core.prims import SingleRigidPrim
    from mind2act_eval.tasks.CP01.mechanism.usd_helpers import material as make_material, rounded

    stage = omni.usd.get_context().get_stage()
    from mind2act_eval.tasks.CP01.mechanism.conveyor_layout import apply_training_conveyor_layout
    conveyor_layout = apply_training_conveyor_layout(stage, P)
    for prim in list(stage.Traverse()):
        if prim.GetName().startswith("PickupMark"):
            stage.RemovePrim(prim.GetPath())
    authored_preview_paths = (
        "/World/CP01/ReferencePass",
        "/World/CP01/StreamGoods",
        "/World/CP01/InspectionTray",
    )
    authored_preview_prims = {
        path: bool(stage.GetPrimAtPath(path)) for path in authored_preview_paths
    }
    for path in authored_preview_paths:
        stage.RemovePrim(path)
    stale_preview_paths = [
        path for path in authored_preview_paths if bool(stage.GetPrimAtPath(path))
    ]
    if stale_preview_paths:
        raise RuntimeError(
            "CP01 authored preview namespace survived reset: "
            + ", ".join(stale_preview_paths)
        )
    material = UsdShade.Material.Define(stage, "/World/TrayGripMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(0.9)
    pm.CreateDynamicFrictionAttr(0.75)
    pm.CreateRestitutionAttr(0.0)

    def collider(prim, dynamic=False):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                "convexHull" if dynamic else "none"
            )
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(0.0005)
        api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, materialPurpose="physics"
        )

    for prefix in [
        "/World/Worktable",
        "/World/TableFrame",
        "/World/PedestalLeft",
        "/World/PedestalRight",
        "/World/CameraStand",
        "/World/CP01/Conveyor",
    ]:
        root = stage.GetPrimAtPath(prefix)
        if not root:
            continue
        for child in Usd.PrimRange(root):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prefix == "/World/CP01/Conveyor":
                if str(child.GetPath()).startswith(prefix + "/OriginalBelt/"):
                    # The historical imported track uses a redundant collision
                    # envelope. Keep it disabled; the moving top surface is the
                    # independently authored BeltContactSurface.
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
                elif child.IsA(UsdGeom.Gprim):
                    # Chassis, rails, fasteners and supports are the visible
                    # finite authored fixtures. Static triangle meshes follow
                    # their real surfaces instead of a convex envelope that
                    # blocks the upstream supply extension.
                    collider(child)
            elif child.IsA(UsdGeom.Gprim):
                collider(child)
    # All X5 root/body/joint/collision configuration and contact reporting are
    # owned by the shared robot initializer below. This task owns its objects.
    traymat = make_material(stage, "MealTray", (0.72, 0.12, 0.09), 0.05, 0.5)
    gripmat = make_material(stage, "MealTrayGrip", (0.25, 0.04, 0.03), 0.0, 0.7)
    padmat = make_material(stage, "MealTrayPad", (0.15, 0.17, 0.18), 0.0, 0.8)
    paths = []
    from mind2act_eval.tasks.CP01.mechanism.food_assets import build_food

    for i, item in enumerate(P["items"]):
        path = "/World/MealTrays/Tray" + str(i)
        paths.append(path)
        obj = build_food(stage, path, item["kind"], item["source_xyz"], dynamic=True)
        for child in Usd.PrimRange(obj.GetPrim()):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                collider(child, True)
        # Robots and trays report all grasp/support contacts. Unselected upstream
        # foods need physical collisions and velocity audits, not belt contact callbacks.
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.GetPrim())
        api.CreateSolverPositionIterationCountAttr(8)
        api.CreateSolverVelocityIterationCountAttr(2)
        api.CreateEnableCCDAttr(False)
    food_count = len(P["items"])
    tray_spec = P["tray_spec"]
    for j in range(len(P["tray_centers"])):
        path = "/World/MealTrays/Tray" + str(food_count + j)
        paths.append(path)
        node = UsdGeom.Xform.Define(stage, path)
        stack = P["tray_initial_stack_height_by_index"]
        height = 0 if stack is None else stack[j]
        pos = P["tray_initial_positions_m"][j]
        node.AddTranslateOp().Set(Gf.Vec3d(*pos))

        def part(name, c, size):
            ob = rounded(
                stage, path + "/" + name, c, size, min(0.004, min(size) / 3), traymat
            )
            collider(ob.GetPrim(), True)

        sx, sy = tray_spec["tray_size_m"]
        feet = 0.031
        part("Floor", (0, 0, feet + 0.0025), (sx, sy, 0.005))
        for sign in [-1, 1]:
            part(
                "X" + str(sign + 1),
                (sign * (sx / 2 - 0.002), 0, feet + 0.009),
                (0.004, sy, 0.008),
            )
            part(
                "Y" + str(sign + 1),
                (0, sign * (sy / 2 - 0.002), feet + 0.009),
                (sx - 0.008, 0.004, 0.008),
            )
        for xi, x in enumerate([-0.105, 0.105]):
            for yi, y in enumerate([-0.045, 0.045]):
                part("Foot" + str(xi) + str(yi), (x, y, feet / 2), (0.020, 0.020, feet))
        for side, handle_x in [("Center", 0.0)]:
            part(
                "GripBridge" + side,
                (handle_x, -sy / 2 - 0.009, tray_spec["handle_bridge_center_z_m"]),
                (0.032, 0.030, 0.010),
            )
            part(
                "Grip" + side,
                (
                    handle_x,
                    -sy / 2 - 0.029,
                    tray_spec["handle_bridge_center_z_m"] + 0.005,
                ),
                (0.05, 0.025, 0.026),
            )
        UsdPhysics.RigidBodyAPI.Apply(node.GetPrim()).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(node.GetPrim())
        mass.CreateMassAttr(0.12)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.033))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(node.GetPrim())
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(8)
        api.CreateAngularDampingAttr(0.3)
        PhysxSchema.PhysxContactReportAPI.Apply(node.GetPrim()).CreateThresholdAttr(0.0)
    # Reference orders are observation-only carriers. They have no collision or
    # rigid-body APIs and leave the public camera view before supply begins.
    reference_ops = []
    reference_roots = []
    ref_config = P["reference_batch"]
    reference_tray_size = P.get("tray_spec", {}).get("tray_size_m", [0.36, 0.14])
    for order_index, foods in enumerate(P["orders"]):
        ref_path = "/World/ReferenceOrders/Order" + str(order_index)
        ref = UsdGeom.Xform.Define(stage, ref_path)
        op = ref.AddTranslateOp()
        op.Set(
            Gf.Vec3d(
                ref_config["start_x"] - order_index * ref_config["spacing_m"],
                P["belt_y_m"],
                0.859,
            )
        )
        reference_ops.append(op)
        reference_roots.append(ref.GetPrim())
        rounded(
            stage,
            ref_path + "/Tray",
            (0, 0, 0.0025),
            (float(reference_tray_size[0]), float(reference_tray_size[1]), 0.005),
            0.012,
            traymat,
        )
        for slot, kind in enumerate(foods):
            visual = build_food(
                stage,
                ref_path + "/Food" + str(slot),
                kind,
                ((slot - 1) * 0.077, 0, 0.008),
                dynamic=False,
            )
            for prim in Usd.PrimRange(visual.GetPrim()):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        # All references exist outside the camera before reset and move as a batch.
    supply = P.get("supply_spec", {})
    upstream_min = float(supply.get("upstream_min_x", -1.70))
    downstream_max = 1.70 + float(supply["budget_s"]) * float(S["speed"])
    belt_length = downstream_max - upstream_min
    belt_center = (downstream_max + upstream_min) / 2
    belt = UsdGeom.Cube.Define(stage, "/World/BeltContactSurface")
    belt.CreateSizeAttr(1)
    belt.AddTranslateOp().Set(Gf.Vec3d(belt_center, P["belt_y_m"], 0.850))
    belt.AddScaleOp().Set(Gf.Vec3f(belt_length, P["belt_width_m"], 0.016))
    belt.MakeInvisible()
    collider(belt.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(belt.GetPrim())
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt.GetPrim())
    sv.CreateSurfaceVelocityEnabledAttr(True)
    sv.CreateSurfaceVelocityLocalSpaceAttr(False)
    speedattr = sv.CreateSurfaceVelocityAttr(Gf.Vec3f(0))
    x5_authored_receipt = ctx.robot.prepare_stage(stage)
    # Task-owned runtime shapes must be repaired before PhysX cooks the stage.
    # This also removes artwork CollisionAPI so the reset restore cannot enable it.
    from mind2act_eval.tasks.CP01.mechanism.contact_assets import repair_foods, repair_fixture_primitives
    cp01_contact_shape_receipt = repair_foods(stage, paths[:food_count])
    cp01_contact_shape_receipt += repair_fixture_primitives(stage)
    # Restore the official camera leaf before reset, under the now enabled
    # canonical camera fixed-mount chain. Preserve official mount and lens.
    ctx.robot.attach_cameras(stage)
    from mind2act_eval.variations import prepare_station
    prepare_station(ctx)
    world = ctx.world
    robots = ctx.robot.articulations(world)
    bodies = [
        world.scene.add(SingleRigidPrim(path, name="Tray" + str(i)))
        for i, path in enumerate(paths)
    ]
    for path in paths[:food_count]:
        root_prim = stage.GetPrimAtPath(path)
        if root_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            root_prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
    from mind2act_eval.robot.read_only import robot_body_readers
    base_readers = robot_body_readers(stage, {"Base"+s: "/World/Robot"+s+"/Asset/base_link" for s in robots}, world=world)
    bases = {s: base_readers["Base"+s] for s in robots}
    # No tray contacts during robot initialization. Enable them only after the
    # home pose has been installed, so reset cannot kick the stock out of place.
    traycolliders = [
        q
        for path in paths
        for q in Usd.PrimRange(stage.GetPrimAtPath(path))
        if q.HasAPI(UsdPhysics.CollisionAPI)
    ]
    for q in traycolliders:
        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
    # CCD must be enabled at scene and moving-body levels. Contact colliders,
    # masses and reporting remain active; this candidate still needs acceptance.
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(True)
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            print("CP01_PREVIOUS_SOLVER", scene_api.GetSolverTypeAttr().Get(), flush=True)
            scene_api.CreateSolverTypeAttr("TGS")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()).startswith("/World/MealTrays"):
            body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body_api.CreateEnableCCDAttr(True)
            # Sweep CCD covers translations; speculative contacts also account
            # for fast rotation of task tray rims and feet. Robot contact properties
            # are owned solely by the selected X5 module; no task-side override.
            body_api.CreateEnableSpeculativeCCDAttr(True)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()).startswith("/World/MealTrays"):
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api.CreateContactOffsetAttr(0.002)
            collision_api.CreateRestOffsetAttr(0.0)
    world.reset()
    rest = ctx.robot.rest_pose()
    commands = {s: rest.copy() for s in robots}
    ctx.robot.reset_to_rest()  # same set positions / zero velocities / hold target as the baseline loop
    x5_native_receipt = ctx.robot.verify(robots, stage)
    # Reset may advance physics while task collisions are disabled for robot
    # initialization. Install the retained task reset once, before enabling
    # contacts and before attach()'s settling ticks. Never write these poses
    # again during observations, grasping or scoring.
    task_reset_receipt = []
    from mind2act_eval.tasks.CP01.mechanism.initial_stack_clearance import stack_clearance
    initial_stack_adapter = stack_clearance(stage,P)
    initial_positions = [item['source_xyz'] for item in P['items']] + P['tray_initial_positions_m']
    for index,(body, position, path) in enumerate(zip(bodies, initial_positions, paths)):
        before_position, before_quaternion = body.get_world_pose()
        adjustment=(np.asarray(initial_stack_adapter['offset_m']) if index>=food_count else np.zeros(3))
        applied_position=np.asarray(position,dtype=float)+adjustment
        body.set_world_pose(applied_position, np.array([1., 0., 0., 0.]))
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
        after_position, after_quaternion = body.get_world_pose()
        task_reset_receipt.append(dict(path=path, before_position=np.asarray(before_position).tolist(),
            before_quaternion=np.asarray(before_quaternion).tolist(), source_position=list(position),
            applied_initial_position=applied_position.tolist(),initialization_adjustment_m=adjustment.tolist(),
            after_position=np.asarray(after_position).tolist(), after_quaternion=np.asarray(after_quaternion).tolist(),
            scope='single task initialization before contact enable; no rollout pose writes'))
    import json
    reset_output = output / 'task-reset-receipt.json'
    reset_output.parent.mkdir(parents=True, exist_ok=True)
    reset_output.write_text(json.dumps(task_reset_receipt, indent=2) + '\n')
    (reset_output.parent/'initial-stack-clearance-receipt.json').write_text(json.dumps(initial_stack_adapter,indent=2)+'\n')
    for q in traycolliders:
        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(True)

    runtime_stage_paths = (
        "/World/MealTrays",
        "/World/ReferenceOrders",
        "/World/BeltContactSurface",
    )
    missing_runtime_paths = [
        path for path in runtime_stage_paths if not bool(stage.GetPrimAtPath(path))
    ]
    if missing_runtime_paths:
        raise RuntimeError(
            "CP01 runtime namespace was not rebuilt: "
            + ", ".join(missing_runtime_paths)
        )
    scene_contract = {
        "authored_preview_paths": list(authored_preview_paths),
        "authored_preview_present_before_reset": authored_preview_prims,
        "authored_preview_present_after_reset": {
            path: bool(stage.GetPrimAtPath(path)) for path in authored_preview_paths
        },
        "runtime_paths": list(runtime_stage_paths),
        "runtime_paths_present": {
            path: bool(stage.GetPrimAtPath(path)) for path in runtime_stage_paths
        },
        "semantics": "authored CP01 preview removed; runtime reference orders and continuous supply rebuilt",
        "effective_x5_profile": x5_authored_receipt["configuration"]["profile_id"],
        "collector_spec_grip_force_N_not_runtime_drive": S.get("grip_force_N"),
    }

    return locals()
