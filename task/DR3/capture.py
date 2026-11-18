# Source: fr3-replay-new15-20261031 / detector_capture.py; see config/source.json.
"""Private DR3 physics evidence capture, shared by all policy types.

Static collision enclosures are stored once; each tick contains rigid poses,
actual door joints and contact normals. Endpoint poses are not sweep proof.
"""
import hashlib
import itertools
import math
import numpy as np


def contact_measurement(pair, points):
    """Preserve collider ordering so a support normal keeps its ownership."""
    impulse = np.asarray([list(point.impulse) for point in points], dtype=float).reshape((-1,3))
    normals = np.asarray([list(point.normal) for point in points], dtype=float).reshape((-1,3))
    if not np.isfinite(impulse).all() or not np.isfinite(normals).all():
        raise ValueError("nonfinite DR3 contact measurement")
    item_index = next((i for i,path in enumerate(pair) if path=="/World/Task/Item" or path.startswith("/World/Task/Item/")),None)
    upward = [] if item_index is None else (normals[:,2]*(1 if item_index==0 else -1)).tolist()
    return dict(pair=list(pair),active=bool(len(points)),impulse_ns=float(np.max(np.linalg.norm(impulse,axis=1),initial=0.)),
                item_upward_normals=upward)


def tray_floor_geometry(triangles, origin):
    """Store the actual inner floor, without replacing it by an inset box."""
    triangles = np.asarray(triangles, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('nonfinite tray floor origin')
    if not triangles.size:
        return []
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError('invalid native tray floor mesh')
    # This is the existing source floor-plane extraction precision, not an
    # enlargement of the accepted XY region. Preserve every real triangle.
    floor = triangles[np.all(np.abs(triangles[:, :, 2]-origin[2]) <= 1e-5, axis=1)]
    return (floor[:, :, :2]-origin[:2]).tolist()


class Capture:
    def __init__(self, stage, item, robots, config, gate_getters, gates):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        from mind2act_eval.robot.read_only import robot_body_readers
        self.bodies = {}
        self.gate_getters = gate_getters
        self.subscription_active=False
        self.contact_errors=[]
        self.contact_report_counter=[0]
        self.audit=dict(last_begin_step=-1,last_sample_tick=None,last_sample_time_s=None,
                        before_step_count=0,sample_count=0,errors=[])
        self.previous_poses=None
        self.previous_tick=None
        cache = UsdGeom.XformCache()
        expected, represented, missing = [], [], []
        definitions = {}
        roots = ["/World/Robot"+side+"/Asset" for side in robots] + ["/World/Task/Item"]
        for root in roots:
            prim_root = stage.GetPrimAtPath(root)
            if not prim_root:
                missing.append(root);continue
            for prim in Usd.PrimRange(prim_root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                    continue
                path = str(prim.GetPath())
                expected.append(path)
                owner = prim
                while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    owner = owner.GetParent()
                if not owner or not str(owner.GetPath()).startswith(root):
                    missing.append(path);continue
                owner_path = str(owner.GetPath())
                try:
                    from .mesh import collider_vertices, collider_convex_parts
                    from scipy.spatial import ConvexHull
                    points=collider_vertices(stage,prim,owner)
                    points=points[ConvexHull(points).vertices]
                    owner_world = np.asarray(cache.GetLocalToWorldTransform(owner))
                    scale = np.linalg.norm(owner_world[:3,:3],axis=1)
                    if np.any(scale<=0):
                        raise ValueError("invalid rigid collider scale")
                    rotation = owner_world[:3,:3]/scale[:,None]
                    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-6) or np.linalg.det(rotation)<0:
                        raise ValueError("unsupported collider shear or reflection")
                    points=points*scale
                    body = definitions.setdefault(owner_path,dict(vertices=[],colliders=[],collision_parts=[],role="item" if root.endswith("/Item") else "robot",
                        arm=next((side for side in robots if root.startswith("/World/Robot"+side)),None),
                        gripper=any("/"+name in owner_path for name in ("link6","link7","link8","camera","camera_base")),vertices_are_actual_collision_points=True))
                    body["vertices"] += points.tolist()
                    parts=collider_convex_parts(stage,prim,owner)
                    if parts is None:body['collision_parts']=None
                    elif body['collision_parts'] is not None:
                        for part in parts:
                            part['vertices']=(np.asarray(part['vertices'])*scale).tolist()
                        body['collision_parts'].extend(parts)
                    body["colliders"].append(path)
                    body["representation"] = "conservative_convex_enclosure"
                    body["rigid_scale"] = scale.tolist()
                    represented.append(path)
                except (ValueError,TypeError):
                    missing.append(path)
        robot_names = {name: "dr3_detector_"+hashlib.sha256(name.encode()).hexdigest()[:12]
                       for name, definition in definitions.items() if definition["role"] == "robot"}
        robot_readers = robot_body_readers(stage, {key: name for name, key in robot_names.items()})
        contact_report_bodies=[]
        for name,definition in list(definitions.items()):
            from scipy.spatial import ConvexHull
            hull=ConvexHull(np.asarray(definition["vertices"]))
            definition["origin_in_actual_collision_hull"]=bool(definition["vertices_are_actual_collision_points"] and np.all(hull.equations[:,-1]<=1e-8))
            prim=stage.GetPrimAtPath(name)
            enabled=prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            threshold=PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get() if enabled else None
            contact_report_bodies.append(dict(body=name,enabled=enabled,threshold=threshold,
                qualified=bool(enabled and threshold is not None and math.isfinite(threshold) and threshold<=1e-5)))
            try:
                if definition["role"]=="item":
                    self.bodies[name]=item
                else:
                    body=robot_readers[robot_names[name]]
                    body.initialize();self.bodies[name]=body
            except Exception:
                missing += definition["colliders"]
                for path in definition["colliders"]:
                    if path in represented:represented.remove(path)
                del definitions[name]
        collider_paths=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                collider_paths.append(str(prim.GetPath()))
        robot_colliders=[path for path in collider_paths if path.startswith("/World/Robot")]
        structural=[]
        for a,b in itertools.combinations(robot_colliders,2):
            if self._structural(a,b):structural.append([a,b])
        for path in robot_colliders:
            if "/base_link/" in path or path.endswith("/base_link"):
                structural += [[path,other] for other in collider_paths if "Pedestal" in other]
        destination=config["DEST"]
        from .mesh import collision_mesh
        floor_triangles=[]
        for prim in stage.Traverse():
            if str(prim.GetPath())=="/World/Task/Tray":
                mesh=collision_mesh(stage,prim)
                matrix=np.asarray(cache.GetLocalToWorldTransform(prim))
                if len(mesh):
                    vertices=(np.c_[mesh.reshape(-1,3),np.ones(mesh.size//3)]@matrix)[:,:3]
                    floor_triangles=vertices.reshape((-1,3,3))
        tray_origin = [destination[0], destination[1], config['Z']+.00855]
        actual_floor = tray_floor_geometry(floor_triangles, tray_origin)
        tray_qualified = bool(actual_floor)
        floor_xy = np.asarray(actual_floor).reshape((-1, 2))
        lower_xy = floor_xy.min(axis=0).tolist() if len(floor_xy) else [0., 0.]
        upper_xy = floor_xy.max(axis=0).tolist() if len(floor_xy) else [0., 0.]
        from .geometry import MEASUREMENT_CONTRACT
        self.manifest=dict(schema="dr3-collision-pose-evidence-v2",physics_dt_s=1/120,
            native_runtime_validated=False,tray_region_qualified=tray_qualified,
            qualification_status="pending_native_geometry_contact_and_sweep_validation",
            contact_report_bodies=contact_report_bodies,
            bodies=definitions,coverage=dict(expected_colliders=sorted(expected),represented_colliders=sorted(represented),missing_colliders=sorted(set(missing))),
            item_root="/World/Task/Item",gates=gates,
            gate_colliders=[p for p in collider_paths if "/Gate/" in p or p.endswith("/Gate")],
            frame_colliders=[p for p in collider_paths if any(word in p for word in ("/Rail","/Post","/Frame"))],
            tray_colliders=[p for p in collider_paths if "/Tray/" in p or p.endswith("/Tray")],
            structural_contact_pairs=structural,
            thresholds=dict(contact_impulse_ns=1e-5,support_normal_z=.9,motion_epsilon_m=1e-6),
            tray=dict(origin=tray_origin,
                low=[*lower_xy, 0.],high=[*upper_xy, config['SZ'][2]],
                floor_triangles_xy=actual_floor,
                provenance='actual inner floor triangles in tray-local XY; no containment inset'),
            motion_contract=MEASUREMENT_CONTRACT,measurement_contract=MEASUREMENT_CONTRACT,
            full_rotation_is_scoring_precondition=False,
            intra_step_continuous_trajectory_claimed=False)
        if not self.manifest["tray_colliders"]:
            self.manifest["coverage"]["missing_colliders"].append("target tray collision inventory")
        if not self.manifest["gate_colliders"]:
            self.manifest["coverage"]["missing_colliders"].append("gate collision inventory")

    @staticmethod
    def _structural(a,b):
        import re
        pattern=r"/World/Robot(Left|Right)/Asset/(base_link|link[0-9]+|camera_base|camera)(?:/|$)"
        left,right=re.search(pattern,a),re.search(pattern,b)
        if not left or not right or left[1]!=right[1]:return False
        assembly={"link6","link7","link8","camera","camera_base"}
        if left[2] in assembly and right[2] in assembly:return True
        def number(name):return 0 if name=="base_link" else int(name[4:]) if name.startswith("link") else None
        x,y=number(left[2]),number(right[2])
        return x is not None and y is not None and abs(x-y)<=1

    def begin_step(self, step):
        if step!=self.audit["last_begin_step"]+1:
            if "before_step clock gap" not in self.audit["errors"]:
                self.audit["errors"].append("before_step clock gap")
        self.audit["last_begin_step"]=step
        self.audit["before_step_count"]+=1

    def sampling_receipt(self):
        geometry=self.manifest["coverage"]
        coverage=bool(geometry["expected_colliders"] and
            set(geometry["expected_colliders"])==set(geometry["represented_colliders"]) and not geometry["missing_colliders"])
        contacts=bool(self.subscription_active and self.contact_report_counter[0]>0 and self.manifest["contact_report_bodies"] and
            all(row["qualified"] for row in self.manifest["contact_report_bodies"]) and not self.contact_errors)
        sequence=bool(not self.audit["errors"] and self.audit["before_step_count"]>0 and
            self.audit["sample_count"]==self.audit["before_step_count"]+1)
        from .geometry import MEASUREMENT_CONTRACT
        return dict(measurement_contract=MEASUREMENT_CONTRACT,
                    full_rotation_is_scoring_precondition=False,intra_step_continuous_trajectory_claimed=False,
                    collision_coverage_complete=coverage,contact_reporting_complete=contacts,
                    physics_samples_complete=sequence,audit=dict(self.audit),contact_errors=list(self.contact_errors),
                    contact_report_callback_count=self.contact_report_counter[0],
                    native_runtime_validated=bool(coverage and contacts and sequence))

    def snapshot(self, contacts, tick, simulation_time_s):
        poses={}
        for name,body in self.bodies.items():
            p,q=body.get_world_pose()
            poses[name]=dict(position=np.asarray(p,float).tolist(),quaternion_wxyz=np.asarray(q,float).tolist(),
                linear_speed_m_s=float(np.linalg.norm(body.get_linear_velocity())),
                angular_speed_rad_s=float(np.linalg.norm(body.get_angular_velocity())))
        sweep={}
        if tick is not None and tick!=self.audit["last_sample_tick"]:
            if self.audit["last_sample_tick"] is not None:
                if tick!=self.audit["last_sample_tick"]+1 or abs(simulation_time_s-self.audit["last_sample_time_s"]-1/120)>1e-7:
                    if "measured sample clock gap" not in self.audit["errors"]:
                        self.audit["errors"].append("measured sample clock gap")
            elif tick!=0:
                self.audit["errors"].append("initial measured state missing")
            self.audit["last_sample_tick"]=tick
            self.audit["last_sample_time_s"]=simulation_time_s
            self.audit["sample_count"]+=1
        if self.previous_poses is not None and tick is not None and tick!=self.previous_tick:
            # Retained only as a diagnostic envelope. The scorer reads actual
            # 120 Hz poses/contact samples, not this arbitrary-rotation bound.
            sweep={name:dict(contract="bounded-rigid-envelope-v1",source="native_pose_full_rotation",
                    actual_collision_vertices=self.manifest["bodies"][name].get("vertices_are_actual_collision_points",False))
                   for name in poses}
        self.previous_poses=poses
        self.previous_tick=tick
        contacts_complete=bool(self.subscription_active and self.contact_report_counter[0]>0 and not self.contact_errors and self.manifest["contact_report_bodies"] and
            all(row["qualified"] for row in self.manifest["contact_report_bodies"]) and
            (tick==0 or self.audit["last_begin_step"]==tick-1))
        return dict(poses=poses,gates={name:float(getter()) for name,getter in self.gate_getters.items()},
                    contacts=[contacts[key] for key in sorted(contacts)],contacts_complete=contacts_complete,sweeps=sweep,
                    sampling=dict(tick=tick,before_step_epoch=self.audit["last_begin_step"],
                        subscription_active=self.subscription_active,errors=list(self.audit["errors"])))
