"""Owned raw PhysX contact manifolds and actual rigid/joint measurements."""
import numpy as np
from .physical import CONTACT_SCHEMA, canonical_support_geometry
from .tcp import measured_tcp


class Contacts:
    def __init__(self):
        self.rows = {}
        self.errors = []

    def receive(self, headers, data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        for header in headers:
            try:
                pair = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.collider0,header.collider1)]
                canonical = tuple(sorted(pair))
                if header.type == ContactEventType.CONTACT_LOST:
                    self.rows.pop(canonical, None)
                    continue
                points = data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]
                geometry = [dict(normal=list(p.normal),separation_m=float(p.separation)) for p in points]
                impulse = np.sum([list(p.impulse) for p in points],axis=0) if points else np.zeros(3)
                if not np.isfinite(impulse).all():
                    raise ValueError('CP08 nonfinite PhysX impulse')
                canonical = tuple(sorted(pair))
                if not points and canonical in self.rows:
                    continue  # Sleeping persistent manifold; retain measured geometry.
                self.rows[canonical] = dict(pair=list(canonical),impulse_ns=impulse.tolist(),
                    contact_count=len(points),owned_support_points=canonical_support_geometry(pair,geometry))
            except (ValueError,TypeError,AttributeError) as error:
                if str(error) not in self.errors:self.errors.append(str(error))


class Sensors:
    def __init__(self, stage, bodies, arms, screen, textures, contacts, subscription, flanges, build_area_bounds=None):
        self.stage,self.bodies,self.arms = stage,bodies,arms
        self.screen,self.textures,self.contacts,self.subscription = screen,textures,contacts,subscription
        self.errors=contacts.errors
        self.flanges=flanges
        self.build_area_bounds=build_area_bounds

    def set_cue(self, visible):
        from pxr import Sdf
        self.screen.Set(Sdf.AssetPath(str(self.textures['plan.png' if visible else 'screen-idle.png'])))

    def read(self, tick):
        blocks={}
        for name,body in self.bodies.items():
            p,q=body.get_world_pose()
            blocks[name]=dict(position=p.tolist(),orientation_wxyz=q.tolist(),
                              linear_velocity_m_s=np.asarray(body.get_linear_velocity()).tolist())
        joints={side:np.asarray(arm.get_joint_positions()).tolist() for side,arm in self.arms.items()}
        flange_poses={}
        for side,body in self.flanges.items():
            positions,orientations=body.get_world_poses()
            p,q=positions[0],orientations[0]
            flange_poses[side]=dict(position=p.tolist(),orientation_wxyz=q.tolist())
        tcps={side:measured_tcp(f['position'],f['orientation_wxyz']) for side,f in flange_poses.items()}
        return dict(build_area_bounds_m=self.build_area_bounds,tick=tick,flange_poses=flange_poses,tcps=tcps,blocks=blocks,contacts=list(self.contacts.rows.values()),
            contact_measurement_schema=CONTACT_SCHEMA if not self.errors and self.subscription is not None else 'invalid',
            cue_visible=str(self.screen.Get().path)==str(self.textures['plan.png']),
            joints={s:q[:6] for s,q in joints.items()},
            measured_finger_positions_m={s:q[6:8] for s,q in joints.items()})

    def close(self):
        if self.subscription is not None:self.subscription.unsubscribe()
        self.subscription=None
