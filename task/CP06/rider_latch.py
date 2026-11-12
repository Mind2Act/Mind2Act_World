"""120 Hz environment hook: construct after rigid prim initialization, call
before EACH sim.step (including push steps). Reset/recreate on episode reset.
Uses world-SI force-at-position API, never USD drive coefficients. Runtime validation is recorded separately in server probe results. Do not also apply the old drive. External push is a force in N.
"""
import math

class RiderLatch:
    stiffness = 2000.0  # N/m
    damping = 20.0      # N s/m
    yield_error = .0005 # m; spring yields at 1 N

    def __init__(self, beam, rider):
        self.beam, self.rider = beam, rider
        self.anchor = None

    def state(self):
        import numpy as np
        from pxr import Gf
        bp, bq = self.beam.get_world_pose()
        rp, rq = self.rider.get_world_pose()
        rot = Gf.Rotation(Gf.Quatd(float(bq[0]), Gf.Vec3d(*map(float,bq[1:]))))
        axis = np.asarray(rot.TransformDir(Gf.Vec3d(1,0,0)), dtype=float)
        delta = np.asarray(rp)-np.asarray(bp)
        q = float(delta @ axis)-.082
        # Relative origin velocity includes rotation of the beam frame.
        vel = np.asarray(self.rider.get_linear_velocity())-np.asarray(self.beam.get_linear_velocity())
        v = float((vel-np.cross(self.beam.get_angular_velocity(),delta)) @ axis)
        return q, v, axis, np.asarray(rp), rq

    def step(self, push_n=0.0):
        import numpy as np
        from pxr import Gf
        q,v,axis,rp,rq = self.state()
        if self.anchor is None:self.anchor=q
        error=q-self.anchor
        # Plastic slip only beyond 0.5 mm; otherwise retain the anchor.
        # No velocity gating: damping is continuous, including through v=0.
        error=max(-self.yield_error,min(self.yield_error,error))
        self.anchor=q-error
        spring=-self.stiffness*error
        force=spring-self.damping*v
        # Same point for internal action/reaction: no artificial net wrench.
        point=rp
        rot=Gf.Rotation(Gf.Quatd(float(rq[0]),Gf.Vec3d(*map(float,rq[1:]))))
        grip=rp+np.asarray(rot.TransformDir(Gf.Vec3d(0,0,.054)),dtype=float)
        def apply(body,f,p):
            body._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.asarray(f,dtype=np.float32).reshape(1,3),positions=np.asarray(p,dtype=np.float32).reshape(1,3),is_global=True)
        # Combine push and latch into one force plus torque to avoid depending
        # on whether a backend accumulates repeated calls on a single body.
        self.rider._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.asarray((force+push_n)*axis,dtype=np.float32).reshape(1,3),torques=np.asarray(np.cross(grip-point,push_n*axis),dtype=np.float32).reshape(1,3),positions=np.asarray(point,dtype=np.float32).reshape(1,3),is_global=True)
        apply(self.beam,-force*axis,point)
        return dict(position_m=q,velocity_m_s=v,anchor_m=self.anchor,latch_force_n=force,spring_force_n=spring,push_force_n=push_n,grip_world_m=grip.tolist())
