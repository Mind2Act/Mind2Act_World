"""F1 button/slider mechanics. `Slider` is pure state (CPU-testable); USD authoring is in `author_*`.

Source: `cases/F/runtime/development_world.py` + `geometry.py` (f-v4-original-train-camera-v1). The slider is
an environment-driven kinematic motor, never robot-attached. Robot mount/cameras/drives/rest pose are owned
by robot/. Reset/reveal follows measured completion without additional observation or reveal waits.
"""
import re

import numpy as np

from .space import is_grid, moved, size, slider_xy, start_position


class Slider:
    def __init__(self, case, layout, press_threshold, release_threshold):
        self.case, self.lay = case, layout
        self.press, self.release = press_threshold, release_threshold
        self.grid = is_grid(case['space'])
        self.pos = start_position(case['space'])
        self.visual = np.array(self.pos, float)
        self.motor_from = self.motor_to = self.visual.copy()
        self.motor_tick = 0
        self.feedback_ticks = round(case['timing']['button_feedback_seconds'] * 120)
        if self.feedback_ticks <= 0:
            raise ValueError('Invalid F feedback timing')
        self.stage = 'exploration'
        self.armed = {b: True for b in case['buttons']}
        self.pending = False
        self.observed = {}

    def update(self, tick, depths, contacts, arm_pairs):
        """One physics tick after the step. Returns (button events, mechanism failures)."""
        events, failures = [], []
        for b in self.case['buttons']:
            if depths[b] < self.release:
                self.armed[b] = True
            if self.armed[b] and depths[b] >= self.press:
                if self.stage == 'exploration':
                    self.pending = True
                self.armed[b] = False
                before = list(self.pos) if self.grid else self.pos
                self.pos = moved(self.case['space'], self.case['mapping'], before, b)
                touching = [s for s in ('Left', 'Right') if arm_pairs[b][s]]
                events.append(dict(tick=tick, stage=self.stage, button=b, depression_m=depths[b], finger_contact=bool(contacts[b]),
                                   acting_arm=touching[0] if len(touching) == 1 else 'Ambiguous',
                                   contact_colliders=[p for s in touching for p in arm_pairs[b][s]],
                                   before=before, after=list(self.pos) if self.grid else self.pos))
                self.motor_from, self.motor_to, self.motor_tick = self.visual.copy(), np.array(self.pos, float), 0
                if self.stage == 'exploration':
                    self.observed[b] = True  # a boundary press with no movement still counts as a probe
        self.motor_tick += 1
        u = min(self.motor_tick / self.feedback_ticks, 1.)
        u = u * u * (3 - 2 * u)
        self.visual = self.motor_from + (self.motor_to - self.motor_from) * u
        released = all(depths[b] < self.release for b in self.case['buttons'])
        if self.pending and self.motion_complete() and released:
            self.pending = False
        return events, failures

    def motion_complete(self):
        return bool(np.allclose(self.visual, np.array(self.pos, float), atol=1e-6))

    def xy_world(self):
        return slider_xy(self.case['layout'], self.case['space'], self.visual.tolist() if self.grid else float(self.visual),
                         self.lay['slider_end_margin_m'])

    def ready_to_reset(self, released):
        return (set(self.observed) == set(self.case['buttons']) and self.motion_complete()
                and released and not self.pending)

    def begin_reset(self):
        self.stage = 'reset'
        self.pos = self.case['execution']['start']
        self.motor_from, self.motor_to, self.motor_tick = self.visual.copy(), np.array(self.pos, float), 0


class Timeline:
    """Environment reset/reveal automation driven by measured readiness only (source sdk_task.Timeline)."""

    def __init__(self, slider, layout, hold_ticks):
        self.slider, self.lay, self.hold = slider, layout, hold_ticks
        self.ready_since = None
        self.deadline = None

    def advance(self, tick, ready, show_goal):
        s = self.slider
        if s.stage == 'exploration' and ready:
            s.begin_reset()
        elif s.stage == 'reset' and s.motion_complete() and s.pos == s.case['execution']['start']:
            show_goal()
            s.stage = 'execution'
        elif s.stage == 'goal_reveal':
            # Archived stages may still include the old reveal interval.
            s.stage = 'execution'


def world_bounds(stage, path):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError('Missing F geometry: ' + path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'], useExtentsHint=False, ignoreVisibility=True)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise ValueError('Empty F geometry: ' + path)
    return bounds


def translate_world(stage, path, delta):
    from pxr import Gf, Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    parent = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim.GetParent())
    local_delta = parent.GetInverse().TransformDir(Gf.Vec3d(*delta))
    transform = UsdGeom.Xformable(prim)
    matrix = transform.GetLocalTransformation()
    matrix.SetTranslateOnly(matrix.ExtractTranslation() + local_delta)
    transform.MakeMatrixXform().Set(matrix)


def lift_easy_rail(stage, space, support_z, lay):
    """The per-case rail closure is authored at z=0 for the line layout, unlike the grid closure."""
    if space['kind'] != 'line':
        return dict(applied=False, reason='grid scene already uses world support height')
    floor = float(world_bounds(stage, '/World/RailBase').GetMin()[2])
    if abs(floor - support_z) < 1e-5:
        return dict(applied=False, reason='rail already on support')
    if abs(floor) > 1e-5:
        raise ValueError('F rail is neither support-relative nor world-aligned')
    pattern = re.compile(lay['rail_prim_pattern'])
    paths = [str(p.GetPath()) for p in stage.GetPrimAtPath('/World').GetChildren()
             if p.GetName() in lay['rail_prims'] or pattern.fullmatch(p.GetName())]
    for path in paths:
        translate_world(stage, path, [0, 0, support_z])
    return dict(applied=True, delta_z=support_z, components=paths)


def align_target(stage, xyz):
    """The target's children may carry baked world XY; move by the measured offset, not to xyz directly."""
    before = world_bounds(stage, '/World/Target').GetMidpoint()
    translate_world(stage, '/World/Target', [float(xyz[0] - before[0]), float(xyz[1] - before[1]), 0])
    after = world_bounds(stage, '/World/Target').GetMidpoint()
    if any(abs(float(after[i] - xyz[i])) > 1e-6 for i in (0, 1)):
        raise ValueError('F rendered target does not match logical target')


def author_buttons(stage, case, support_z, lay):
    from pxr import Usd, UsdPhysics, PhysxSchema, Gf
    for b, (x, y) in zip(case['buttons'], case['layout']['button_centers_xy']):
        root = '/World/Button' + b + '/Asset'
        prim = stage.GetPrimAtPath(root)
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateSolverPositionIterationCountAttr(lay['button_solver_iterations'])
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr(True)
                UsdPhysics.RigidBodyAPI(child).CreateKinematicEnabledAttr(False)
                PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.)
            if child.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(child).CreateJointEnabledAttr(True)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(True)
                api = PhysxSchema.PhysxCollisionAPI.Apply(child)
                api.CreateContactOffsetAttr(lay['button_contact_offset_m'])
                api.CreateRestOffsetAttr(0.)
        UsdPhysics.FixedJoint(stage.GetPrimAtPath(root + '/E_body_16/FixedJoint')).CreateLocalPos0Attr(
            Gf.Vec3f(x, y, support_z + lay['button_base_lift_m']))
        cap = stage.GetPrimAtPath(root + '/E_button_01_13')
        if cap.HasAttribute('xformOp:translate:press'):
            cap.GetAttribute('xformOp:translate:press').Set(Gf.Vec3d(0))
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(root + '/E_button_01_13/PrismaticJoint_orangeswitch_up'), 'linear')
        drive.CreateTargetPositionAttr(lay['button_drive_target_m'])


def author_colliders(stage):
    from pxr import UsdGeom, UsdPhysics
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith('/World/') or any(t in path for t in ['/Robot', '/Button', '/Looks', '/Target', '/Slider']):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')


def slider_world_z(case, support_z, lay):
    return support_z + lay['slider_height_m']['grid' if is_grid(case['space']) else 'line']


def subscribe_contacts(table):
    import math
    from pxr import PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType

    def report(headers, data):
        for h in headers:
            pair = tuple(str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1))
            points = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
            table.report(pair, h.type != ContactEventType.CONTACT_LOST,
                         max((math.hypot(*p.impulse) for p in points), default=0.),
                         min((float(p.separation) for p in points), default=1.))
    return get_physx_simulation_interface().subscribe_contact_report_events(report)
