"""Opt-in private PR2 measurements. Never feeds the task recorder or controller.

The shared contact ledger retains impulse magnitude, not its normal component.
This independent subscription preserves the native points before that reduction.
"""
import hashlib
import json
import math
from pathlib import Path

from .mechanism import KNOB, ROOT


def _vector(value):
    if value is None:
        return None
    out = [float(x) for x in value]
    return out if len(out) == 3 and all(math.isfinite(x) for x in out) else None


def contact_point(point, dt):
    """Impulse projection / actual physics dt; absent normal means unknown force.

    Preserve signed projection too: magnitude does not assume which collider the
    native normal points towards. Tangential impulse is never called normal force.
    """
    normal, impulse = _vector(getattr(point, 'normal', None)), _vector(getattr(point, 'impulse', None))
    projection = None
    if normal is not None and impulse is not None:
        length = math.hypot(*normal)
        if length > 0:
            projection = sum(a * b for a, b in zip(normal, impulse)) / length
    return dict(position_m=_vector(getattr(point, 'position', None)), normal=normal, impulse_Ns=impulse,
                separation_m=float(point.separation), normal_impulse_signed_Ns=projection,
                normal_force_magnitude_N=None if projection is None else abs(projection) / dt)


def _safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [_safe(x) for x in value]
    except TypeError:
        return str(value)


def usd_readback(stage):
    """Resolved USD values + authored property opinions; no USD/PhysX writes.

    USD defaults are labelled separately from authored opinions. This is not a
    claim about unexposed native solver internals or effective material mixing.
    """
    from pxr import UsdPhysics, UsdShade
    prims = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        task = path == ROOT or path.startswith(ROOT + '/')
        robot = path.startswith(('/World/RobotLeft/', '/World/RobotRight/'))
        if not task and not robot:
            continue
        if robot and not (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdPhysics.Joint)
                          or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            continue
        values = {}
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(('physics:', 'physx', 'drive:', 'state:')):
                values[attr.GetName()] = dict(value=_safe(attr.Get()), authored=attr.HasAuthoredValueOpinion(),
                    opinions=[dict(layer=p.layer.identifier, value=_safe(p.default)) for p in attr.GetPropertyStack()])
        rels = {rel.GetName(): [str(x) for x in rel.GetTargets()] for rel in prim.GetRelationships()
                if rel.GetName().startswith(('physics:', 'physx', 'material:binding'))}
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        friction = None
        if material:
            friction = {a.GetName(): _safe(a.Get()) for a in material.GetPrim().GetAttributes()
                        if a.GetName().startswith(('physics:', 'physx'))}
        prims[path] = dict(type=prim.GetTypeName(), schemas=list(prim.GetAppliedSchemas()), attributes=values,
                           relationships=rels, physics_material=str(material.GetPath()) if material else None,
                           material_attributes=friction)
    return dict(schema='mind2act_eval.PR2.usd_readback.v1', prims=prims,
                root_layer_sha256=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest(),
                note='Resolved USD readback, not a native solver/material-combine measurement')


class ContactProbe:
    def __init__(self, stage, directory, get_dt, *, subscribe, decode_path):
        self.stage, self.get_dt, self.decode_path = stage, get_dt, decode_path
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / 'contacts.jsonl').open('x', encoding='utf-8')
        self.motion = (self.directory / 'motion.jsonl').open('x', encoding='utf-8')
        self.tick = None
        self._snapshot('before-first-step.json')
        self.subscription = subscribe(self.receive)

    def _snapshot(self, name):
        (self.directory / name).write_text(json.dumps(usd_readback(self.stage), indent=2) + '\n', encoding='utf-8')

    def before_step(self, tick):
        self.tick = tick

    def receive(self, headers, data):
        dt = float(self.get_dt())
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('PR2 diagnostic requires measured positive physics dt')
        for header in headers:
            pair = [str(self.decode_path(header.collider0)), str(self.decode_path(header.collider1))]
            # Include unintended knob/fixture contacts, as well as either arm's tips.
            if not any(p == KNOB or p.startswith(KNOB + '/') for p in pair):
                continue
            points = [contact_point(p, dt) for p in data[
                header.contact_data_offset:header.contact_data_offset + header.num_contact_data]]
            self.stream.write(json.dumps(dict(tick_before_step=self.tick, dt_s=dt, pair=pair,
                event_type=str(header.type), points=points)) + '\n')
        self.stream.flush()

    def after_step(self, tick):
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        transforms = {}
        paths = [KNOB] + ['/World/Robot' + side + '/Asset/' + link
                         for side in ('Left', 'Right') for link in ('link6', 'link7', 'link8')]
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim:
                transforms[path] = _safe(cache.GetLocalToWorldTransform(prim))
        self.motion.write(json.dumps(dict(tick=tick, world_transforms=transforms)) + '\n')
        self.motion.flush()

    def close(self):
        self.subscription = None
        self._snapshot('after-last-step.json')
        self.stream.close()
        self.motion.close()


def native_probe(ctx):
    from pxr import PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface
    return ContactProbe(ctx.sim.stage, Path(ctx.output_dir) / 'private' / 'PR2' / 'contact-diagnostic',
                        ctx.sim.world.get_physics_dt,
                        subscribe=get_physx_simulation_interface().subscribe_contact_report_events,
                        decode_path=PhysicsSchemaTools.intToSdfPath)
