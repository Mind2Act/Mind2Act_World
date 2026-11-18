"""One explicit PR2 acquisition contact exception; no general X5 override API."""
from copy import deepcopy
import json
import math
from pathlib import Path

from mind2act_eval.contracts import to_json
from .task_collision_overlay import ROOTS, digest, protected_values, snapshot, verify_official_parameters

DECISION = 'Preserve the task fingertip contact material.'
SOURCE_WORLD_SHA256 = '76b316fd45f5461e8c49a5eeb7c993ea2f9d0a7187e7b7b30404ed4d2ffcfbd2'
COLLIDERS = tuple(root + '/link%d/collisions/link%d/node_STL_BINARY_' % (joint, joint)
                  for root in ROOTS for joint in (7, 8))
CONTACT_ATTRS = ('physxCollision:contactOffset', 'physxCollision:restOffset')
CONTACT_REL = 'material:binding:physics'


def expected_descriptor(spec):
    from importlib import resources
    from mind2act_eval.tasks.PR2.instances import CONFIG_VERSION
    layout = json.loads(resources.files('mind2act_eval.tasks.PR2').joinpath('config', 'layout.json').read_text())
    if (spec.task_id != 'PR2' or spec.difficulty not in layout['source_ids']
            or spec.source_id != layout['source_ids'][spec.difficulty]
            or spec.config_version != CONFIG_VERSION or spec.clock.physics_hz != 120):
        raise ValueError('PR2 contacts require the active selected acquisition source')
    expected = dict(decision=DECISION, source_world_sha256=SOURCE_WORLD_SHA256,
                    material_path='/World/PrecisionContactMaterial', static_friction=.9,
                    dynamic_friction=.7, restitution=0., contact_offset_m=.00015, rest_offset_m=0.)
    physical = {k: v for k, v in expected.items() if k not in ('decision', 'source_world_sha256')}
    if any(layout.get('fingertip_contact_exception', {}).get(k) != v for k, v in physical.items()):
        raise ValueError('PR2 contact configuration differs from the approved source exception')
    return dict(schema='mind2act.pr2-source-contacts.v1', task_id='PR2',
                colliders=list(COLLIDERS), **expected)


def _targets(stage, descriptor):
    from pxr import UsdPhysics, UsdShade
    material = UsdShade.Material(stage.GetPrimAtPath(descriptor['material_path']))
    if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('PR2 source contact material is missing')
    api = UsdPhysics.MaterialAPI(material.GetPrim())
    for attr, name in [(api.GetStaticFrictionAttr(), 'static_friction'),
                       (api.GetDynamicFrictionAttr(), 'dynamic_friction'),
                       (api.GetRestitutionAttr(), 'restitution')]:
        value = attr.Get()
        if value is None or not math.isclose(float(value), descriptor[name], rel_tol=0, abs_tol=1e-7):
            raise ValueError('PR2 source contact material differs')
    prims = [stage.GetPrimAtPath(path) for path in descriptor['colliders']]
    if any(not p or not p.IsActive() or not p.HasAPI(UsdPhysics.CollisionAPI) for p in prims):
        raise ValueError('PR2 requires all four existing source fingertip colliders')
    return material, prims


def validate_actual(stage, descriptor):
    from pxr import UsdShade
    _, prims = _targets(stage, descriptor)
    for prim in prims:
        targets = prim.GetRelationship(CONTACT_REL).GetTargets()
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        if list(map(str, targets)) != [descriptor['material_path']] or str(material.GetPath()) != descriptor['material_path']:
            raise ValueError('PR2 source fingertip material binding changed')
        for name, expected in zip(CONTACT_ATTRS, (descriptor['contact_offset_m'], descriptor['rest_offset_m'])):
            value = prim.GetAttribute(name).Get()
            if value is None or not math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-9):
                raise ValueError('PR2 source fingertip contact offset changed')


class SourceContactGuard:
    def __init__(self, stage, descriptor):
        verify_official_parameters(stage)
        self.descriptor = deepcopy(descriptor)
        self.before = snapshot(stage)
        self.protected = protected_values(stage)
        self.handles = {path: stage.GetAttributeAtPath(path) for path in self.protected}

    def authorize_after(self, stage):
        after = snapshot(stage)
        if set(after) != set(self.before):
            raise ValueError('PR2 contact exception cannot add or remove robot geometry')
        for path, before in self.before.items():
            old, new = deepcopy(before), deepcopy(after[path])
            if path in self.descriptor['colliders']:
                for row in (old, new):
                    for attr in CONTACT_ATTRS: row['attrs'].pop(attr, None)
                    row['relationships'].pop(CONTACT_REL, None)
                    row['schemas'] = [s for s in row['schemas'] if s not in ('PhysxCollisionAPI', 'MaterialBindingAPI')]
            if old != new:
                raise ValueError('PR2 contact exception changed unauthorized X5 semantics: ' + path)
        self.verify(stage)
        return after

    def verify(self, stage):
        validate_actual(stage, self.descriptor)
        if {path: repr(attr.Get()) for path, attr in self.handles.items()} != self.protected:
            raise ValueError('official X5 drives, solver or limits changed after PR2 contacts')


def _write_snapshot_summary(path, values):
    """Bound disk evidence without reducing the guard's full in-memory snapshot.

    Mesh arrays can make each full snapshot hundreds of MB. No runtime reader
    consumes those JSON values; the receipt binds the complete snapshot hash.
    """
    summary = dict(schema='mind2act.robot_snapshot_summary.v1',
        snapshot_sha256=digest(values), full_values_stored=False,
        prim_count=len(values),
        attribute_count=sum(len(p['attrs']) for p in values.values()),
        relationship_count=sum(len(p['relationships']) for p in values.values()),
        time_sample_count=sum(len(a['samples']) for p in values.values() for a in p['attrs'].values()))
    path.write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    return summary['snapshot_sha256']


def apply(stage, spec, descriptor, output_dir):
    from pxr import Sdf, UsdShade
    expected = expected_descriptor(spec)
    if any(descriptor.get(k) != v for k, v in expected.items() if k not in ('decision', 'source_world_sha256')):
        raise ValueError('arbitrary PR2 contact descriptors are forbidden')
    material, prims = _targets(stage, expected)  # Validate every target before writing any.
    guard = SourceContactGuard(stage, expected)
    output = Path(output_dir) / 'robot-contact-overlay' / 'PR2'
    output.mkdir(parents=True, exist_ok=False)
    for prim in prims:
        prim.AddAppliedSchema('PhysxCollisionAPI')
        prim.CreateAttribute(CONTACT_ATTRS[0], Sdf.ValueTypeNames.Float).Set(expected['contact_offset_m'])
        prim.CreateAttribute(CONTACT_ATTRS[1], Sdf.ValueTypeNames.Float).Set(expected['rest_offset_m'])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    after = guard.authorize_after(stage)
    snapshots = {}
    for name, values in [('before', guard.before), ('after', after)]:
        snapshots[name] = _write_snapshot_summary(output / (name + '.json'), values)
    receipt = dict(schema='mind2act.pr2-source-contact-binding.v1', task_id=spec.task_id,
        instance_id=spec.instance_id, source_id=spec.source_id, decision=DECISION,
        source_sha256=SOURCE_WORLD_SHA256, descriptor_sha256=digest(expected),
        active_spec_sha256=digest(to_json(spec)), descriptor=expected,
        before_snapshot_sha256=snapshots['before'], after_snapshot_sha256=snapshots['after'],
        added_robot_prims=[], changed_robot_prims=[p for p in after if after[p] != guard.before[p]],
        official_drive_solver_limits_unchanged=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return guard, receipt
